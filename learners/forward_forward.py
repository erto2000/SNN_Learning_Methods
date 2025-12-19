import dataclasses
import torch
import torch.nn.functional as F
from typing import Optional
from .base import BaseLearner
from networks.snn_core import SNNCore
from networks.specs import NetConfig
from utils.time_gating import make_time_weights


class FFLearner(BaseLearner):
    """
    Forward-Forward (greedy, layerwise) with optional time-gated spike sums.
    """
    def __init__(
        self,
        net_cfg: NetConfig,
        meta: dict,
        device: torch.device,
        alpha: float = 0.6,
        gain: float = 5.0,
        lr: float = 1e-3,
        total_epochs: int = 10,
        time_gating: Optional[dict] = None,
    ):
        layers = [dataclasses.replace(
            net_cfg.layers[0],
            dim_in=net_cfg.layers[0].dim_in + meta["n_classes"],
        )]
        layers += net_cfg.layers[1:]
        cfg = dataclasses.replace(net_cfg, head=None, layers=layers)
        super().__init__(cfg, meta, device)

        self.alpha = alpha
        self.gain  = gain
        self.lr    = lr

        self.time_gating = time_gating or {"enabled": False}

        L = len(self.model.fcs)
        self.epochs_per_layer = max(1, total_epochs // max(1, L))

        self.layer_opts = []
        for i in range(L):
            params = list(self.model.fcs[i].parameters()) + list(self.model.lifs[i].parameters())
            self.layer_opts.append(torch.optim.Adam(params, lr=lr))

        for fc in self.model.fcs:
            if fc.bias is not None:
                with torch.no_grad():
                    fc.bias.zero_()
                fc.bias.requires_grad = False

        self.current_layer = 0
        self.epoch_in_layer = 0
        self._refresh_requires_grad()

    def _build_model(self):
        return SNNCore(self.cfg, self.meta["n_classes"])

    def _refresh_requires_grad(self):
        L = len(self.model.fcs)
        for i in range(L):
            req = (i == self.current_layer)
            for p in self.model.fcs[i].parameters():
                p.requires_grad = (req and p is self.model.fcs[i].weight)
            for p in self.model.lifs[i].parameters():
                p.requires_grad = req

    def _time_weights(self, T: int, device, dtype) -> torch.Tensor:
        if not self.time_gating.get("enabled", False):
            return torch.ones(T, device=device, dtype=dtype)
        return make_time_weights(
            T,
            device=device,
            dtype=dtype,
            start_u=self.time_gating.get("start_u", 0.5),
            mode=self.time_gating.get("mode", "hard"),
            ramp_u=self.time_gating.get("ramp_u", 0.0),
            sharpness=self.time_gating.get("sharpness", 20.0),
        )

    # ---------- utilities ----------
    def _normalize_current(self, cur: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        return cur / (cur.norm(p=2, dim=1, keepdim=True) + eps) * self.gain

    def _add_label_channels(self, x_btD: torch.Tensor, y_b: torch.Tensor) -> torch.Tensor:
        B, T, D = x_btD.shape
        C = self.meta["n_classes"]
        hot = F.one_hot(y_b, num_classes=C).to(x_btD.dtype)
        s = x_btD.amax(dim=(1, 2)).clamp_min(1e-6)
        hot = hot * s.unsqueeze(1)
        lbl = hot.unsqueeze(1).expand(B, T, C)
        return torch.cat([x_btD, lbl], dim=-1)

    def _ff_step_collect(self, X_t: torch.Tensor, mems: list[torch.Tensor]):
        layer_spikes = []
        new_mems = []
        h = X_t
        for i, (fc, lif) in enumerate(zip(self.model.fcs, self.model.lifs)):
            pre = self._normalize_current(fc(h))
            spk, mem = lif(pre, mems[i])
            new_mems.append(mem)
            layer_spikes.append(spk)
            h = spk
        return layer_spikes, new_mems

    def _goodness_scores(self, X_lbl: torch.Tensor) -> torch.Tensor:
        B, T, _ = X_lbl.shape
        device = X_lbl.device
        L = len(self.model.fcs)

        mems = [torch.zeros(B, self.model.fcs[i].out_features, device=device, dtype=X_lbl.dtype) for i in range(L)]
        spk_sums = [torch.zeros_like(m) for m in mems]

        w = self._time_weights(T, device=device, dtype=X_lbl.dtype)

        for t in range(T):
            layer_spikes, mems = self._ff_step_collect(X_lbl[:, t, :], mems)
            wt = w[t]
            for i in range(L):
                spk_sums[i] += wt * layer_spikes[i]

        goodness = sum((s ** 2).mean(1) for s in spk_sums)
        return goodness

    # ------------ contract ------------
    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        N, T, D = X.shape
        C = self.meta["n_classes"]
        device = X.device

        X_rep = X.unsqueeze(1).expand(N, C, T, D).reshape(N * C, T, D)
        labels = torch.arange(C, device=device).unsqueeze(0).expand(N, C).reshape(-1)
        X_lbl = self._add_label_channels(X_rep, labels)

        scores_flat = self._goodness_scores(X_lbl)
        return scores_flat.view(N, C)

    # ------------- training -------------
    def _goodness_for_layer(self, X_lbl: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if X_lbl.dim() != 3:
            raise ValueError("X_lbl must be [B, T, D]")

        B, T, _ = X_lbl.shape
        device = X_lbl.device
        w = self._time_weights(T, device=device, dtype=X_lbl.dtype)

        if layer_idx == 0:
            mem = torch.zeros(B, self.model.fcs[0].out_features, device=device, dtype=X_lbl.dtype)
            spk_sum = torch.zeros_like(mem)
            for t in range(T):
                pre = self._normalize_current(self.model.fcs[0](X_lbl[:, t, :]))
                spk, mem = self.model.lifs[0](pre, mem)
                spk_sum += w[t] * spk
            return (spk_sum ** 2).mean(dim=1)

        mems = [torch.zeros(B, self.model.fcs[i].out_features, device=device, dtype=X_lbl.dtype)
                for i in range(layer_idx + 1)]
        spk_sum = torch.zeros(B, self.model.fcs[layer_idx].out_features, device=device, dtype=X_lbl.dtype)

        for t in range(T):
            h = X_lbl[:, t, :]
            for i in range(layer_idx + 1):
                pre = self._normalize_current(self.model.fcs[i](h))
                spk, mems[i] = self.model.lifs[i](pre, mems[i])
                h = spk
            spk_sum += w[t] * h

        return (spk_sum ** 2).mean(dim=1)

    def train_step(self, X: torch.Tensor, y: torch.Tensor) -> dict:
        layer_idx = self.current_layer
        opt = self.layer_opts[layer_idx]

        X, y = X.to(self.device), y.to(self.device)

        X_pos = self._add_label_channels(X, y)
        perm = torch.randperm(y.size(0), device=self.device)
        X_neg = self._add_label_channels(X, y[perm])

        Gpos = self._goodness_for_layer(X_pos, layer_idx)
        Gneg = self._goodness_for_layer(X_neg, layer_idx)

        delta = Gpos - Gneg
        loss = F.softplus(-self.alpha * delta).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        with torch.no_grad():
            preds = self.predict_batch(X)
            acc = (preds == y).float().mean().item() * 100.0

        return {"layer": layer_idx, "loss": loss.item(), "acc": acc}

    def on_epoch_end(self):
        L = len(self.model.fcs)
        state = {"layer": self.current_layer, "layer_advanced": False}
        self.epoch_in_layer += 1
        if self.epoch_in_layer >= self.epochs_per_layer and self.current_layer < L - 1:
            self.current_layer += 1
            self.epoch_in_layer = 0
            self._refresh_requires_grad()
            state.update({"layer": self.current_layer, "layer_advanced": True})
        return state
