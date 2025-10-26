import dataclasses
import torch
import torch.nn.functional as F
from .base import BaseLearner
from networks.snn_core import SNNCore
from networks.specs import NetConfig

class FFLearner(BaseLearner):
    """
    Forward-Forward (greedy, layerwise pretraining), single-batch update.
    - main() owns batching and epochs; call `on_epoch_end()` at epoch boundary.
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
    ):
        # 1) Bump the first layer input by +K for label channels; ensure no head.
        layers = [dataclasses.replace(net_cfg.layers[0],
                                      dim_in=net_cfg.layers[0].dim_in + meta["n_classes"])]
        layers += net_cfg.layers[1:]
        cfg = dataclasses.replace(net_cfg, head=None, layers=layers)
        super().__init__(cfg, meta, device)

        self.alpha = alpha
        self.gain  = gain
        self.lr    = lr

        L = len(self.model.fcs)
        self.epochs_per_layer = max(1, total_epochs // max(1, L))

        # Build per-layer optimizers; enable grads only for active layer
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

    # ---------------- base ----------------
    def _build_model(self):
        return SNNCore(self.cfg, self.meta["n_classes"])

    def _refresh_requires_grad(self):
        L = len(self.model.fcs)
        for i in range(L):
            req = (i == self.current_layer)
            for p in self.model.fcs[i].parameters(): p.requires_grad = (req and p is self.model.fcs[i].weight)
            for p in self.model.lifs[i].parameters(): p.requires_grad = req

    # ------------- FF utilities -------------
    def _normalize_current(self, cur: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        return cur / (cur.norm(p=2, dim=1, keepdim=True) + eps) * self.gain

    def _add_label_channels(self, x_btD: torch.Tensor, y_b: torch.Tensor) -> torch.Tensor:
        B,T,D = x_btD.shape
        C = self.meta["n_classes"]
        hot = F.one_hot(y_b, num_classes=C).to(x_btD.dtype)
        s = x_btD.amax(dim=(1,2)).clamp_min(1e-6)
        hot = hot * s.unsqueeze(1)
        lbl = hot.unsqueeze(1).expand(B,T,C)
        return torch.cat([x_btD, lbl], dim=-1)

    def _ff_step_collect(self, X_t: torch.Tensor, mems: list[torch.Tensor]):
        layer_spikes = []
        new_mems = []
        h = X_t
        for i, (fc, lif) in enumerate(zip(self.model.fcs, self.model.lifs)):
            pre = fc(h)
            pre = self._normalize_current(pre)
            spk, mem = lif(pre, mems[i])
            new_mems.append(mem)
            layer_spikes.append(spk)
            h = spk
        return layer_spikes, new_mems

    def _collapsed_input_for_layer(self, X_lbl: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if layer_idx == 0:
            return X_lbl  # [B,T,D+K]
        B,T,_ = X_lbl.shape
        h = X_lbl
        with torch.no_grad():
            for i in range(layer_idx):
                mems = [torch.zeros(B, self.model.fcs[j].out_features, device=X_lbl.device, dtype=X_lbl.dtype)
                        for j in range(len(self.model.fcs))]
                spk_sum = torch.zeros(B, self.model.fcs[i].out_features, device=X_lbl.device, dtype=X_lbl.dtype)
                for t in range(T):
                    layer_spikes, mems = self._ff_step_collect(h[:,t,:], mems)
                    spk_sum += layer_spikes[i]
                h = spk_sum
        return h

    def _goodness_for_layer(self, X_lbl: torch.Tensor, layer_idx: int) -> torch.Tensor:
        B = X_lbl.size(0)
        if layer_idx == 0 and X_lbl.dim() == 3:
            T = X_lbl.size(1)
            mem = torch.zeros(B, self.model.fcs[0].out_features, device=X_lbl.device, dtype=X_lbl.dtype)
            spk_sum = torch.zeros_like(mem)
            for t in range(T):
                pre = self.model.fcs[0](X_lbl[:,t,:])
                pre = self._normalize_current(pre)
                spk, mem = self.model.lifs[0](pre, mem)
                spk_sum += spk
            return (spk_sum**2).mean(dim=1)
        if X_lbl.dim() == 3:
            X_lbl = self._collapsed_input_for_layer(X_lbl, layer_idx)
        mem = torch.zeros(B, self.model.fcs[layer_idx].out_features, device=X_lbl.device, dtype=X_lbl.dtype)
        pre = self.model.fcs[layer_idx](X_lbl)
        pre = self._normalize_current(pre)
        spk, _ = self.model.lifs[layer_idx](pre, mem)
        return (spk**2).mean(dim=1)

    # ------------- public API -------------
    def train_step(self, X: torch.Tensor, y: torch.Tensor) -> dict:
        layer_idx = self.current_layer
        opt = self.layer_opts[layer_idx]

        X, y = X.to(self.device), y.to(self.device)
        # pos / neg
        X_pos = self._add_label_channels(X, y)
        perm  = torch.randperm(y.size(0), device=self.device)
        X_neg = self._add_label_channels(X, y[perm])

        Gpos = self._goodness_for_layer(X_pos, layer_idx)  # [B]
        Gneg = self._goodness_for_layer(X_neg, layer_idx)  # [B]
        delta = Gpos - Gneg
        loss = F.softplus(-self.alpha * delta).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        return {"layer": layer_idx, "loss": loss.item()}

    def on_epoch_end(self):
        """
        Call this from main() at the end of each epoch.
        Advances the greedy layer schedule based on epochs_per_layer.
        """
        L = len(self.model.fcs)
        state = {"layer": self.current_layer, "layer_advanced": False}
        self.epoch_in_layer += 1
        if self.epoch_in_layer >= self.epochs_per_layer and self.current_layer < L - 1:
            self.current_layer += 1
            self.epoch_in_layer = 0
            self._refresh_requires_grad()
            state.update({"layer": self.current_layer, "layer_advanced": True})
        return state

    @torch.no_grad()
    def predict_batch(self, X: torch.Tensor) -> torch.Tensor:
        N,T,D = X.shape
        C = self.meta["n_classes"]
        device = X.device
        X_rep = X.unsqueeze(1).expand(N, C, T, D).reshape(N*C, T, D)
        labels = torch.arange(C, device=device).unsqueeze(0).expand(N, C).reshape(-1)
        X_lbl = self._add_label_channels(X_rep, labels)  # [N*C, T, D+K]

        B = X_lbl.size(0)
        L = len(self.model.fcs)
        mems = [torch.zeros(B, self.model.fcs[i].out_features, device=device, dtype=X_lbl.dtype) for i in range(L)]
        spk_sums = [torch.zeros(B, self.model.fcs[i].out_features, device=device, dtype=X_lbl.dtype) for i in range(L)]
        for t in range(T):
            layer_spikes, mems = self._ff_step_collect(X_lbl[:,t,:], mems)
            for i in range(L):
                spk_sums[i] += layer_spikes[i]
        goodness = sum((s**2).mean(1) for s in spk_sums)  # [N*C]
        return goodness.view(N, C).argmax(1)
