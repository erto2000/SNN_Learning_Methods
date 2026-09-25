import dataclasses
import torch
import torch.nn.functional as F
from .base import BaseLearner
from networks.snn_core import SNNCore
from networks.specs import NetConfig


class FFLearner(BaseLearner):
    """
    Forward-Forward (greedy, layerwise). Forward returns class scores (goodness)
    per sample without exposing temporal internals.

    Each layer uses normalized input currents and a local contrastive loss.
    Positive and negative examples always carry different class labels.
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
        optimizer: str = "adam",
        adam_eps: float = 1e-8
    ):
        if meta["n_classes"] < 2:
            raise ValueError("FFLearner needs at least two classes for negative labels.")
        if any(layer.bias for layer in net_cfg.layers):
            raise ValueError("FFLearner requires HIDDEN_BIAS=False.")
        if any(layer.recurrent or layer.norm is not None for layer in net_cfg.layers):
            raise ValueError("FFLearner requires RECURRENT=False and NORM=None.")
        # augment first layer input with K label channels
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

        L = len(self.model.fcs)
        self.epochs_per_layer = max(1, total_epochs // max(1, L))

        optimizer = optimizer.lower()
        self.layer_opts = []
        for i in range(L):
            params = [self.model.fcs[i].weight]
            if optimizer == "adam":
                self.layer_opts.append(torch.optim.Adam(params, lr=lr, eps=adam_eps))
            elif optimizer == "sgd":
                self.layer_opts.append(torch.optim.SGD(params, lr=lr))
            else:
                raise ValueError(f"Unknown optimizer: {optimizer}")

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
                p.requires_grad = False

    # ---------- utilities ----------

    def _normalize_current(self, cur: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        """
        Your original per-sample L2 normalization + gain.
        This is actually very FF-friendly, so we keep it.
        """
        return cur / (cur.norm(p=2, dim=1, keepdim=True) + eps) * self.gain

    def _add_label_channels(self, x_btD: torch.Tensor, y_b: torch.Tensor) -> torch.Tensor:
        B, T, D = x_btD.shape
        C = self.meta["n_classes"]
        hot = F.one_hot(y_b, num_classes=C).to(x_btD.dtype)   # [B, C]
        s = x_btD.amax(dim=(1, 2)).clamp_min(1e-6)           # [B]
        hot = hot * s.unsqueeze(1)                           # scale
        lbl = hot.unsqueeze(1).expand(B, T, C)               # [B, T, C]
        return torch.cat([x_btD, lbl], dim=-1)               # [B, T, D+C]

    def _ff_step_collect(self, X_t: torch.Tensor, mems: list[torch.Tensor]):
        """
        One time-step forward through *all* layers, returning spikes and new mems.
        """
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

    def _goodness_scores(self, X_lbl: torch.Tensor, return_activity: bool = False):
        """
        Compute class goodness for label-conditioned inputs.
        Returns:
          - goodness [B] if return_activity=False
          - (goodness [B], activity dict) if return_activity=True
        """
        B, T, _ = X_lbl.shape
        device = X_lbl.device
        L = len(self.model.fcs)

        mems = [
            torch.zeros(
                B,
                self.model.fcs[i].out_features,
                device=device,
                dtype=X_lbl.dtype,
            )
            for i in range(L)
        ]
        spk_sums = [torch.zeros_like(m) for m in mems]
        layer_spike_counts = [0.0 for _ in range(L)]

        for t in range(T):
            layer_spikes, mems = self._ff_step_collect(X_lbl[:, t, :], mems)
            for i in range(L):
                spk_sums[i] += layer_spikes[i]
                if return_activity:
                    layer_spike_counts[i] += float(layer_spikes[i].detach().sum().item())

        goodness = sum((s ** 2).mean(1) for s in spk_sums)  # [B]

        if not return_activity:
            return goodness

        activity = self._make_activity_dict(
            layer_spike_counts=layer_spike_counts,
            num_samples=B,
            num_timesteps=T,
        )
        return goodness, activity

    # ------------ contract ------------

    @torch.no_grad()
    def forward(self, X: torch.Tensor, return_activity: bool = False):
        """
        X: [B, T, D] -> class scores [B, K] OR (scores [B, K], activity).
        Uses all layers' goodness (sum), just like before.
        """
        N, T, D = X.shape
        X = X.to(self.device)
        C = self.meta["n_classes"]
        device = X.device

        # Evaluate every class-conditioned copy
        X_rep = X.unsqueeze(1).expand(N, C, T, D).reshape(N * C, T, D)
        labels = torch.arange(C, device=device).unsqueeze(0).expand(N, C).reshape(-1)
        X_lbl = self._add_label_channels(X_rep, labels)  # [N*C, T, D+K]

        if not return_activity:
            scores_flat = self._goodness_scores(X_lbl, return_activity=False)
            return scores_flat.view(N, C)

        scores_flat, activity = self._goodness_scores(X_lbl, return_activity=True)

        activity = dict(activity)
        activity["num_samples"] = int(N)

        activity["ff_label_conditioned"] = True
        activity["ff_effective_batch_multiplier"] = int(C)
        activity["ff_internal_num_samples"] = int(activity.get("num_samples", N * C))

        return scores_flat.view(N, C), activity

    # ------------- training -------------

    def _goodness_for_layer(self, X_lbl: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        FF goodness for a single layer, using proper temporal unrolling
        through all layers up to `layer_idx`.

        X_lbl: [B, T, D+K]
        Returns: [B]
        """
        if X_lbl.dim() != 3:
            raise ValueError("X_lbl must be [B, T, D]")

        B, T, _ = X_lbl.shape
        device = X_lbl.device

        # Layer 0: keep your original behavior
        if layer_idx == 0:
            mem = torch.zeros(
                B,
                self.model.fcs[0].out_features,
                device=device,
                dtype=X_lbl.dtype,
            )
            spk_sum = torch.zeros_like(mem)
            for t in range(T):
                pre = self._normalize_current(self.model.fcs[0](X_lbl[:, t, :]))
                spk, mem = self.model.lifs[0](pre, mem)
                spk_sum += spk
            return (spk_sum ** 2).mean(dim=1)

        # Deeper layers: unroll through layers 0..layer_idx over time,
        # and accumulate spikes at layer_idx.
        mems = [
            torch.zeros(
                B,
                self.model.fcs[i].out_features,
                device=device,
                dtype=X_lbl.dtype,
            )
            for i in range(layer_idx + 1)
        ]
        spk_sum = torch.zeros(
            B,
            self.model.fcs[layer_idx].out_features,
            device=device,
            dtype=X_lbl.dtype,
        )

        for t in range(T):
            h = X_lbl[:, t, :]
            for i in range(layer_idx + 1):
                pre = self._normalize_current(self.model.fcs[i](h))
                spk, mems[i] = self.model.lifs[i](pre, mems[i])
                h = spk
            # h now is spikes of `layer_idx`
            spk_sum += h

        return (spk_sum ** 2).mean(dim=1)

    def train_step(self, X: torch.Tensor, y: torch.Tensor) -> dict:
        """
        Greedy layer-wise FF training (your original logic),
        now with a better `_goodness_for_layer` for deeper layers.
        """
        layer_idx = self.current_layer
        opt = self.layer_opts[layer_idx]

        X, y = X.to(self.device), y.to(self.device)

        # Positive / negative batches
        X_pos = self._add_label_channels(X, y)
        # A permutation can leave labels unchanged, especially in binary or
        # single-class batches. Draw uniformly from the other K-1 labels.
        offsets = torch.randint(1, self.meta["n_classes"], y.shape, device=y.device)
        y_neg = (y + offsets) % self.meta["n_classes"]
        X_neg = self._add_label_channels(X, y_neg)

        Gpos = self._goodness_for_layer(X_pos, layer_idx)
        Gneg = self._goodness_for_layer(X_neg, layer_idx)
        delta = Gpos - Gneg
        loss = F.softplus(-self.alpha * delta).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # Accuracy via forward()
        with torch.no_grad():
            preds = self.predict_batch(X)
            acc = (preds == y).float().mean().item() * 100.0

        return {
            "layer": layer_idx,
            "loss": loss.item(),
            "acc": acc,
        }

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

    def _cost_components(self, batch: int, time_steps: int, dims: dict) -> dict:
        B, T = int(batch), int(time_steps)
        A, U = dims["A"], dims["U"]
        d = dims["d"]
        a = dims["a"]
        tilde_inputs = dims["tilde_inputs"]
        f_l = dims["f_l"]
        p_l = dims["p_l"]
        L = dims["L"]
        d0_eff = dims["tilde_d0"]

        state_terms = [B * sum(d[:k]) + B * T * d[k] for k in range(L)] if L else [0]
        intermediate_terms = [B * T * (tilde_inputs[k] + d[k]) for k in range(L)] if L else [0]
        grad_terms = list(a) if a else [0]

        repeated_forward = sum((L - l + 1) * f_l[l - 1] for l in range(1, L + 1))
        repeated_compute = sum((L - l + 2) * (a[l - 1] + d[l - 1]) for l in range(1, L + 1))

        return {
            "memory": {
                "input": B * T * d0_eff,
                "param": A,
                "state": max(state_terms),
                "intermediate": max(intermediate_terms),
                "grad": max(grad_terms),
            },
            "compute": {
                "layerwise_positive_negative": 2 * B * T * repeated_compute,
            },
            "access": {
                "repeated_forward": 2 * B * T * repeated_forward,
                "temporal_neuron": 2 * B * T * U,
                "local_update": 2 * B * T * sum(p_l),
                "param_read_write": 2 * A,
            },
        }
