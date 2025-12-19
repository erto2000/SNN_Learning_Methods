import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .base import BaseLearner
from networks.snn_core import SNNCore
from utils.time_gating import make_time_weights


class PepitaLearner(BaseLearner):
    def __init__(
        self,
        net_cfg,
        meta,
        device,
        mode: str = "original",
        lr: float = 0.01,
        max_rel_step: float = 0.05,
        target_modulation_ratio: float = 0.1,
        time_gating: Optional[dict] = None,
    ):
        super().__init__(net_cfg, meta, device)

        self.mode = mode
        self.lr = lr
        self.max_rel_step = max_rel_step
        self.target_modulation_ratio = target_modulation_ratio

        self.time_gating = time_gating or {"enabled": False}

        K = meta["n_classes"]
        D0 = net_cfg.layers[0].dim_in

        F_mat = torch.empty(K, D0, device=device)
        nn.init.orthogonal_(F_mat)
        self.F = F_mat
        self._F_calibrated = False

    def _build_model(self):
        return SNNCore(self.cfg, self.meta["n_classes"])

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

    def _apply_update(self, W: torch.Tensor, dW: torch.Tensor) -> None:
        with torch.no_grad():
            update = self.lr * dW
            w_norm = W.norm()
            u_norm = update.norm()
            if w_norm > 0 and u_norm > self.max_rel_step * w_norm:
                scale = (self.max_rel_step * w_norm) / (u_norm + 1e-8)
                update = update * scale
            W.add_(update)

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, T, _ = X.shape
        X = X.to(self.device)
        state, head_mem = self.model.init_state(B, X.device, X.dtype)
        logits = torch.zeros(B, self.meta["n_classes"], device=X.device, dtype=X.dtype)

        w = self._time_weights(T, device=X.device, dtype=X.dtype)

        for t in range(T):
            _, head_out, state, head_mem, _, _ = self.model.forward_step(X[:, t, :], state, head_mem)
            logits += w[t] * head_out

        return logits

    @torch.no_grad()
    def _first_pass(self, X: torch.Tensor):
        B, T, _ = X.shape
        state, head_mem = self.model.init_state(B, X.device, X.dtype)
        L = len(self.model.fcs)
        h_rec = [[] for _ in range(L)]
        logits = torch.zeros(B, self.meta["n_classes"], device=X.device, dtype=X.dtype)

        w = self._time_weights(T, device=X.device, dtype=X.dtype)

        for t in range(T):
            _, head_out, state, head_mem, layer_spikes, _ = self.model.forward_step(X[:, t, :], state, head_mem)
            for i, h in enumerate(layer_spikes):
                h_rec[i].append(h)
            logits += w[t] * head_out

        h_seq = [torch.stack(seq, 0) for seq in h_rec]  # [T,B,H_l]
        return {"h_seq": h_seq, "logits": logits, "w": w}

    @torch.no_grad()
    def train_step(self, X: torch.Tensor, y: torch.Tensor):
        K = self.meta["n_classes"]
        X, y = X.to(self.device), y.to(self.device)
        B, T, _ = X.shape

        fp = self._first_pass(X)
        logits = fp["logits"]
        w = fp["w"]  # [T]

        p = torch.softmax(logits, dim=1)
        e = p - F.one_hot(y, num_classes=K).to(X.dtype)
        loss = F.cross_entropy(logits, y)

        if not self._F_calibrated:
            delta = e @ self.F
            s_x = X.std()
            s_delta = delta.std()
            if s_delta > 0 and s_x > 0:
                scale = (self.target_modulation_ratio * s_x) / s_delta
                self.F.mul_(scale)
            self._F_calibrated = True

        X_mod = X + (e @ self.F).unsqueeze(1)

        sp = self._first_pass(X_mod)
        h_seq = fp["h_seq"]
        h_mod_seq = sp["h_seq"]

        # Apply time weights to diffs: diff[t] *= w[t]
        w3 = w.view(T, 1, 1)

        # ----- Layer 0 -----
        diff0 = (h_seq[0] - h_mod_seq[0]) * w3  # [T,B,H0]
        x_mod_T = X_mod.permute(1, 0, 2)        # [T,B,D0]

        diff0_2d = diff0.reshape(T * B, -1)
        x_mod_2d = x_mod_T.reshape(T * B, -1)
        dW0 = -(diff0_2d.t() @ x_mod_2d) / max(1, B * T)
        self._apply_update(self.model.fcs[0].weight.data, dW0)

        # ----- Deeper layers -----
        L = len(self.model.fcs)
        for l in range(1, L):
            pre  = h_mod_seq[l - 1]                 # [T,B,H_prev]
            diff = (h_seq[l] - h_mod_seq[l]) * w3   # [T,B,H_l]

            diff_2d = diff.reshape(T * B, -1)
            pre_2d  = pre.reshape(T * B, -1)

            dWl = -(diff_2d.t() @ pre_2d) / max(1, B * T)
            self._apply_update(self.model.fcs[l].weight.data, dWl)

        # ----- Readout update (weighted mean over time) -----
        denom = w.sum().clamp_min(1e-8)
        h_last_mod = (h_mod_seq[-1] * w3).sum(0) / denom  # [B,H]
        dWo = -(e.t() @ h_last_mod) / max(1, B)
        self._apply_update(self.model.head.weight.data, dWo)

        acc = (logits.argmax(1) == y).float().mean().item() * 100.0
        return {"acc": acc, "loss": loss.item()}
