# learners/pepita.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseLearner
from networks.snn_core import SNNCore


class PepitaLearner(BaseLearner):
    """
    PEPITA-like two-pass update with feedback matrix F.
    Forward returns accumulated logits over time.

    Additions vs a basic PEPITA implementation:
      - Data-driven calibration of F on the first batch so that
        the modulation (e @ F) has a fixed ratio to input magnitude.
      - Relative step-size control: each update ΔW is constrained so that
        ||ΔW|| / ||W|| <= max_rel_step.
    """

    def __init__(
        self,
        net_cfg,
        meta,
        device,
        mode: str = "original",
        lr: float = 0.01,
        max_rel_step: float = 0.05,
        target_modulation_ratio: float = 0.1,
    ):
        """
        Args:
            mode: "original" or "accum" (for memory estimation logic only).
            lr: base learning rate (scaled internally by relative step control).
            max_rel_step: max allowed relative step size per update
                          (||ΔW|| / ||W|| <= max_rel_step).
            target_modulation_ratio: target std((e @ F)) / std(X) on first batch.
        """
        super().__init__(net_cfg, meta, device)

        self.mode = mode
        self.lr = lr
        self.max_rel_step = max_rel_step
        self.target_modulation_ratio = target_modulation_ratio

        K = meta["n_classes"]
        D0 = net_cfg.layers[0].dim_in

        # Initialize F with approximately orthogonal rows/cols.
        F_mat = torch.empty(K, D0, device=device)
        nn.init.orthogonal_(F_mat)
        self.F = F_mat  # kept as a plain tensor; updated only via calibration

        # Will be set after first batch using real (X, e)
        self._F_calibrated = False

    # -------------------------------------------------------------------------
    # BaseLearner hooks
    # -------------------------------------------------------------------------
    def _build_model(self):
        return SNNCore(self.cfg, self.meta["n_classes"])

    # -------------------------------------------------------------------------
    # Helper: safe parameter update with relative step-size control
    # -------------------------------------------------------------------------
    def _apply_update(self, W: torch.Tensor, dW: torch.Tensor) -> None:
        """
        Apply an update to W using the 'gradient-like' dW, but enforce
        a bound on the relative step size: ||ΔW|| / ||W|| <= max_rel_step.
        """
        with torch.no_grad():
            update = self.lr * dW
            w_norm = W.norm()
            u_norm = update.norm()

            if w_norm > 0 and u_norm > self.max_rel_step * w_norm:
                # Scale down the update so that relative step is bounded.
                scale = (self.max_rel_step * w_norm) / (u_norm + 1e-8)
                update = update * scale

            W.add_(update)

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X:[B,T,D] -> logits:[B,K] (sum of head outputs over time)."""
        B, T, _ = X.shape
        X = X.to(self.device)
        state, head_mem = self.model.init_state(B, X.device, X.dtype)
        logits = torch.zeros(B, self.meta["n_classes"], device=X.device)

        for t in range(T):
            _, head_out, state, head_mem, _, _ = self.model.forward_step(
                X[:, t, :], state, head_mem
            )
            logits += head_out

        return logits

    # -------------------------------------------------------------------------
    # First-pass helper (collect spike sequences + logits)
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def _first_pass(self, X: torch.Tensor):
        B, T, _ = X.shape
        state, head_mem = self.model.init_state(B, X.device, X.dtype)
        L = len(self.model.fcs)
        h_rec = [[] for _ in range(L)]
        logits = torch.zeros(B, self.meta["n_classes"], device=X.device)

        for t in range(T):
            _, head_out, state, head_mem, layer_spikes, _ = self.model.forward_step(
                X[:, t, :], state, head_mem
            )
            for i, h in enumerate(layer_spikes):
                h_rec[i].append(h)
            logits += head_out

        # list of [T,B,H_l]
        h_seq = [torch.stack(seq, 0) for seq in h_rec]
        return {"h_seq": h_seq, "logits": logits}

    # -------------------------------------------------------------------------
    # Training step with PEPITA update + F calibration + relative step control
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def train_step(self, X: torch.Tensor, y: torch.Tensor):
        K = self.meta["n_classes"]
        X, y = X.to(self.device), y.to(self.device)
        B, T, _ = X.shape

        # ----- First pass: standard input -----
        fp = self._first_pass(X)
        logits = fp["logits"]

        # Cross-entropy and error signal at output
        p = torch.softmax(logits, dim=1)  # [B,K]
        e = p - F.one_hot(y, num_classes=K).float()  # [B,K]
        loss = F.cross_entropy(logits, y)

        # ----- One-time calibration of F using real data -----
        # Aim: std(e @ F) ≈ target_modulation_ratio * std(X)
        if not self._F_calibrated:
            delta = e @ self.F  # [B,D0]
            s_x = X.std()
            s_delta = delta.std()

            if s_delta > 0 and s_x > 0:
                scale = (self.target_modulation_ratio * s_x) / s_delta
                self.F.mul_(scale)

            self._F_calibrated = True

        # ----- Modulate input with calibrated F -----
        X_mod = X + (e @ self.F).unsqueeze(1)  # [B,T,D]

        # ----- Second pass: modulated input -----
        L = len(self.model.fcs)
        sp = self._first_pass(X_mod)

        h_seq = fp["h_seq"]          # list of [T,B,H_l]
        h_mod_seq = sp["h_seq"]      # list of [T,B,H_l]

        # ----- Layer 0 update -----
        diff0 = h_seq[0] - h_mod_seq[0]             # [T,B,H0]
        x_mod_T = X_mod.permute(1, 0, 2)            # [T,B,D]
        mult0 = diff0.unsqueeze(3) * x_mod_T.unsqueeze(2)  # [T,B,H0,D]
        dW0 = -mult0.sum(dim=(0, 1)) / max(1, B * T)       # [H0,D]

        self._apply_update(self.model.fcs[0].weight.data, dW0)

        # ----- Deeper layers -----
        for l in range(1, L):
            pre = h_mod_seq[l - 1]                  # [T,B,H_{l-1}]
            diff = h_seq[l] - h_mod_seq[l]          # [T,B,H_l]
            mult = diff.unsqueeze(3) * pre.unsqueeze(2)     # [T,B,H_l,H_{l-1}]
            dWl = -mult.sum(dim=(0, 1)) / max(1, B * T)     # [H_l,H_{l-1}]
            self._apply_update(self.model.fcs[l].weight.data, dWl)

        # ----- Readout layer update -----
        # use second-pass last layer average over time
        h_last_mod = h_mod_seq[-1].mean(0)          # [B,H_{L-1}]
        dWo = -(e.t() @ h_last_mod) / max(1, B)     # [K,H_{L-1}]
        self._apply_update(self.model.head.weight.data, dWo)

        # ----- Accuracy for logging -----
        acc = (logits.argmax(1) == y).float().mean().item() * 100.0
        return {"acc": acc, "loss": loss.item()}

    # -------------------------------------------------------------------------
    # Memory estimation
    # -------------------------------------------------------------------------
    def get_static_memory_bytes(self, fp_bytes: int = 4) -> int:
        """
        Network params (excluding unused recurrent weights) + feedback matrix F.
        """
        base = super().get_static_memory_bytes(fp_bytes=fp_bytes)
        F_elems = self.F.numel()
        return base + F_elems * fp_bytes

    def get_training_memory_bytes(
        self,
        batch: int,
        time_steps: int,
        fp_bytes: int = 4,
    ) -> int:
        """
        Approximate training memory:

        - mode='original': first pass stores activations over all time steps;
          second pass does not store activations.
        - mode='accum': assume an accumulated variant that only keeps per-layer
          running states (no T factor).
        """
        Hs = [fc.out_features for fc in self.model.fcs]
        sum_hidden = sum(Hs)

        if self.mode == "accum":
            # Running stats per layer, no temporal history.
            elems = batch * sum_hidden
        else:
            # 'original' PEPITA: store first-pass activations for all time steps.
            elems = batch * time_steps * sum_hidden

        return elems * fp_bytes
