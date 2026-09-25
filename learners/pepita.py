import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseLearner
from networks.snn_core import SNNCore


class PepitaLearner(BaseLearner):
    """
    PEPITA-like two-pass update with feedback matrix F.
    Forward returns accumulated logits over time.

    Modes:
      - original:
          For every timestep t, compute the phase difference at that timestep
          and apply one update using the current timestep's presynaptic activity.

      - accum:
          Accumulate spike counts across time, divide by T to obtain spike rates,
          then update once using spike-rate differences between the two phases.

    Additions vs a basic PEPITA implementation:
      - Data-driven calibration of F on the first batch so that
        the modulation (e @ F) has a fixed ratio to input magnitude.
      - Optional relative step-size control: when max_rel_step is set,
        each update obeys ||ΔW|| / ||W|| <= max_rel_step.
    """

    VALID_MODES = {"original", "accum"}

    def __init__(
        self,
        net_cfg,
        meta,
        device,
        mode: str = "accum",
        lr: float = 1e-3,
        max_rel_step: float | None = None,
        target_modulation_ratio: float = 0.1,
        optimizer: str = "adam",
        adam_eps: float = 1e-8,
    ):
        """
        Args:
            mode:
                "original": apply one update per timestep using the timestep-wise
                            difference between phase-1 and phase-2 spikes.
                "accum":    apply one update using spike-rate differences, where
                            spike rate = accumulated spike count / T.
            lr: base learning rate.
            max_rel_step: optional relative step bound; None uses plain
                          optimizer steps (no post-update rescaling).
            target_modulation_ratio: target std((e @ F)) / std(X) on first batch.
            optimizer: parameter update rule applied to the PEPITA directions
                       ("adam" or "sgd").
        """
        if net_cfg.head != "logits":
            raise ValueError("PepitaLearner requires a linear logits head.")
        if net_cfg.head_bias or any(layer.bias for layer in net_cfg.layers):
            raise ValueError("PepitaLearner requires HIDDEN_BIAS=False and HEAD_BIAS=False.")
        if any(layer.recurrent or layer.norm is not None for layer in net_cfg.layers):
            raise ValueError("PepitaLearner requires RECURRENT=False and NORM=None.")
        super().__init__(net_cfg, meta, device)
        for param in self.model.parameters():
            param.requires_grad_(False)

        if mode not in self.VALID_MODES:
            raise ValueError(f"Unknown PepitaLearner mode: {mode!r}. Expected one of {sorted(self.VALID_MODES)}.")

        self.mode = mode
        self.lr = lr
        self.max_rel_step = max_rel_step
        self.target_modulation_ratio = target_modulation_ratio
        self.optimizer_name = optimizer.lower()

        # PEPITA constructs its own update directions. The optimizer consumes
        # those directions without using autograd through the two forward phases.
        opt_params = [fc.weight for fc in self.model.fcs]
        if self.model.head is not None:
            opt_params.append(self.model.head.weight)

        if self.optimizer_name == "adam":
            self.opt = torch.optim.Adam(opt_params, lr=self.lr, eps=adam_eps)
        elif self.optimizer_name == "sgd":
            self.opt = torch.optim.SGD(opt_params, lr=self.lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        K = meta["n_classes"]
        D0 = net_cfg.layers[0].dim_in

        # Feedback matrix from output error to input space.
        F_mat = torch.empty(K, D0, device=device)
        nn.init.uniform_(F_mat, a=-1.0, b=1.0)
        self.F = F_mat  # kept as a plain tensor; updated only via calibration

        # Will be set after first batch using real (X, e)
        self._F_calibrated = False

    # -------------------------------------------------------------------------
    # BaseLearner hooks
    # -------------------------------------------------------------------------
    def _build_model(self):
        return SNNCore(self.cfg, self.meta["n_classes"])

    # -------------------------------------------------------------------------
    # Helper: apply PEPITA directions with SGD or Adam
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def _apply_directions(self, directions) -> None:
        """
        Apply a collection of PEPITA update directions.

        Each item is ``(parameter, direction)``, where ``direction`` follows the
        original PEPITA convention ``W <- W + lr * direction``. PyTorch
        optimizers use ``W <- W - ... * grad``, so ``-direction`` is supplied
        as the optimizer gradient.

        ``max_rel_step`` is enforced on the actual parameter change produced by
        the selected optimizer, so the same stability constraint is available
        for both SGD and Adam.
        """
        if not directions:
            return

        snapshots = None
        if self.max_rel_step is not None:
            snapshots = [(param, param.detach().clone()) for param, _ in directions]

        self.opt.zero_grad(set_to_none=True)
        for param, direction in directions:
            param.grad = (-direction).detach()
        self.opt.step()

        if snapshots is None:
            return

        for param, old in snapshots:
            old_norm = old.norm()
            if old_norm <= 0:
                continue

            delta = param.detach() - old
            delta_norm = delta.norm()
            max_norm = self.max_rel_step * old_norm
            if delta_norm > max_norm:
                scale = max_norm / (delta_norm + 1e-8)
                param.copy_(old + delta * scale)

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, X: torch.Tensor, return_activity: bool = False):
        """X:[B,T,D] -> logits:[B,K] OR (logits, activity)."""
        B, T, _ = X.shape
        X = X.to(self.device)

        if self.F.dtype != X.dtype or self.F.device != X.device:
            self.F = self.F.to(device=X.device, dtype=X.dtype)

        state, head_mem = self.model.init_state(B, X.device, X.dtype)
        logits = torch.zeros(B, self.meta["n_classes"], device=X.device, dtype=X.dtype)

        layer_spike_counts = [0.0 for _ in self.model.fcs]

        for t in range(T):
            _, head_out, state, head_mem, layer_spikes, _ = self.model.forward_step(
                X[:, t, :], state, head_mem
            )
            logits += head_out

            if return_activity:
                for i, spk in enumerate(layer_spikes):
                    layer_spike_counts[i] += float(spk.detach().sum().item())

        if not return_activity:
            return logits

        activity = self._make_activity_dict(
            layer_spike_counts=layer_spike_counts,
            num_samples=B,
            num_timesteps=T,
        )
        return logits, activity

    # -------------------------------------------------------------------------
    # Pass helper: collect timestep spike sequences and/or spike rates
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def _run_pass(
        self,
        X: torch.Tensor,
        *,
        return_seq: bool = False,
        return_rate: bool = False,
    ):
        """
        Run one SNN phase over all timesteps.

        Args:
            X: [B,T,D]
            return_seq:
                If True, return per-timestep spikes as a list of tensors
                [T,B,H_l], one per layer.
            return_rate:
                If True, return spike rates as a list of tensors [B,H_l],
                one per layer, where rate = spike_count / T.

        Returns:
            dict with:
                logits: [B,K], accumulated over time.
                h_seq: optional list of [T,B,H_l].
                h_rate: optional list of [B,H_l].
        """
        B, T, _ = X.shape
        state, head_mem = self.model.init_state(B, X.device, X.dtype)
        L = len(self.model.fcs)

        logits = torch.zeros(B, self.meta["n_classes"], device=X.device, dtype=X.dtype)

        h_rec = [[] for _ in range(L)] if return_seq else None
        h_sum = (
            [
                torch.zeros(B, fc.out_features, device=X.device, dtype=X.dtype)
                for fc in self.model.fcs
            ]
            if return_rate
            else None
        )

        for t in range(T):
            _, head_out, state, head_mem, layer_spikes, _ = self.model.forward_step(
                X[:, t, :], state, head_mem
            )
            logits += head_out

            if return_seq:
                for i, h in enumerate(layer_spikes):
                    h_rec[i].append(h)

            if return_rate:
                for i, h in enumerate(layer_spikes):
                    h_sum[i].add_(h)

        out = {"logits": logits}

        if return_seq:
            out["h_seq"] = [torch.stack(seq, 0) for seq in h_rec]

        if return_rate:
            denom = max(1, T)
            out["h_rate"] = [h / denom for h in h_sum]

        return out

    # Backward-compatible alias for old internal calls.
    @torch.no_grad()
    def _first_pass(self, X: torch.Tensor):
        return self._run_pass(X, return_seq=True, return_rate=False)

    # -------------------------------------------------------------------------
    # Original mode update: timestep-wise differences, one update per timestep
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def _update_original(
        self,
        X_mod: torch.Tensor,
        h_seq,
        h_mod_seq,
        e: torch.Tensor,
    ) -> None:
        """
        Original PEPITA-style temporal update.

        For each timestep t:
          Δh_l(t) = h_l_phase1(t) - h_l_phase2(t)

        Then apply one update per timestep using the current timestep's
        presynaptic activity from the second/modulated phase.
        """
        B, T, _ = X_mod.shape
        L = len(self.model.fcs)

        if L == 0:
            raise RuntimeError("PepitaLearner expects at least one hidden FC layer.")

        denom = max(1, B)

        for t in range(T):
            directions = []

            # ----- Layer 0 update -----
            diff0_t = h_seq[0][t] - h_mod_seq[0][t]  # [B,H0]
            x_mod_t = X_mod[:, t, :]                 # [B,D0]
            dW0_t = -(diff0_t.t() @ x_mod_t) / denom # [H0,D0]
            directions.append((self.model.fcs[0].weight, dW0_t))

            # ----- Deeper layer updates -----
            for l in range(1, L):
                pre_t = h_mod_seq[l - 1][t]             # [B,H_{l-1}]
                diff_t = h_seq[l][t] - h_mod_seq[l][t]  # [B,H_l]
                dWl_t = -(diff_t.t() @ pre_t) / denom   # [H_l,H_{l-1}]
                directions.append((self.model.fcs[l].weight, dWl_t))

            # ----- Readout update -----
            # There is no hidden-layer phase difference at the head. Use the
            # same output error e with the current second-phase final-layer spikes.
            h_last_mod_t = h_mod_seq[-1][t]          # [B,H_{L-1}]
            dWo_t = -(e.t() @ h_last_mod_t) / denom  # [K,H_{L-1}]
            directions.append((self.model.head.weight, dWo_t))

            self._apply_directions(directions)

    # -------------------------------------------------------------------------
    # Accum mode update: spike-rate differences, one update per batch
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def _update_accum(
        self,
        X_mod: torch.Tensor,
        h_rate,
        h_mod_rate,
        e: torch.Tensor,
    ) -> None:
        """
        Accumulation-mode update.

        For every hidden layer l:
          r_l      = spike_count_l_phase1 / T
          r'_l     = spike_count_l_phase2 / T
          Δr_l     = r_l - r'_l

        Updates are computed from these spike-rate differences, not from
        per-timestep differences.
        """
        B, T, _ = X_mod.shape
        L = len(self.model.fcs)

        if L == 0:
            raise RuntimeError("PepitaLearner expects at least one hidden FC layer.")

        denom = max(1, B)

        # For layer 0, the presynaptic signal is the time-averaged modulated input.
        # Hidden layers use second-phase presynaptic spike rates.
        x_mod_rate = X_mod.mean(dim=1)               # [B,D0]

        directions = []

        # ----- Layer 0 update -----
        diff0_rate = h_rate[0] - h_mod_rate[0]       # [B,H0]
        dW0 = -(diff0_rate.t() @ x_mod_rate) / denom # [H0,D0]
        directions.append((self.model.fcs[0].weight, dW0))

        # ----- Deeper layer updates -----
        for l in range(1, L):
            pre_rate = h_mod_rate[l - 1]             # [B,H_{l-1}]
            diff_rate = h_rate[l] - h_mod_rate[l]    # [B,H_l]
            dWl = -(diff_rate.t() @ pre_rate) / denom
            directions.append((self.model.fcs[l].weight, dWl))

        # ----- Readout layer update -----
        # Use the second-phase final-layer spike rate.
        h_last_mod_rate = h_mod_rate[-1]             # [B,H_{L-1}]
        dWo = -(e.t() @ h_last_mod_rate) / denom     # [K,H_{L-1}]
        directions.append((self.model.head.weight, dWo))

        self._apply_directions(directions)

    # -------------------------------------------------------------------------
    # Training step with PEPITA update + F calibration + relative step control
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def train_step(self, X: torch.Tensor, y: torch.Tensor):
        K = self.meta["n_classes"]
        X, y = X.to(self.device), y.to(self.device)
        B, T, _ = X.shape

        if self.F.dtype != X.dtype or self.F.device != X.device:
            self.F = self.F.to(device=X.device, dtype=X.dtype)

        # ----- First pass: standard input -----
        if self.mode == "original":
            # Need timestep-wise spikes for per-timestep differences.
            fp = self._run_pass(X, return_seq=True, return_rate=False)
        elif self.mode == "accum":
            # Need spike rates only; no full time history is required.
            fp = self._run_pass(X, return_seq=False, return_rate=True)
        else:
            # Should be impossible because __init__ validates mode.
            raise RuntimeError(f"Invalid mode: {self.mode!r}")

        logits = fp["logits"]

        # Cross-entropy and error signal at output.
        p = torch.softmax(logits, dim=1)  # [B,K]
        e = p - F.one_hot(y, num_classes=K).to(p.dtype)
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
        input_modulation = e @ self.F               # [B,D0]
        X_mod = X + input_modulation.unsqueeze(1)   # [B,T,D0]

        # ----- Second pass and mode-specific update -----
        if self.mode == "original":
            sp = self._run_pass(X_mod, return_seq=True, return_rate=False)
            self._update_original(
                X_mod=X_mod,
                h_seq=fp["h_seq"],
                h_mod_seq=sp["h_seq"],
                e=e,
            )
        else:  # self.mode == "accum"
            sp = self._run_pass(X_mod, return_seq=False, return_rate=True)
            self._update_accum(
                X_mod=X_mod,
                h_rate=fp["h_rate"],
                h_mod_rate=sp["h_rate"],
                e=e,
            )

        # ----- Accuracy for logging -----
        acc = (logits.argmax(1) == y).float().mean().item() * 100.0
        return {"acc": acc, "loss": loss.item()}

    # -------------------------------------------------------------------------
    # Memory estimation
    # -------------------------------------------------------------------------
    def _cost_components(self, batch: int, time_steps: int, dims: dict) -> dict:
        B, T = int(batch), int(time_steps)
        A, U, V, C = dims["A"], dims["U"], dims["V"], dims["C"]
        d0 = dims["d0"]
        d0_eff = dims["tilde_d0"]
        F, P, Y = dims["F"], dims["P"], dims["Y"]

        if self.mode != "accum":
            raise RuntimeError("Estimated RATE-PEPITA costs are defined for mode='accum'.")

        return {
            "memory": {
                "input": B * T * d0_eff,
                "param": A + V,
                "state": B * U,
                # Retain accumulated rates from the original and perturbed phases.
                "rate_buffer": 2 * B * U,
                "feedback_matrix": C * d0,
                "output_error": B * C,
                "input_perturbation": B * d0,
                "grad": A + V,
            },
            "compute": {
                "two_forward_phases": 2 * B * T * (A + V + U),
                "feedback_projection_and_error": B * (C * d0 + C),
                "update": B * (A + V),
            },
            "access": {
                "two_forward_phases": 2 * B * T * (F + Y),
                "state_and_rate": 4 * B * T * U,
                "feedback_projection": B * (C * d0 + C + d0),
                "update": B * (P + Y),
                "param_read_write": 2 * (A + V),
            },
        }
