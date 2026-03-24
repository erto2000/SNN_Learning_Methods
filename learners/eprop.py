from typing import List, Dict, Optional
import torch
import torch.nn.functional as F
from .base import BaseLearner
from networks.snn_core import SNNCore

class EpropLearner(BaseLearner):
    """
    E-Prop with single-pass online update (manual grads).
    Forward returns a logits vector per sample using cumulative rate code.
    """
    def __init__(
        self,
        net_cfg,
        meta: Dict,
        device: torch.device,
        lr_in: float = 5e-4,
        lr_rec: float = 5e-4,
        lr_out: float = 1e-3,
        drop_diag: bool = True,
        weight_clip: Optional[float] = 1.5,
    ):
        super().__init__(net_cfg, meta, device)

        # Disable autograd on model params; updates are manual
        for p in self.model.parameters():
            p.requires_grad = False

        self.rec_flags: List[bool] = [ls.recurrent for ls in net_cfg.layers]
        self.lr_in = lr_in
        self.lr_rec = lr_rec
        self.lr_out = lr_out
        self.drop_diag = drop_diag
        self.weight_clip = weight_clip

        if getattr(self.model, "head_lif", None) is not None:
            raise ValueError("EpropLearner expects cfg.head == 'logits' (linear head).")

    def _build_model(self):
        return SNNCore(self.cfg, self.meta["n_classes"])

    @staticmethod
    def _surrogate_fast_sigmoid(u: torch.Tensor, slope: float) -> torch.Tensor:
        sig = torch.sigmoid(slope * u)
        return slope * sig * (1.0 - sig)

    @torch.no_grad()
    def _clamp_rec(self):
        for l, W in enumerate(self.model.Wrecs):
            if not self.rec_flags[l]:
                continue
            if self.drop_diag:
                W.fill_diagonal_(0.0)
            if self.weight_clip is not None:
                W.clamp_(-self.weight_clip, self.weight_clip)

    # ------------ contract ------------
    @torch.no_grad()
    def forward(self, X: torch.Tensor, return_activity: bool = False):
        """
        X: [B,T,D] -> logits: [B,K] OR (logits, activity)
        Cumulative rate-code readout (matches train_step logic).
        """
        B, T, _ = X.shape
        X = X.to(self.device)
        state, head_mem = self.model.init_state(B, X.device, X.dtype)
        H_last = self.model.fcs[-1].out_features
        r_sum = torch.zeros(B, H_last, device=X.device)
        W_out = self.model.head.weight
        b_out = self.model.head.bias
        logits = torch.zeros(B, self.meta["n_classes"], device=X.device)

        layer_spike_counts = [0.0 for _ in self.model.fcs]

        for t in range(T):
            _, _, state, head_mem, layer_spikes, _ = self.model.forward_step(
                X[:, t, :], state, head_mem, need_pre=False
            )
            r_sum = r_sum + layer_spikes[-1]
            logits = r_sum @ W_out.T + (b_out if b_out is not None else 0.0)

            if return_activity:
                for i, spk in enumerate(layer_spikes):
                    layer_spike_counts[i] += float(spk.detach().sum().item())

        if not return_activity:
            return logits  # [B,K]

        activity = self._make_activity_dict(
            layer_spike_counts=layer_spike_counts,
            num_samples=B,
            num_timesteps=T,
        )
        return logits, activity

    @torch.no_grad()
    def train_step(self, X: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """
        Single-batch online update over sequence length T.

        Change vs your original version:
        - Eligibilities are still updated online at each time step.
        - But the loss / learning signal is computed ONCE at the end
          from the final cumulative rate code (r_sum).
        - This avoids summing CE gradients at every time step and
          keeps the scale of the update reasonable.
        """
        K = self.meta["n_classes"]
        beta = self.cfg.beta
        slope = self.cfg.slope
        th = self.cfg.threshold

        X = X.to(self.device)  # [B, T, D]
        y = y.to(self.device)  # [B]
        B, T, _ = X.shape

        state, head_mem = self.model.init_state(B, X.device, X.dtype)
        L = len(self.model.fcs)
        Hs = [fc.out_features for fc in self.model.fcs]
        in_dims = [self.model.fcs[0].in_features] + [fc.out_features for fc in self.model.fcs[:-1]]

        # eligibilities: e_ff:[B, in_l, H_l], e_rec:[B, H_l, H_l]
        e_ff  = [torch.zeros(B, in_dims[i], Hs[i], device=X.device) for i in range(L)]
        e_rec = [torch.zeros(B, Hs[i], Hs[i], device=X.device) if self.rec_flags[i] else None
                 for i in range(L)]

        # grads (same shapes as weights)
        dW_ff  = [torch.zeros_like(self.model.fcs[i].weight) for i in range(L)]
        dW_rec = [torch.zeros_like(self.model.Wrecs[i]) if self.rec_flags[i] else None
                  for i in range(L)]
        dW_out = torch.zeros_like(self.model.head.weight)
        db_out = torch.zeros_like(self.model.head.bias) if self.model.head.bias is not None else None

        # cumulative rate code from last hidden layer
        H_last = Hs[-1]
        r_sum = torch.zeros(B, H_last, device=X.device)

        # -------- 1) unroll in time: update eligibilities + r_sum only --------
        for t in range(T):
            # store previous membrane for surrogate gradient
            v_prev_list = [m.clone() for m in state.mems]

            # forward one step, get spikes + pre-activations
            _, _, state, head_mem, layer_spikes, pres = self.model.forward_step(
                X[:, t, :], state, head_mem, need_pre=True
            )

            z_last = layer_spikes[-1]
            r_sum = r_sum + z_last  # cumulative spike count (rate code)

            # surrogate derivatives for each layer
            psis = []
            for l, pre in enumerate(pres):
                u = beta * v_prev_list[l] + pre - th
                psis.append(self._surrogate_fast_sigmoid(u, slope))

            # update feedforward eligibilities
            e_ff[0] = beta * e_ff[0] + X[:, t, :].unsqueeze(2) * psis[0].unsqueeze(1)
            if self.rec_flags[0]:
                spk0 = layer_spikes[0]
                e_rec[0] = beta * e_rec[0] + spk0.unsqueeze(2) * psis[0].unsqueeze(1)

            for l in range(1, L):
                pre_l = layer_spikes[l - 1]
                e_ff[l] = beta * e_ff[l] + pre_l.unsqueeze(2) * psis[l].unsqueeze(1)
                if self.rec_flags[l]:
                    spk_l = layer_spikes[l]
                    e_rec[l] = beta * e_rec[l] + spk_l.unsqueeze(2) * psis[l].unsqueeze(1)

        # -------- 2) single loss / learning signal at final time --------
        W_out = self.model.head.weight
        b_out = self.model.head.bias
        logits = r_sum @ W_out.T + (b_out if b_out is not None else 0.0)  # [B,K]

        probs = torch.softmax(logits, dim=1)
        grad_logits = probs - F.one_hot(y, num_classes=K).float()  # [B,K]

        # layer-wise learning signals (backprop through static readout)
        L_sig: List[Optional[torch.Tensor]] = [None for _ in range(L)]
        L_sig[L - 1] = grad_logits @ W_out               # [B, H_last]
        for l in range(L - 1, 0, -1):
            W_l = self.model.fcs[l].weight               # [H_l, H_{l-1}]
            L_sig[l - 1] = L_sig[l] @ W_l               # [B, H_{l-1}]

        # -------- 3) compute weight gradients from eligibilities --------
        for l in range(L):
            # e_ff[l]: [B, in_l, H_l], L_sig[l]: [B, H_l]
            g_in_out = torch.einsum("bij,bj->ij", e_ff[l], L_sig[l])  # [in_l, H_l]
            dW_ff[l] = g_in_out.T  # [H_l, in_l] to match self.model.fcs[l].weight

            if self.rec_flags[l]:
                # e_rec[l]: [B, H_l, H_l]
                dW_rec[l] = torch.einsum("bij,bj->ij", e_rec[l], L_sig[l])  # [H_l, H_l]

        dW_out = grad_logits.T @ r_sum  # [K,H_last]
        if db_out is not None:
            db_out = grad_logits.sum(dim=0)

        # -------- 4) apply updates (normalize by batch size) --------
        norm = max(1, B)
        for l in range(L):
            self.model.fcs[l].weight.add_(-(self.lr_in / norm) * dW_ff[l])
        for l in range(L):
            if self.rec_flags[l]:
                self.model.Wrecs[l].add_(-(self.lr_rec / norm) * dW_rec[l])

        self.model.head.weight.data.add_(-(self.lr_out / norm) * dW_out)
        if db_out is not None:
            self.model.head.bias.data.add_(-(self.lr_out / norm) * db_out)

        # keep recurrent weights sane
        self._clamp_rec()

        final_loss = F.cross_entropy(logits, y, reduction="mean").item()
        acc = (logits.argmax(1) == y).float().mean().item() * 100.0

        return {"loss": final_loss, "acc": acc}

    def get_training_memory_bytes(self, batch: int, time_steps: int, fp_bytes: int = 4) -> int:
        """
        Approx: eligibility traces (feedforward + recurrent).
        No activation history; ignores tiny state buffers.
        """
        Hs = [fc.out_features for fc in self.model.fcs]
        in_dims = [self.model.fcs[0].in_features] + [fc.out_features for fc in self.model.fcs[:-1]]

        # e_ff: [B, in_l, H_l] per layer
        e_ff_elems = sum(batch * din * hout for din, hout in zip(in_dims, Hs))

        # e_rec: [B, H_l, H_l] for recurrent layers only
        e_rec_elems = sum(
            batch * h * h
            for flag, h in zip(self.rec_flags, Hs)
            if flag
        )

        return (e_ff_elems + e_rec_elems) * fp_bytes