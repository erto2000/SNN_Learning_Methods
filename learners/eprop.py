# learners/eprop.py
from typing import List, Dict, Optional
import torch
import torch.nn.functional as F

from .base import BaseLearner
from networks.snn_core import SNNCore


class EpropLearner(BaseLearner):
    """
    E-Prop with one-step spatial credit using the actual forward weights.

    Key points:
    - Recurrence is defined globally in NetConfig.LayerSpec.recurrent (per layer).
      This learner *does not* mutate network topology.
    - Single-batch `train_step` performs one online pass over T steps, accumulates
      eligibility traces and local learning signals, and applies an SGD update.
    - Readout uses a cumulative rate code over the last hidden layer spikes.

    Assumptions:
    - SNNCore(head="logits"): self.model.head is a nn.Linear producing logits.
    - SNNCore handles recurrence internally using cfg.layers[i].recurrent.
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

        # We update weights manually; disable autograd on model params
        for p in self.model.parameters():
            p.requires_grad = False

        # Per-layer recurrence flags sourced from the global NetConfig
        self.rec_flags: List[bool] = [ls.recurrent for ls in net_cfg.layers]

        self.lr_in = lr_in
        self.lr_rec = lr_rec
        self.lr_out = lr_out
        self.drop_diag = drop_diag
        self.weight_clip = weight_clip

        # Safety: this implementation expects a logits head (nn.Linear)
        if getattr(self.model, "head_lif", None) is not None:
            raise ValueError("EpropLearner expects cfg.head == 'logits' (linear head).")

    def _build_model(self):
        return SNNCore(self.cfg, self.meta["n_classes"])

    # --------- internal helpers ---------
    @staticmethod
    def _surrogate_fast_sigmoid(u: torch.Tensor, slope: float) -> torch.Tensor:
        # derivative of a fast-sigmoid surrogate at membrane potential u
        sig = torch.sigmoid(slope * u)
        return slope * sig * (1.0 - sig)

    @torch.no_grad()
    def _clamp_rec(self):
        # Clamp recurrent weights only for layers that are actually recurrent
        for l, W in enumerate(self.model.Wrecs):
            if not self.rec_flags[l]:
                continue
            if self.drop_diag:
                W.fill_diagonal_(0.0)
            if self.weight_clip is not None:
                W.clamp_(-self.weight_clip, self.weight_clip)

    # --------- public API ---------
    @torch.no_grad()
    def train_step(self, X: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """
        Single-batch online update over sequence length T.
        X: [B, T, D]
        y: [B]
        Returns: {"loss_per_step": float, "acc": float[%]}
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

        # Eligibilities
        # e_ff[l]: [B, in_l, H_l]
        e_ff = [torch.zeros(B, in_dims[i], Hs[i], device=X.device) for i in range(L)]
        # e_rec[l]: [B, H_l, H_l] if recurrent else None
        e_rec = [
            torch.zeros(B, Hs[i], Hs[i], device=X.device) if self.rec_flags[i] else None
            for i in range(L)
        ]

        # Accumulators for gradients
        dW_ff  = [torch.zeros_like(self.model.fcs[i].weight) for i in range(L)]  # [H_l, in_l]
        dW_rec = [
            torch.zeros_like(self.model.Wrecs[i]) if self.rec_flags[i] else None
            for i in range(L)
        ]  # [H_l, H_l]
        dW_out = torch.zeros_like(self.model.head.weight)  # [K, H_L]
        db_out = (
            torch.zeros_like(self.model.head.bias)
            if self.model.head.bias is not None
            else None
        )

        # Cumulative rate code for readout (last hidden layer)
        H_last = Hs[-1]
        r_sum = torch.zeros(B, H_last, device=X.device)
        last_logits = None  # for accuracy at the final step

        for t in range(T):
            # Preserve previous membranes for surrogate derivative psi
            v_prev_list = [m.clone() for m in state.mems]

            # One forward step (ask for pre-activations for psi)
            _, _ignored_logits, state, head_mem, layer_spikes, pres = self.model.forward_step(
                X[:, t, :], state, head_mem, need_pre=True
            )

            # Accumulate spikes from last hidden layer (rate code)
            z_last = layer_spikes[-1]         # [B, H_L]
            r_sum = r_sum + z_last            # [B, H_L]

            # Logits via cumulative rate code
            W_out = self.model.head.weight    # [K, H_L]
            b_out = self.model.head.bias
            logits_t = r_sum @ W_out.T + (b_out if b_out is not None else 0.0)  # [B, K]
            last_logits = logits_t

            # Surrogate psi per layer: u = beta * v_prev + pre - threshold
            psis: List[torch.Tensor] = []
            for l, pre in enumerate(pres):  # pre: [B, H_l]
                u = beta * v_prev_list[l] + pre - th
                psis.append(self._surrogate_fast_sigmoid(u, slope))  # [B, H_l]

            # Update eligibilities
            # Feedforward layer 0 uses input X_t
            e_ff[0] = beta * e_ff[0] + X[:, t, :].unsqueeze(2) * psis[0].unsqueeze(1)  # [B,in0,H0]
            if self.rec_flags[0]:
                spk0 = layer_spikes[0]  # [B, H0]
                e_rec[0] = beta * e_rec[0] + spk0.unsqueeze(2) * psis[0].unsqueeze(1)  # [B,H0,H0]

            # Higher layers use previous layer spikes
            for l in range(1, L):
                pre_l = layer_spikes[l - 1]  # [B, H_{l-1}]
                e_ff[l] = beta * e_ff[l] + pre_l.unsqueeze(2) * psis[l].unsqueeze(1)  # [B,in_l,H_l]
                if self.rec_flags[l]:
                    spk_l = layer_spikes[l]  # [B, H_l]
                    e_rec[l] = beta * e_rec[l] + spk_l.unsqueeze(2) * psis[l].unsqueeze(1)  # [B,H_l,H_l]

            # Local learning signal from readout (cross-entropy gradient wrt logits)
            probs_t = torch.softmax(logits_t, dim=1)
            grad_logits_t = probs_t - F.one_hot(y, num_classes=K).float()  # [B, K]

            # Top hidden learning signal
            L_sig: List[Optional[torch.Tensor]] = [None for _ in range(L)]
            L_sig[L - 1] = grad_logits_t @ W_out  # [B, H_L]

            # Propagate down via forward weights (NO transpose)
            for l in range(L - 1, 0, -1):
                W_l = self.model.fcs[l].weight  # [H_l, H_{l-1}]
                L_sig[l - 1] = L_sig[l] @ W_l   # [B, H_{l-1}]

            # Accumulate synaptic gradients
            for l in range(L):
                # e_ff[l]: [B, in_l, H_l], L_sig[l]: [B, H_l]
                g_in_out = torch.einsum("bij,bj->ij", e_ff[l], L_sig[l])  # [in_l, H_l]
                dW_ff[l] += g_in_out.T  # to [H_l, in_l]

            for l in range(L):
                if self.rec_flags[l]:
                    # e_rec[l]: [B, H_l, H_l]
                    dW_rec[l] += torch.einsum("bij,bj->ij", e_rec[l], L_sig[l])  # [H_l, H_l]

            # Readout gradients (use cumulative r_sum)
            dW_out += grad_logits_t.T @ r_sum  # [K, H_L]
            if db_out is not None:
                db_out += grad_logits_t.sum(dim=0)  # [K]

        # --------- SGD step (normalized by batch) ---------
        norm = max(1, B)
        for l in range(L):
            self.model.fcs[l].weight.add_(-(self.lr_in / norm) * dW_ff[l])
        for l in range(L):
            if self.rec_flags[l]:
                self.model.Wrecs[l].add_(-(self.lr_rec / norm) * dW_rec[l])
        self.model.head.weight.data.add_(-(self.lr_out / norm) * dW_out)
        if db_out is not None:
            self.model.head.bias.data.add_(-(self.lr_out / norm) * db_out)

        # Clamp recurrent weights if requested
        self._clamp_rec()

        # Compute loss from final step only
        final_loss = F.cross_entropy(last_logits, y, reduction="mean").item()

        # Final-step accuracy (argmax over last logits)
        acc = (last_logits.argmax(1) == y).float().mean().item() * 100.0

        return {
            "loss": final_loss,
            "acc": acc,
        }

    @torch.no_grad()
    def predict_batch(self, X: torch.Tensor) -> torch.Tensor:
        """
        Greedy sequence classification using cumulative rate code.
        X: [B, T, D]
        Returns: LongTensor[B] of predicted classes.
        """
        self.model.eval()
        B, T, _ = X.shape
        X = X.to(self.device)

        state, head_mem = self.model.init_state(B, X.device, X.dtype)
        H_last = self.model.fcs[-1].out_features
        r_sum = torch.zeros(B, H_last, device=X.device)

        W_out = self.model.head.weight
        b_out = self.model.head.bias
        logits = torch.zeros(B, self.meta["n_classes"], device=X.device)

        for t in range(T):
            _, _, state, head_mem, layer_spikes, _ = self.model.forward_step(
                X[:, t, :], state, head_mem, need_pre=False
            )
            r_sum = r_sum + layer_spikes[-1]
            logits = r_sum @ W_out.T + (b_out if b_out is not None else 0.0)

        return logits.argmax(1)
