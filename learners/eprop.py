import dataclasses
from typing import List, Dict
import torch
import torch.nn.functional as F
from .base import BaseLearner
from networks.snn_core import SNNCore

class EpropLearner(BaseLearner):
    """
    E-Prop with one-step spatial credit using actual forward weights.
    Single-batch `train_step` updates weights once for the given batch.
    """

    def __init__(
        self,
        net_cfg,
        meta: Dict,
        device: torch.device,
        use_recurrence: bool = False,
        lr_in: float = 5e-4,
        lr_rec: float = 5e-4,
        lr_out: float = 1e-3,
        drop_diag: bool = True,
        weight_clip: float = 1.5,
    ):
        layers = [dataclasses.replace(ls, recurrent=use_recurrence) for ls in net_cfg.layers]
        cfg = dataclasses.replace(net_cfg, layers=layers, head="logits")
        super().__init__(cfg, meta, device)

        for p in self.model.parameters():
            p.requires_grad = False

        self.use_rec = use_recurrence
        self.lr_in, self.lr_rec, self.lr_out = lr_in, lr_rec, lr_out
        self.drop_diag, self.weight_clip = drop_diag, weight_clip

    def _build_model(self):
        return SNNCore(self.cfg, self.meta["n_classes"])

    @staticmethod
    def _surrogate_fast_sigmoid(u: torch.Tensor, slope: float) -> torch.Tensor:
        sig = torch.sigmoid(slope * u)
        return slope * sig * (1.0 - sig)

    def _rec_mask(self) -> List[bool]:
        return [self.use_rec] * len(self.model.fcs)

    @torch.no_grad()
    def _clamp_rec(self):
        if not self.use_rec:
            return
        for W in self.model.Wrecs:
            if self.drop_diag:
                W.fill_diagonal_(0.0)
            if self.weight_clip is not None:
                W.clamp_(-self.weight_clip, self.weight_clip)

    @torch.no_grad()
    def train_step(self, X: torch.Tensor, y: torch.Tensor):
        """
        Single-batch update.
        Returns: {"loss_per_step": float, "acc": float[%]}
        """
        total_loss_per_step = 0.0

        K = self.meta["n_classes"]
        beta = self.cfg.beta
        slope = self.cfg.slope
        th = self.cfg.threshold

        X: torch.Tensor = X.to(self.device)  # [B,T,D]
        y: torch.Tensor = y.to(self.device)  # [B]
        B, T, _ = X.shape

        state, head_mem = self.model.init_state(B, X.device, X.dtype)
        L = len(self.model.fcs)
        Hs = [fc.out_features for fc in self.model.fcs]
        in_dims = [self.model.fcs[0].in_features] + [fc.out_features for fc in self.model.fcs[:-1]]

        # eligibilities
        e_ff = [torch.zeros(B, in_dims[i], Hs[i], device=X.device) for i in range(L)]          # [B, in, H]
        e_rec = [torch.zeros(B, Hs[i], Hs[i], device=X.device) if self.use_rec else None
                 for i in range(L)]

        # accumulators
        dW_ff = [torch.zeros_like(self.model.fcs[i].weight) for i in range(L)]                 # [H, in]
        dW_rec = [torch.zeros_like(self.model.Wrecs[i]) if self.use_rec else None for i in range(L)]
        dW_out = torch.zeros_like(self.model.head.weight)                                      # [K, H]
        db_out = torch.zeros_like(self.model.head.bias) if self.model.head.bias is not None else None

        # cumulative rate code for readout
        H_last = Hs[-1]
        r_sum = torch.zeros(B, H_last, device=X.device)  # [B, H]

        rec_mask = self._rec_mask()
        last_logits = None

        for t in range(T):
            # Keep previous membrane values for psi
            v_prev_list = [m.clone() for m in state.mems]

            # forward one step (need pre-activations)
            _, _ignored_logits, state, head_mem, layer_spikes, pres = self.model.forward_step(
                X[:, t, :], state, head_mem, need_pre=True, use_recurrence_mask=rec_mask
            )

            # cumulative spikes (rate code)
            z_last = layer_spikes[-1]  # [B, H]
            r_sum = r_sum + z_last

            # logits from cumulative spikes
            W_out = self.model.head.weight
            b_out = self.model.head.bias
            logits_t = r_sum @ W_out.T + (b_out if b_out is not None else 0.0)  # [B, K]
            last_logits = logits_t

            # surrogate psi per layer: use u = beta*v_prev + pre - th
            psis = []
            for l, pre in enumerate(pres):  # pre: [B, H_l]
                u = beta * v_prev_list[l] + pre - th
                psis.append(self._surrogate_fast_sigmoid(u, slope))

            # update eligibilities
            e_ff[0] = beta * e_ff[0] + X[:, t, :].unsqueeze(2) * psis[0].unsqueeze(1)
            if self.use_rec:
                spk0 = layer_spikes[0]
                e_rec[0] = beta * e_rec[0] + spk0.unsqueeze(2) * psis[0].unsqueeze(1)
            for l in range(1, L):
                pre_l = layer_spikes[l - 1]
                e_ff[l] = beta * e_ff[l] + pre_l.unsqueeze(2) * psis[l].unsqueeze(1)
                if self.use_rec:
                    spk_l = layer_spikes[l]
                    e_rec[l] = beta * e_rec[l] + spk_l.unsqueeze(2) * psis[l].unsqueeze(1)

            # learning signal at readout / top layer
            probs_t = torch.softmax(logits_t, dim=1)
            grad_logits_t = probs_t - F.one_hot(y, num_classes=K).float()

            # top hidden learning signal
            L_sig = [None for _ in range(L)]
            L_sig[L - 1] = grad_logits_t @ W_out

            # propagate down via forward weights (no transpose)
            for l in range(L - 1, 0, -1):
                W_l = self.model.fcs[l].weight
                L_sig[l - 1] = L_sig[l] @ W_l

            # accumulate synaptic grads
            for l in range(L):
                g_in_out = torch.einsum("bij,bj->ij", e_ff[l], L_sig[l])               # [in_l, H_l]
                dW_ff[l] += g_in_out.T
            if self.use_rec:
                for l in range(L):
                    dW_rec[l] += torch.einsum("bij,bj->ij", e_rec[l], L_sig[l])

            # readout gradients use cumulative r_sum
            dW_out += grad_logits_t.T @ r_sum
            if db_out is not None:
                db_out += grad_logits_t.sum(dim=0)

            # diagnostics
            total_loss_per_step += F.cross_entropy(logits_t, y, reduction="mean").item()

        # SGD step (normalize by batch size)
        norm = max(1, B)
        for l in range(L):
            self.model.fcs[l].weight.add_(-(self.lr_in / norm) * dW_ff[l])
        if self.use_rec:
            for l in range(L):
                self.model.Wrecs[l].add_(-(self.lr_rec / norm) * dW_rec[l])
        self.model.head.weight.data.add_(-(self.lr_out / norm) * dW_out)
        if db_out is not None:
            self.model.head.bias.data.add_(-(self.lr_out / norm) * db_out)

        self._clamp_rec()

        acc = (last_logits.argmax(1) == y).float().mean().item() * 100.0
        return {
            "loss_per_step": total_loss_per_step / max(1, T),
            "acc": acc,
        }

    @torch.no_grad()
    def predict_batch(self, X: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        B,T,_ = X.shape
        state, head_mem = self.model.init_state(B, X.device, X.dtype)
        H = self.model.fcs[-1].out_features
        r_sum = torch.zeros(B, H, device=X.device)
        W_out = self.model.head.weight
        b_out = self.model.head.bias
        logits = torch.zeros(B, self.meta["n_classes"], device=X.device)
        rec_mask = self._rec_mask()
        for t in range(T):
            _, _, state, head_mem, layer_spikes, _ = self.model.forward_step(
                X[:, t, :], state, head_mem, need_pre=False, use_recurrence_mask=rec_mask
            )
            r_sum = r_sum + layer_spikes[-1]
            logits = r_sum @ W_out.T + (b_out if b_out is not None else 0.0)
        return logits.argmax(1)
