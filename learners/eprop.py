from typing import List, Dict, Optional
import torch
import torch.nn.functional as F
from .base import BaseLearner
from networks.snn_core import SNNCore
from utils.time_gating import make_time_weights


class EpropLearner(BaseLearner):
    """
    E-Prop with single-pass online update (manual grads).
    Forward returns logits using (optionally) time-gated cumulative rate code.
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
        time_gating: Optional[dict] = None,
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

        self.time_gating = time_gating or {"enabled": False}

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

    # ------------ contract ------------
    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X:[B,T,D] -> logits:[B,K]
        Uses (optionally) time-gated cumulative rate code.
        """
        B, T, _ = X.shape
        X = X.to(self.device)
        state, head_mem = self.model.init_state(B, X.device, X.dtype)

        H_last = self.model.fcs[-1].out_features
        r_sum = torch.zeros(B, H_last, device=X.device, dtype=X.dtype)

        W_out = self.model.head.weight
        b_out = self.model.head.bias

        w = self._time_weights(T, device=X.device, dtype=X.dtype)  # [T]

        for t in range(T):
            _, _, state, head_mem, layer_spikes, _ = self.model.forward_step(
                X[:, t, :], state, head_mem, need_pre=False
            )
            r_sum = r_sum + w[t] * layer_spikes[-1]

        logits = r_sum @ W_out.T + (b_out if b_out is not None else 0.0)
        return logits

    @torch.no_grad()
    def train_step(self, X: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """
        Online eligibilities, single final learning signal.
        Time gating is applied by accumulating *weighted* eligibility sums
        and a weighted r_sum.
        """
        K = self.meta["n_classes"]
        beta = self.cfg.beta
        slope = self.cfg.slope
        th = self.cfg.threshold

        X = X.to(self.device)
        y = y.to(self.device)
        B, T, _ = X.shape

        state, head_mem = self.model.init_state(B, X.device, X.dtype)

        L = len(self.model.fcs)
        Hs = [fc.out_features for fc in self.model.fcs]
        in_dims = [self.model.fcs[0].in_features] + [fc.out_features for fc in self.model.fcs[:-1]]

        # eligibility traces (stateful)
        e_ff  = [torch.zeros(B, in_dims[i], Hs[i], device=X.device, dtype=X.dtype) for i in range(L)]
        e_rec = [torch.zeros(B, Hs[i], Hs[i], device=X.device, dtype=X.dtype) if self.rec_flags[i] else None
                 for i in range(L)]

        # weighted accumulators over time (this is where gating acts)
        e_ff_acc  = [torch.zeros_like(e_ff[i]) for i in range(L)]
        e_rec_acc = [torch.zeros_like(e_rec[i]) if self.rec_flags[i] else None for i in range(L)]

        # weighted rate code
        r_sum = torch.zeros(B, Hs[-1], device=X.device, dtype=X.dtype)

        w = self._time_weights(T, device=X.device, dtype=X.dtype)  # [T]

        # -------- unroll time --------
        for t in range(T):
            v_prev_list = [m.clone() for m in state.mems]
            _, _, state, head_mem, layer_spikes, pres = self.model.forward_step(
                X[:, t, :], state, head_mem, need_pre=True
            )

            # weighted rate code
            r_sum = r_sum + w[t] * layer_spikes[-1]

            # surrogate derivatives
            psis = []
            for l, pre in enumerate(pres):
                u = beta * v_prev_list[l] + pre - th
                psis.append(self._surrogate_fast_sigmoid(u, slope).to(X.dtype))

            # update traces
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

            # accumulate weighted sums (gating)
            wt = w[t]
            for l in range(L):
                e_ff_acc[l].add_(wt * e_ff[l])
                if self.rec_flags[l]:
                    e_rec_acc[l].add_(wt * e_rec[l])

        # -------- final loss / learning signal --------
        W_out = self.model.head.weight
        b_out = self.model.head.bias
        logits = r_sum @ W_out.T + (b_out if b_out is not None else 0.0)

        probs = torch.softmax(logits, dim=1)
        grad_logits = probs - F.one_hot(y, num_classes=K).to(X.dtype)

        # layer learning signals
        L_sig: List[Optional[torch.Tensor]] = [None for _ in range(L)]
        L_sig[L - 1] = grad_logits @ W_out
        for l in range(L - 1, 0, -1):
            W_l = self.model.fcs[l].weight
            L_sig[l - 1] = L_sig[l] @ W_l

        # grads from weighted eligibility sums
        dW_ff  = [torch.zeros_like(self.model.fcs[i].weight) for i in range(L)]
        dW_rec = [torch.zeros_like(self.model.Wrecs[i]) if self.rec_flags[i] else None for i in range(L)]

        for l in range(L):
            g_in_out = torch.einsum("bij,bj->ij", e_ff_acc[l], L_sig[l])  # [in_l,H_l]
            dW_ff[l] = g_in_out.T
            if self.rec_flags[l]:
                dW_rec[l] = torch.einsum("bij,bj->ij", e_rec_acc[l], L_sig[l])

        dW_out = grad_logits.T @ r_sum
        db_out = grad_logits.sum(dim=0) if self.model.head.bias is not None else None

        # apply updates
        norm = max(1, B)
        for l in range(L):
            self.model.fcs[l].weight.add_(-(self.lr_in / norm) * dW_ff[l])
        for l in range(L):
            if self.rec_flags[l]:
                self.model.Wrecs[l].add_(-(self.lr_rec / norm) * dW_rec[l])

        self.model.head.weight.data.add_(-(self.lr_out / norm) * dW_out)
        if db_out is not None:
            self.model.head.bias.data.add_(-(self.lr_out / norm) * db_out)

        self._clamp_rec()

        loss = F.cross_entropy(logits, y).item()
        acc = (logits.argmax(1) == y).float().mean().item() * 100.0
        return {"loss": loss, "acc": acc}
