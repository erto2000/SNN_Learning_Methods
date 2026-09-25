from typing import List, Dict, Optional
import torch
import torch.nn.functional as F
from .base import BaseLearner
from networks.snn_core import SNNCore

class EpropLearner(BaseLearner):
    """
    E-Prop with online eligibility traces and a final cumulative-spike loss.

    Traces use the backbone's fast-sigmoid surrogate and detached LIF reset.
    Multilayer learning signals use static feedback through forward weights;
    they approximate the downstream temporal derivatives rather than BPTT.
    SGD and Adam apply the same gradient estimates.
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
        optimizer: str = "adam",
        adam_eps: float = 1e-8,
    ):
        super().__init__(net_cfg, meta, device)

        # E-PROP supplies gradients explicitly; no autograd graph is required.
        for p in self.model.parameters():
            p.requires_grad = False

        self.rec_flags: List[bool] = [ls.recurrent for ls in net_cfg.layers]
        self.lr_in = lr_in
        self.lr_rec = lr_rec
        self.lr_out = lr_out
        self.drop_diag = drop_diag
        self.weight_clip = weight_clip
        self.optimizer_name = optimizer.lower()

        if self.model.head is None or self.model.head_lif is not None:
            raise ValueError("EpropLearner expects cfg.head == 'logits' (linear head).")
        if net_cfg.spike_grad.lower() != "fast_sigmoid":
            raise ValueError("EpropLearner supports spike_grad='fast_sigmoid' only.")
        if any(fc.bias is not None for fc in self.model.fcs):
            raise ValueError("EpropLearner requires hidden biases to be disabled.")
        if any(ls.norm is not None for ls in net_cfg.layers):
            raise ValueError("EpropLearner requires hidden normalization to be disabled.")

        # E-PROP computes its own gradient estimates from eligibility traces.
        # The optimizer only determines how those gradient estimates are applied.
        # Separate parameter groups preserve the learner's input/recurrent/output
        # learning-rate controls for both SGD and Adam.
        param_groups = [
            {"params": [fc.weight for fc in self.model.fcs], "lr": self.lr_in},
        ]

        recurrent_params = [
            self.model.Wrecs[l]
            for l in range(len(self.model.Wrecs))
            if self.rec_flags[l]
        ]
        if recurrent_params:
            param_groups.append({"params": recurrent_params, "lr": self.lr_rec})

        head_params = [self.model.head.weight]
        if self.model.head.bias is not None:
            head_params.append(self.model.head.bias)
        param_groups.append({"params": head_params, "lr": self.lr_out})

        if self.optimizer_name == "adam":
            self.opt = torch.optim.Adam(param_groups, eps=adam_eps)
        elif self.optimizer_name == "sgd":
            self.opt = torch.optim.SGD(param_groups)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

    def _build_model(self):
        return SNNCore(self.cfg, self.meta["n_classes"])

    @staticmethod
    def _surrogate_fast_sigmoid(u: torch.Tensor, slope: float) -> torch.Tensor:
        # Same derivative as snntorch.surrogate.fast_sigmoid.
        return (1.0 + slope * u.abs()).pow(-2)

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
        W_out = self.model.head.weight
        b_out = self.model.head.bias

        r_sum = torch.zeros(B, H_last, device=X.device, dtype=X.dtype)
        logits = torch.zeros(B, self.meta["n_classes"], device=X.device, dtype=W_out.dtype)

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
        Accumulate local spike eligibilities online, then apply the final loss.

        The membrane eligibility is a filtered presynaptic signal. Multiplying
        it by the current spike derivative gives the spike eligibility, which
        must be summed over time because the readout sums all output spikes.
        """
        self.model.train()
        K = self.meta["n_classes"]
        slope = self.cfg.slope

        X = X.to(self.device)  # [B, T, D]
        y = y.to(self.device)  # [B]
        B, T, _ = X.shape

        state, head_mem = self.model.init_state(B, X.device, X.dtype)
        L = len(self.model.fcs)
        Hs = [fc.out_features for fc in self.model.fcs]
        in_dims = [self.model.fcs[0].in_features] + [fc.out_features for fc in self.model.fcs[:-1]]

        # Filtered presynaptic signals and accumulated spike eligibilities.
        # The local membrane derivative ignores paths through other neurons.
        pre_ff = [torch.zeros(B, din, device=X.device, dtype=X.dtype) for din in in_dims]
        pre_rec = [torch.zeros(B, Hs[i], device=X.device, dtype=X.dtype) if self.rec_flags[i] else None
                   for i in range(L)]
        e_ff = [torch.zeros(B, in_dims[i], Hs[i], device=X.device, dtype=X.dtype) for i in range(L)]
        e_rec = [torch.zeros(B, Hs[i], Hs[i], device=X.device, dtype=X.dtype) if self.rec_flags[i] else None
                 for i in range(L)]

        # cumulative rate code from last hidden layer
        H_last = Hs[-1]
        r_sum = torch.zeros(B, H_last, device=X.device, dtype=X.dtype)

        # -------- 1) unroll in time: update eligibilities + r_sum only --------
        for t in range(T):
            # Recurrence in SNNCore consumes the preceding timestep's spikes.
            previous_spikes = state.spikes

            _, _, state, head_mem, layer_spikes, _ = self.model.forward_step(
                X[:, t, :], state, head_mem
            )

            z_last = layer_spikes[-1]
            r_sum = r_sum + z_last  # cumulative spike count (rate code)

            for l, lif in enumerate(self.model.lifs):
                # The returned membrane includes snnTorch's delayed reset.
                psi = self._surrogate_fast_sigmoid(state.mems[l] - lif.threshold, slope)
                beta = lif.beta.clamp(0, 1)
                presynaptic = X[:, t, :] if l == 0 else layer_spikes[l - 1]
                pre_ff[l] = beta * pre_ff[l] + presynaptic
                e_ff[l].add_(pre_ff[l].unsqueeze(2) * psi.unsqueeze(1))
                if self.rec_flags[l]:
                    pre_rec[l] = beta * pre_rec[l] + previous_spikes[l]
                    e_rec[l].add_(pre_rec[l].unsqueeze(2) * psi.unsqueeze(1))

        # -------- 2) single loss / learning signal at final time --------
        W_out = self.model.head.weight
        b_out = self.model.head.bias
        logits = r_sum @ W_out.T + (b_out if b_out is not None else 0.0)  # [B,K]

        probs = torch.softmax(logits, dim=1)
        grad_logits = probs - F.one_hot(y, num_classes=K).to(probs.dtype)  # [B,K]

        # Approximate layer-wise feedback; this omits downstream spike/time
        # derivatives for earlier hidden layers, as in the original learner.
        L_sig: List[Optional[torch.Tensor]] = [None for _ in range(L)]
        L_sig[L - 1] = grad_logits @ W_out               # [B, H_last]
        for l in range(L - 1, 0, -1):
            W_l = self.model.fcs[l].weight               # [H_l, H_{l-1}]
            L_sig[l - 1] = L_sig[l] @ W_l               # [B, H_{l-1}]

        # -------- 3) apply mean-batch gradients from eligibilities --------
        norm = max(1, B)
        self.opt.zero_grad(set_to_none=True)
        for l in range(L):
            # e_ff[l]: [B, in_l, H_l], L_sig[l]: [B, H_l]
            g_in_out = torch.einsum("bij,bj->ij", e_ff[l], L_sig[l])  # [in_l, H_l]
            self.model.fcs[l].weight.grad = g_in_out.T / norm

            if self.rec_flags[l]:
                # Wrec uses [presynaptic, postsynaptic] orientation in SNNCore.
                self.model.Wrecs[l].grad = torch.einsum("bij,bj->ij", e_rec[l], L_sig[l]) / norm

        self.model.head.weight.grad = grad_logits.T @ r_sum / norm
        if b_out is not None:
            b_out.grad = grad_logits.mean(dim=0)
        self.opt.step()

        # keep recurrent weights sane
        self._clamp_rec()

        final_loss = F.cross_entropy(logits, y, reduction="mean").item()
        acc = (logits.argmax(1) == y).float().mean().item() * 100.0

        return {"loss": final_loss, "acc": acc}

    def _cost_components(self, batch: int, time_steps: int, dims: dict) -> dict:
        B, T = int(batch), int(time_steps)
        A, U, V, C = dims["A"], dims["U"], dims["V"], dims["C"]
        d0_eff = dims["tilde_d0"]
        d_last = dims["d"][-1] if dims["d"] else 0
        F, P, Y = dims["F"], dims["P"], dims["Y"]

        return {
            "memory": {
                "input": B * T * d0_eff,
                "param": A + V,
                "state": B * U,
                "eligibility_traces": B * A,
                "learning_signals": B * U,
                "final_rate": B * d_last,
                "output_error": B * C,
                "grad": A + V,
            },
            "compute": {
                "forward": B * T * (A + V + U),
                "eligibility": B * T * A,
                "spatial_learning_signal": B * (A + V),
                "update": B * (A + V),
                "output_error": B * C,
            },
            "access": {
                "forward": B * T * (F + Y),
                "eligibility": 2 * B * T * A,
                "spatial_learning_signal": B * (P + Y),
                "update": B * (P + Y),
                "param_read_write": 2 * (A + V),
            },
        }
