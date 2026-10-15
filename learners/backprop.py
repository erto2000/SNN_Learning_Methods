import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .base import BaseLearner
from networks.snn_core import SNNCore

class BackpropLearner(BaseLearner):
    """
    Standard backprop through time with a cumulative temporal readout.

    Summing the per-timestep readout trains on the evidence accumulated over
    the sequence, matching the rate/count objective used by the other temporal
    learners. ``mean`` and ``last`` remain available for explicit experiments.
    """
    def __init__(
        self,
        net_cfg,
        meta,
        device,
        agg: str = "sum",
        lr: float = 1e-3,
        optimizer: str = "adam",
        adam_eps: float = 1e-8,
    ):
        super().__init__(net_cfg, meta, device)
        self.agg = agg
        self.loss = nn.CrossEntropyLoss()
        optimizer = optimizer.lower()
        if optimizer == "adam":
            self.opt = optim.Adam(self.model.parameters(), lr=lr, eps=adam_eps)
        elif optimizer == "sgd":
            self.opt = optim.SGD(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

    def _build_model(self):
        return SNNCore(self.cfg, self.meta["n_classes"])

    def _aggregate(self, seq_BTK: torch.Tensor, time_mask: torch.Tensor | None = None) -> torch.Tensor:
        # seq_BTK: [B,T,K] -> [B,K]
        if time_mask is not None:
            lengths = time_mask.sum(dim=1)
            if self.agg == "sum":
                return seq_BTK.sum(dim=1)
            if self.agg == "mean":
                return seq_BTK.sum(dim=1) / lengths.unsqueeze(1)
            if self.agg == "last":
                rows = torch.arange(seq_BTK.shape[0], device=seq_BTK.device)
                return seq_BTK[rows, lengths - 1]
        if self.agg == "sum":  return seq_BTK.sum(dim=1)
        if self.agg == "mean": return seq_BTK.mean(dim=1)
        if self.agg == "last": return seq_BTK[:, -1, :]
        raise ValueError(f"Unknown agg: {self.agg}")

    def forward(self, X: torch.Tensor, return_activity: bool = False, time_mask: torch.Tensor | None = None):
        # X: [B,T,D] -> logits: [B,K] OR (logits, activity)
        B, T, _ = X.shape
        X = X.to(self.device)
        time_mask = self._prepare_time_mask(X, time_mask)
        state, head_mem = self.model.init_state(B, X.device, X.dtype)

        outs = []
        layer_spike_counts = [0.0 for _ in self.model.fcs]

        for t in range(T):
            _, head_out, state, head_mem, layer_spikes, _ = self.model.forward_step(
                X[:, t, :], state, head_mem,
                time_mask=None if time_mask is None else time_mask[:, t],
            )
            outs.append(head_out)
            if return_activity:
                for i, spk in enumerate(layer_spikes):
                    layer_spike_counts[i] += float(spk.detach().sum().item())

        seq_TBK = torch.stack(outs, dim=0)  # [T,B,K]
        seq_BTK = seq_TBK.permute(1, 0, 2).contiguous()
        logits = self._aggregate(seq_BTK, time_mask)  # [B,K]
        assert logits.shape == (B, self.meta["n_classes"])

        if not return_activity:
            return logits

        activity = self._make_activity_dict(
            layer_spike_counts=layer_spike_counts,
            num_samples=B,
            num_timesteps=T,
            valid_steps=None if time_mask is None else int(time_mask.sum().item()),
        )
        return logits, activity

    def train_step(self, X: torch.Tensor, y: torch.Tensor, time_mask: torch.Tensor | None = None, sample_weight=None) -> dict:
        self.model.train()
        X, y = X.to(self.device), y.to(self.device)
        sample_weight = self._prepare_sample_weight(X, sample_weight)
        logits = self.forward(X, time_mask=time_mask)  # [B,K]
        loss = (
            self.loss(logits, y) if sample_weight is None else
            (F.cross_entropy(logits, y, reduction="none") * sample_weight).mean()
        )
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self._optimizer_step(self.opt)
        acc = (logits.argmax(1) == y).float().mean().item() * 100.0
        return {"loss": float(loss.item()), "acc": acc}

    def _cost_components(self, batch: int, time_steps: int, dims: dict) -> dict:
        B, T = int(batch), int(time_steps)
        A, U, V, C = dims["A"], dims["U"], dims["V"], dims["C"]
        d0_eff = dims["tilde_d0"]
        F, P, Y = dims["F"], dims["P"], dims["Y"]

        return {
            "memory": {
                "input": B * T * d0_eff,
                "param": A + V,
                "state": B * T * U,
                "intermediate_hidden": B * T * U,
                "intermediate_readout": B * T * C,
                "grad": A + V,
            },
            "compute": {
                "forward": B * T * (A + V + U),
                "backward_update": B * T * (A + V + U),
                "readout_temporal": B * T * C,
            },
            "access": {
                "forward": B * T * (F + Y),
                "temporal_hidden": 4 * B * T * U,
                "temporal_readout": 2 * B * T * C,
                "backward_update": B * T * (P + Y),
                "param_read_write": 2 * (A + V),
            },
        }
