import torch
import torch.nn as nn
import torch.optim as optim
from .base import BaseLearner
from networks.snn_core import SNNCore

class BackpropLearner(BaseLearner):
    """
    Standard backprop with a time-aggregated readout.
    Aggregation over time: 'mean' | 'sum' | 'last' (default: 'mean').
    """
    def __init__(self, net_cfg, meta, device, agg: str = "mean", lr: float = 1e-3):
        super().__init__(net_cfg, meta, device)
        self.agg = agg
        self.loss = nn.CrossEntropyLoss()
        self.opt = optim.Adam(self.model.parameters(), lr=lr)

    def _build_model(self):
        return SNNCore(self.cfg, self.meta["n_classes"])

    def _aggregate(self, seq_BTK: torch.Tensor) -> torch.Tensor:
        # seq_BTK: [B,T,K] -> [B,K]
        if self.agg == "sum":  return seq_BTK.sum(dim=1)
        if self.agg == "mean": return seq_BTK.mean(dim=1)
        if self.agg == "last": return seq_BTK[:, -1, :]
        raise ValueError(f"Unknown agg: {self.agg}")

    def forward(self, X: torch.Tensor, return_activity: bool = False):
        # X: [B,T,D] -> logits: [B,K] OR (logits, activity)
        B, T, _ = X.shape
        X = X.to(self.device)
        state, head_mem = self.model.init_state(B, X.device, X.dtype)

        outs = []
        layer_spike_counts = [0.0 for _ in self.model.fcs]

        for t in range(T):
            _, head_out, state, head_mem, layer_spikes, _ = self.model.forward_step(
                X[:, t, :], state, head_mem
            )
            outs.append(head_out)
            if return_activity:
                for i, spk in enumerate(layer_spikes):
                    layer_spike_counts[i] += float(spk.detach().sum().item())

        seq_TBK = torch.stack(outs, dim=0)  # [T,B,K]
        seq_BTK = seq_TBK.permute(1, 0, 2).contiguous()
        logits = self._aggregate(seq_BTK)  # [B,K]
        assert logits.shape == (B, self.meta["n_classes"])

        if not return_activity:
            return logits

        activity = self._make_activity_dict(
            layer_spike_counts=layer_spike_counts,
            num_samples=B,
            num_timesteps=T,
        )
        return logits, activity

    def train_step(self, X: torch.Tensor, y: torch.Tensor) -> dict:
        self.model.train()
        X, y = X.to(self.device), y.to(self.device)
        logits = self.forward(X)                  # [B,K]
        loss = self.loss(logits, y)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        acc = (logits.argmax(1) == y).float().mean().item() * 100.0
        return {"loss": float(loss.item()), "acc": acc}

    def get_training_memory_bytes(self, batch: int, time_steps: int, fp_bytes: int = 4) -> int:
        """
        Approx: membrane states for all hidden layers + output logits, both across time.
        """
        Hs = [fc.out_features for fc in self.model.fcs]
        mem_hidden = batch * time_steps * sum(Hs)

        K = int(self.meta.get("n_classes", 0))
        mem_out = batch * time_steps * K

        return (mem_hidden + mem_out) * fp_bytes
