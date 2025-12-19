import torch
import torch.nn as nn
import torch.optim as optim
from .base import BaseLearner
from networks.snn_core import SNNCore
from utils.time_gating import make_time_weights

class BackpropLearner(BaseLearner):
    """
    Standard backprop with a time-aggregated readout.
    Aggregation over time: 'mean' | 'sum' | 'last' (default: 'mean').
    """
    def __init__(self, net_cfg, meta, device, agg: str = "mean", lr: float = 1e-3, time_gating: dict = None):
        super().__init__(net_cfg, meta, device)
        self.agg = agg
        self.loss = nn.CrossEntropyLoss()
        self.opt = optim.Adam(self.model.parameters(), lr=lr)
        self.time_gating = time_gating or {"enabled": False}

    def _build_model(self):
        return SNNCore(self.cfg, self.meta["n_classes"])

    def _aggregate(self, seq_BTK: torch.Tensor) -> torch.Tensor:
        # seq_BTK: [B,T,K] -> [B,K]
        if self.agg == "sum":  return seq_BTK.sum(dim=1)
        if self.agg == "mean": return seq_BTK.mean(dim=1)
        if self.agg == "last": return seq_BTK[:, -1, :]
        raise ValueError(f"Unknown agg: {self.agg}")

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, T, _ = X.shape
        X = X.to(self.device)
        state, head_mem = self.model.init_state(B, X.device, X.dtype)

        outs = []
        for t in range(T):
            _, head_out, state, head_mem, _, _ = self.model.forward_step(X[:, t, :], state, head_mem)
            outs.append(head_out)  # [B,K]

        seq_TBK = torch.stack(outs, dim=0)  # [T,B,K]
        seq_BTK = seq_TBK.permute(1, 0, 2)  # [B,T,K]

        # --- time gating here ---
        if self.time_gating.get("enabled", False):
            w = make_time_weights(
                T,
                device=seq_BTK.device,
                dtype=seq_BTK.dtype,
                start_u=self.time_gating.get("start_u", 0.5),
                mode=self.time_gating.get("mode", "hard"),
                ramp_u=self.time_gating.get("ramp_u", 0.0),
                sharpness=self.time_gating.get("sharpness", 20.0),
            )  # [T]
            w = w.view(1, T, 1)  # broadcast to [B,T,K]

            if self.agg in ("mean", "sum"):
                weighted = seq_BTK * w
                if self.agg == "sum":
                    logits = weighted.sum(dim=1)
                else:
                    denom = w.sum(dim=1).clamp_min(1e-8)
                    logits = weighted.sum(dim=1) / denom
            elif self.agg == "last":
                # last doesn't really need gating; but you could choose last after start
                logits = seq_BTK[:, -1, :]
            else:
                raise ValueError(f"Unknown agg: {self.agg}")
        else:
            logits = self._aggregate(seq_BTK)

        return logits

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
        Approx: store membrane states for all hidden layers across time
        + head outputs per time step.
        """
        Hs = [fc.out_features for fc in self.model.fcs]
        total_hidden = sum(Hs)

        # mem for hidden layers over time
        mem_hidden = batch * time_steps * total_hidden

        # head outputs at each step (logits)
        mem_head = batch * time_steps * self.meta["n_classes"]

        return (mem_hidden + mem_head) * fp_bytes
