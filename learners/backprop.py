import dataclasses
import torch
import torch.nn as nn
import torch.optim as optim
from .base import BaseLearner
from networks.snn_core import SNNCore

class BackpropLearner(BaseLearner):
    """
    Standard rate-code training:
      - head="logits" (default) or "lif" (spike counts)
      - aggregation over time is learner-owned: "sum" | "mean" | "last"
      - train_step updates once for the given batch.
    """
    def __init__(self, net_cfg, meta, device, agg: str = "sum", head: str = "logits", lr: float = 1e-3):
        assert head in ("logits", "lif")
        cfg = dataclasses.replace(net_cfg, head=head)
        super().__init__(cfg, meta, device)
        self.agg = agg
        self.loss = nn.CrossEntropyLoss()
        self.opt = optim.Adam(self.model.parameters(), lr=lr)

    def _build_model(self):
        return SNNCore(self.cfg, self.meta["n_classes"])

    def _aggregate(self, seq_TNC: torch.Tensor) -> torch.Tensor:  # [T,B,C] -> [B,C]
        if self.agg == "sum":  return seq_TNC.sum(0)
        if self.agg == "mean": return seq_TNC.mean(0)
        if self.agg == "last": return seq_TNC[-1]
        raise ValueError(f"Unknown agg: {self.agg}")

    def scores_sequence(self, X: torch.Tensor) -> torch.Tensor:
        B,T,_ = X.shape
        state, head_mem = self.model.init_state(B, X.device, X.dtype)
        outs = []
        for t in range(T):
            _, head_out, state, head_mem, _ , _ = self.model.forward_step(X[:,t,:], state, head_mem)
            outs.append(head_out)   # [B,C]
        return torch.stack(outs, 0)

    def train_step(self, X: torch.Tensor, y: torch.Tensor) -> dict:
        self.model.train()
        X, y = X.to(self.device), y.to(self.device)
        seq = self.scores_sequence(X)          # [T,B,C]
        scores = self._aggregate(seq)          # [B,C]
        loss = self.loss(scores, y)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        acc = (scores.argmax(1) == y).float().mean().item() * 100.0
        return {"loss": loss.item(), "acc": acc}

    @torch.no_grad()
    def predict_batch(self, X: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        seq = self.scores_sequence(X)
        scores = self._aggregate(seq)
        return scores.argmax(1)
