from abc import ABC, abstractmethod
import torch
from networks.specs import NetConfig

class BaseLearner(ABC):
    """
    Minimal, window-agnostic learner base.
    Contract:
      - forward(X) -> [B,K] logits (handles time inside)
      - train_step(X,y) -> dict with loss/acc
      - predict_batch(X) -> [B] class indices
    """
    def __init__(self, net_cfg: NetConfig, meta: dict, device: torch.device):
        self.cfg = net_cfg
        self.meta = meta
        self.device = device
        self.model = self._build_model().to(device)

    @abstractmethod
    def _build_model(self) -> torch.nn.Module: ...

    @abstractmethod
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X:[B,T,D] -> logits:[B,K]."""
        ...

    @abstractmethod
    def train_step(self, X: torch.Tensor, y: torch.Tensor) -> dict:
        """Single-batch update. Returns logs like {'loss': float, 'acc': float[%]}."""
        ...

    @torch.no_grad()
    def predict_batch(self, X: torch.Tensor) -> torch.Tensor:
        """Greedy prediction: X:[B,T,D] -> LongTensor[B]."""
        self.model.eval()
        X = X.to(self.device)
        logits = self.forward(X)              # [B,K]
        return logits.argmax(dim=1)
