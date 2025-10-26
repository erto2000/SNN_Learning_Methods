from abc import ABC, abstractmethod
import torch
from networks.specs import NetConfig

class BaseLearner(ABC):
    def __init__(self, net_cfg: NetConfig, meta: dict, device: torch.device):
        self.cfg = net_cfg
        self.meta = meta
        self.device = device
        self.model = self._build_model().to(device)

    @abstractmethod
    def _build_model(self) -> torch.nn.Module: ...

    # Single-batch update: X:[B,T,D], y:[B]
    @abstractmethod
    def train_step(self, X: torch.Tensor, y: torch.Tensor) -> dict: ...

    # Predict classes for X:[B,T,D] -> LongTensor[B]
    @abstractmethod
    def predict_batch(self, X: torch.Tensor) -> torch.Tensor: ...

    # Optional: return [T,B,C] scores (logits or spike counts)
    def scores_sequence(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
