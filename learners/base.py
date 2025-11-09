# learners/base.py
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

    # --------- memory estimates ---------
    def get_static_memory_bytes(self, fp_bytes: int = 4) -> int:
        """
        Theoretical static memory for network parameters (and, if overridden, algorithm
        state). Excludes recurrent weights for layers where recurrence is disabled.
        """
        param_count = 0

        # figure out which recurrent weight tensors to exclude
        exclude_ids = set()
        if hasattr(self.model, "Wrecs") and hasattr(self.model, "Wrec_flags"):
            for flag, W in zip(self.model.Wrec_flags, self.model.Wrecs):
                if not flag:
                    exclude_ids.add(id(W))

        for p in self.model.parameters():
            if id(p) in exclude_ids:
                continue
            param_count += p.numel()

        return param_count * fp_bytes

    def get_training_memory_bytes(self, batch: int, time_steps: int, fp_bytes: int = 4) -> int:
        """
        Theoretical training-time memory (activations, algorithm buffers, etc.)
        Default: 0 (override in subclasses).
        """
        return 0

    @torch.no_grad()
    def predict_batch(self, X: torch.Tensor) -> torch.Tensor:
        """Greedy prediction: X:[B,T,D] -> LongTensor[B]."""
        self.model.eval()
        X = X.to(self.device)
        logits = self.forward(X)              # [B,K]
        return logits.argmax(dim=1)
