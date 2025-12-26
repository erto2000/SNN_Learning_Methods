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
        Theoretical static memory for network
        """
        # ----- read dims from the actual network -----
        fcs = getattr(self.model, "fcs", None)
        if fcs is None or len(fcs) == 0:
            return 0

        # layer widths
        d0 = fcs[0].in_features
        d = [fc.out_features for fc in fcs]  # [d1..dL]
        L = len(d)

        # recurrence flags
        r_flags = getattr(self.model, "Wrec_flags", [False] * L)
        r_flags = list(r_flags)

        # number of classes C (prefer meta, fall back to model.n_classes)
        C = int(self.meta.get("n_classes", getattr(self.model, "n_classes", 0)))

        # ----- theoretical param counts -----
        # feedforward weights: sum d_l d_{l-1}
        Nf = d[0] * d0 + sum(d[l] * d[l - 1] for l in range(1, L))

        # recurrent weights: sum r_l d_l^2
        Nr = sum((d[l] * d[l]) for l in range(L) if r_flags[l])

        # output head weights: C d_L if head exists
        Nout = C * d[-1] if getattr(self.model, "head", None) is not None else 0

        Nstatic = Nf + Nr + Nout
        return Nstatic * fp_bytes

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
