# datasets/mnist_rate.py
from typing import Optional, List, Tuple
import torch
from torch.utils.data import Dataset

try:
    from torchvision import datasets as tvds, transforms as T
    _HAS_TV = True
except Exception:
    _HAS_TV = False

_REAL_T = 20

class _MNISTRateSpike(Dataset):
    def __init__(self, train: bool, root: str, real_T: int = _REAL_T,
                 gain: float = 0.7, seed: int = 0):
        assert _HAS_TV, "torchvision is required for MNIST."
        self.ds = tvds.MNIST(root=root, train=train, download=True, transform=T.ToTensor())
        self.real_T, self.gain = int(real_T), float(gain)
        self.rng = torch.Generator().manual_seed(seed + (0 if train else 1))
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        x, y = self.ds[idx]  # [1,28,28]
        x01 = (x - x.min()) / (x.max() - x.min() + 1e-8)
        p = torch.clamp(self.gain * x01, 0.0, 1.0)
        spikes = torch.bernoulli(p.expand(self.real_T, -1, -1, -1), generator=self.rng)
        return spikes.squeeze(1).reshape(self.real_T, 28*28).float(), int(y)

def build_mnist_rate(root: str, max_samples: Optional[int] = None, gain: float = 0.7, **_):
    train = _MNISTRateSpike(train=True,  root=root, real_T=_REAL_T, gain=gain, seed=0)
    test  = _MNISTRateSpike(train=False, root=root, real_T=_REAL_T, gain=gain, seed=0)

    if max_samples is not None:
        train = torch.utils.data.Subset(train, range(min(max_samples, len(train))))
        test  = torch.utils.data.Subset(test,  range(max(1, min(max_samples // 4 if max_samples and max_samples > 4 else 1, len(test)))))

    return train, test, [str(i) for i in range(10)]
