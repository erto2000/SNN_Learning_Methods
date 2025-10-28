# timeseries/datasets/mnist.py
from __future__ import annotations
from typing import Optional, List, Tuple
import torch
from torch.utils.data import Dataset

try:
    from torchvision import datasets as tvds, transforms as T
    _HAS_TV = True
except Exception:
    _HAS_TV = False

class MNISTRaw(Dataset):
    """IO-only: returns static vector as a single-step time series [1,784] in [0,1]."""
    def __init__(self, train: bool, root: str):
        assert _HAS_TV, "torchvision is required for MNIST."
        self.ds = tvds.MNIST(root=root, train=train, download=True, transform=T.ToTensor())
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        x, y = self.ds[idx]          # [1,28,28] in [0,1]
        x = x.view(1, -1).float()    # [1,784]
        info = {"id": idx, "length": 1, "dim": x.shape[1]}
        return x, int(y), info

def build_mnist_raw(root: str, max_samples: Optional[int] = None):
    train = MNISTRaw(train=True,  root=root)
    test  = MNISTRaw(train=False, root=root)
    if max_samples is not None:
        from torch.utils.data import Subset
        train = Subset(train, range(min(max_samples, len(train))))
        test  = Subset(test,  range(max(1, min(max_samples // 4 if max_samples and max_samples > 4 else 1, len(test)))))
    return train, test, [str(i) for i in range(10)]
