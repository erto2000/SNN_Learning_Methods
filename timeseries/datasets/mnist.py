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

def build_mnist_raw(root: str,
                    max_samples: Optional[int] = None,
                    *,
                    seed: int = 123,
                    min_per_class: int = 10):
    from torch.utils.data import Subset
    from ._subsample import stratified_indices

    train = MNISTRaw(train=True,  root=root)
    test  = MNISTRaw(train=False, root=root)

    info = {
        "true_train_total": len(train),
        "true_test_total": len(test)
    }

    if max_samples is not None:
        max_tr = max(1, min(max_samples, len(train)))
        max_te = max(1, min(max(max_samples // 4, 100), len(test)))

        tr_idx = stratified_indices(train, max_tr, seed=seed, min_per_class=min_per_class)
        te_idx = stratified_indices(test,  max_te, seed=seed, min_per_class=max(5, min_per_class//2))

        train = Subset(train, tr_idx)
        test  = Subset(test,  te_idx)

    return train, test, [str(i) for i in range(10)], info
