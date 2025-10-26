# datasets/mnist_static.py
from typing import Optional, Tuple, List
import torch
from torch.utils.data import Dataset

try:
    from torchvision import datasets as tvds, transforms as T
    _HAS_TV = True
except Exception:
    _HAS_TV = False

_REAL_T = 50

class _MNISTRepeatSeq(Dataset):
    def __init__(self, train: bool, root: str, real_T: int = _REAL_T):
        assert _HAS_TV, "torchvision is required for MNIST."
        self.ds = tvds.MNIST(root=root, train=train, download=True, transform=T.ToTensor())
        self.real_T = int(real_T)
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        x, y = self.ds[idx]        # [1,28,28] in [0,1]
        x = x.view(-1)             # [784]
        series = x.unsqueeze(0).repeat(self.real_T, 1)  # [T,784]
        return series, int(y)

def build_mnist_static(root: str, max_samples: Optional[int] = None, **_):
    assert _HAS_TV, "torchvision is required for MNIST."
    train = _MNISTRepeatSeq(train=True,  root=root, real_T=_REAL_T)
    test  = _MNISTRepeatSeq(train=False, root=root, real_T=_REAL_T)

    if max_samples is not None:
        max_tr = max(1, min(max_samples, len(train)))
        max_te = max(1, min(max_samples // 4 if max_samples and max_samples > 4 else 1, len(test)))
        train = torch.utils.data.Subset(train, range(max_tr))
        test  = torch.utils.data.Subset(test,  range(max_te))

    return train, test, [str(i) for i in range(10)]
