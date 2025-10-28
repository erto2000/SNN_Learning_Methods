# timeseries/core.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Protocol, Any
import torch
from torch.utils.data import Dataset

@dataclass
class Sample:
    x: torch.Tensor    # [T,D], float32, unnormalized/raw
    y: int
    info: Dict[str, Any]  # {"id":..., "length": T, ...}

class TimeSeriesDataset(Protocol):
    """IO-only protocol: returns (x:[T,D], y:int, info:dict)."""
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        ...
    def __len__(self) -> int:
        ...

class MapDataset(Dataset):
    """Wrap a dataset with a (x,y,info)->(x',y,info') transform pipeline."""
    def __init__(self, ds: Dataset, transform):
        self.ds, self.transform = ds, transform
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        x, y, info = self.ds[i]
        return self.transform(x, y, info)
