# datasets/core.py
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Protocol, Any
import torch
from torch.utils.data import Dataset

@dataclass
class DatasetMeta:
    n_classes: int
    input_dim: int       # feature dim per time step
    time_steps: int      # segment length used by the loader (T after collate)
    class_names: List[str]
    num_train_samples: Optional[int] = None
    num_test_samples: Optional[int] = None
    norm_mean: Optional[List[float]] = None
    norm_std: Optional[List[float]] = None

class TimeSeriesDataset(Dataset):
    """
    Base protocol: each __getitem__(i) returns (x:[T,D] or [T], y:int).
    No segmentation, no padding here. Just raw per-sample time series.
    """
    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:  # type: ignore[override]
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

class HasClassNames(Protocol):
    class_names: List[str]
