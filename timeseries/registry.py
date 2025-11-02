# timeseries/registry.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Callable, Optional, Tuple, List
import torch
from torch.utils.data import DataLoader, Dataset
from .collate import collate_pad
from .transforms import Compose, Transform
from .core import MapDataset

# raw builders
from .datasets.har import build_har_raw
from .datasets.mnist import build_mnist_raw
from .datasets.speech_commands import build_sc_raw
from .datasets.esc50 import build_esc50_raw
from .datasets.urban8k import build_urban8k_raw
from .datasets.pamap2 import build_pamap2_raw
from .datasets.mitbih import build_mitbih_raw
from .datasets.dvs_gesture import build_dvs_gesture_raw

@dataclass
class DatasetMeta:
    n_classes: int
    input_dim: int
    class_names: List[str]
    num_train_samples: Optional[int] = None
    num_test_samples: Optional[int] = None
    time_steps: Optional[int] = None   # approximate (from segment length if present)

LoaderFn = Callable[..., Tuple[Dataset, Dataset, List[str], dict]]

_REGISTRY: Dict[str, LoaderFn] = {
    "har": build_har_raw,
    "mnist": build_mnist_raw,
    "speech_commands": build_sc_raw,
    "esc50": build_esc50_raw,
    "urban8k": build_urban8k_raw,
    "pamap2": build_pamap2_raw,
    "mitbih": build_mitbih_raw,
    "dvs_gesture": build_dvs_gesture_raw,
}

def _maybe_fit_pipeline(train_ds, transform) -> None:
    if isinstance(transform, Compose):
        # use a subset to estimate mean/std
        transform.fit(train_ds, max_samples=min(2000, len(train_ds)))
    elif hasattr(transform, "fit"):
        transform.fit(train_ds)

def get_dataloaders(dataset: str,
                    root: str = "./data",
                    batch_size: int = 128,
                    max_samples: Optional[int] = None,
                    transform: Optional[Transform] = None,
                    num_workers: int = 4,
                    pin_memory: bool | None = None,
                    **kwargs):
    name = dataset.lower()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset!r}. Registered: {list(_REGISTRY)}")

    train_ds, test_ds, class_names, info = _REGISTRY[name](root=root, max_samples=max_samples, **kwargs)

    if transform is not None:
        _maybe_fit_pipeline(train_ds, transform)
        train_ds = MapDataset(train_ds, transform)
        test_ds = MapDataset(test_ds, transform)

    pin_mem = torch.cuda.is_available() if pin_memory is None else bool(pin_memory)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        drop_last=False, pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        collate_fn=collate_pad,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        drop_last=False, pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        collate_fn=collate_pad,
    )

    # probe transformed sample (pre-collate) for input_dim and T
    x0, _, _ = train_ds[0]
    if x0.dim() == 2:
        input_dim = x0.shape[-1]
        time_steps = x0.shape[0]
    elif x0.dim() == 3:
        input_dim = x0.shape[-1]
        time_steps = x0.shape[-2]
    else:
        input_dim = 1
        time_steps = None

    meta = DatasetMeta(
        n_classes=len(class_names),
        input_dim=int(input_dim),
        class_names=class_names,
        num_train_samples=len(train_ds),
        num_test_samples=len(test_ds),
        time_steps=int(time_steps) if time_steps is not None else None,
    )
    return train_loader, test_loader, asdict(meta)
