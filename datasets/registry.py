# datasets/registry.py
from typing import Callable, Dict, Optional, Tuple, List
import torch
from torch.utils.data import DataLoader, Dataset
from .collate import make_segment_collate
from .core import DatasetMeta
from dataclasses import asdict

# Import builders once, explicitly:
from .har import build_har
from .speech_commands import build_sc
from .mnist_static import build_mnist_static
from .mnist_rate import build_mnist_rate

LoaderFn = Callable[..., Tuple[Dataset, Dataset, List[str]]]

_REGISTRY: Dict[str, LoaderFn] = {
    "har": build_har,
    "speech_commands": build_sc,
    "mnist": build_mnist_static,
    "mnist_rate": build_mnist_rate,
}

def get_dataloaders(dataset: str,
                    root: str = "./data",
                    batch_size: int = 128,
                    max_samples: Optional[int] = None,
                    sample_length: Optional[int] = None,
                    stride: Optional[int] = None,
                    **kwargs):
    name = dataset.lower()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset!r}. Registered: {list(_REGISTRY)}")

    train_ds, test_ds, class_names = _REGISTRY[name](root=root, max_samples=max_samples, **kwargs)

    collate_fn = make_segment_collate(sample_length, stride)
    pin_mem = torch.cuda.is_available()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=False, pin_memory=pin_mem, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              drop_last=False, pin_memory=pin_mem, collate_fn=collate_fn)

    # Probe to infer T/D
    tmpX, _ = next(iter(DataLoader(train_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)))
    meta = DatasetMeta(
        n_classes=len(class_names),
        input_dim=int(tmpX.shape[3]),
        time_steps=int(tmpX.shape[2]),
        class_names=class_names,
        num_train_samples=len(train_ds),
        num_test_samples=len(test_ds),
    )
    return train_loader, test_loader, asdict(meta)
