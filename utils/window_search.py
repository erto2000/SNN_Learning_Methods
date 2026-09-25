"""Training-only sequence-length inspection for window searches."""
from copy import deepcopy

import torch

from timeseries.registry import get_dataloaders
from utils.common import set_seed


def inspect_training_lengths(config, pipeline, *, max_samples: int = 256) -> dict:
    """Inspect a deterministic subset after preprocessing but before windowing.

    The validation and test sets are never read. Dataset construction also fits
    preprocessing exactly as a normal run does, so the measured lengths match
    the tensors seen by training. Limiting the inspection keeps variable-length
    audio datasets inexpensive relative to the full search.
    """
    g = deepcopy(config)
    set_seed(int(g["SEED"]))
    loaders, _ = get_dataloaders(
        g["DATASET"],
        root=g["DATA_ROOT"],
        batch_size=g["BATCH_SIZE"],
        max_samples=g["MAX_SAMPLES"],
        transform=deepcopy(pipeline),
        num_workers=g.get("NUM_WORKERS"),
        pin_memory=g.get("PIN_MEMORY"),
        seed=g["SEED"],
        data_split=g.get("DATA_SPLIT"),
        **g.get("DATASET_KW", {}),
    )
    dataset = loaders["train"].dataset
    count = min(len(dataset), max(1, int(max_samples)))
    generator = torch.Generator().manual_seed(int(g["SEED"]))
    indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()

    lengths = []
    for index in indices:
        x, _, _ = dataset[index]
        if x.dim() != 2:
            raise ValueError(
                "Window-bound inspection expects unwindowed [T,D] samples; "
                f"received shape {tuple(x.shape)}"
            )
        lengths.append(int(x.shape[0]))

    if not lengths:
        raise ValueError("Cannot derive window bounds from an empty training set")
    return {
        "minimum": min(lengths),
        "maximum": max(lengths),
        "inspected_samples": len(lengths),
        "training_samples": len(dataset),
    }
