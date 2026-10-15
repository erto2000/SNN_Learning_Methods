"""Training-only sequence-length inspection for window searches."""
from copy import deepcopy

import torch

from timeseries.registry import get_dataloaders
from utils.common import set_seed


def _has_fixed_horizon(config, pipeline):
    """Recognize fixed-output pipelines and the current fixed-length loaders.

    A uniform inspection sample alone is insufficient for variable-length
    recordings. Unknown transforms therefore keep conservative deduplication.
    """
    from timeseries import transforms as tr

    ops = pipeline.ops if type(pipeline) is tr.Compose else [pipeline]
    preserving = (tr.ToFloat32, tr.Ensure2D, tr.ZScore, tr.Flatten,
                  tr.SpecAugmentLike)
    if ops and type(ops[-1]) in preserving:
        return _has_fixed_horizon(config, tr.Compose(ops[:-1]))
    if ops and type(ops[-1]) in (tr.DeterministicSpikes, tr.EventToVoxel,
                                tr.RandomTimeCrop):
        return True
    if ops and type(ops[-1]) is tr.RepeatStatic:
        return (config["DATASET"].lower() == "mnist" and
                all(type(op) in preserving for op in ops[:-1]))
    # Fixed raw time series, or audio clips made equal in duration by a loader.
    known = (tr.ToFloat32, tr.Ensure2D, tr.ZScore, tr.Flatten, tr.Resample,
             tr.ToLogMel, tr.SpecAugmentLike)
    if not all(type(op) in known for op in ops):
        return False
    name = config["DATASET"].lower()
    kw = config.get("DATASET_KW", {})
    if name in ("har", "speech_commands", "mitbih"):
        return True
    if name == "pamap2":
        return kw.get("time_steps") is not None
    if name == "esc50":
        return kw.get("duration") is not None
    if name == "large_scale_audio":
        # Integer-second segments resample to equal lengths at a fixed rate.
        duration = kw.get("duration", 3.0)
        resamplers = [op for op in ops if type(op) is tr.Resample]
        return (duration is not None and float(duration).is_integer() and
                bool(resamplers) and all(op.orig_sr is None for op in resamplers))
    return False


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
        "lengths": sorted(set(lengths)),
        "all_lengths_inspected": len(lengths) == len(dataset),
        "fixed_length": (lengths[0] if len(set(lengths)) == 1 and
                         _has_fixed_horizon(g, pipeline) else None),
    }
