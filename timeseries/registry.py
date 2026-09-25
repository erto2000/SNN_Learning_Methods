"""Shared dataset preparation for training, search, and final evaluation."""
from copy import deepcopy
import torch
from torch.utils.data import DataLoader
from .collate import collate_pad
from .transforms import Compose, DeterministicSpikes
from .core import MapDataset
from .splitting import load_dataset_partitions
from .datasets.har import build_har_raw
from .datasets.mnist import build_mnist_raw
from .datasets.speech_commands import build_sc_raw
from .datasets.esc50 import build_esc50_raw
from .datasets.urban8k import build_urban8k_raw
from .datasets.pamap2 import build_pamap2_raw
from .datasets.mitbih import build_mitbih_raw
from .datasets.dvs_gesture import build_dvs_gesture_raw
from .datasets.large_scale_audio import build_large_scale_audio_raw

_REGISTRY = dict(har=build_har_raw, mnist=build_mnist_raw,
    speech_commands=build_sc_raw, esc50=build_esc50_raw, urban8k=build_urban8k_raw,
    pamap2=build_pamap2_raw, mitbih=build_mitbih_raw, dvs_gesture=build_dvs_gesture_raw,
    large_scale_audio=build_large_scale_audio_raw)


def get_dataloaders(dataset, root='./data', batch_size=128, max_samples=None,
                    transform=None, num_workers=4, pin_memory=None, seed=123,
                    data_split=None, **kwargs):
    """Return a train/validation/test loader dictionary and shared metadata.

    Percentages apply to the combined sample pool returned by the raw builder,
    after class filtering/sample limits. Split before fitting preprocessing or
    creating temporal windows. All callers use the same global seed.
    """
    name = dataset.lower()
    if name not in _REGISTRY:
        raise ValueError(f'Unknown dataset: {dataset!r}')
    subsets, class_names, metadata = load_dataset_partitions(
        _REGISTRY[name], root=root, max_samples=max_samples, seed=seed,
        data_split=data_split, **kwargs)
    if transform is not None:
        pipeline = transform if isinstance(transform, Compose) else Compose([transform])
        if any(isinstance(op, DeterministicSpikes) for op in pipeline.ops):
            pipeline = deepcopy(pipeline)
            for op in pipeline.ops:
                if isinstance(op, DeterministicSpikes):
                    op.base_seed = int(seed)
        pipeline.fit(subsets['train'], max_samples=min(2000, len(subsets['train'])))
        subsets = {key: MapDataset(ds, pipeline) for key, ds in subsets.items()}
    pin_mem = torch.cuda.is_available() if pin_memory is None else bool(pin_memory)
    loaders = {key: DataLoader(ds, batch_size=batch_size, shuffle=(key == 'train'),
        drop_last=False, pin_memory=pin_mem, num_workers=num_workers,
        persistent_workers=(num_workers > 0), collate_fn=collate_pad,
        generator=torch.Generator().manual_seed(seed)) for key, ds in subsets.items()}
    x0, _, sample_info = subsets['train'][0]
    time_steps = int(x0.shape[-2]) if x0.dim() in (2, 3) else None
    metadata.update(
        n_classes=len(class_names), class_names=class_names, input_dim=int(x0.shape[-1]),
        num_train_samples=len(subsets['train']), num_validation_samples=len(subsets['validation']),
        num_test_samples=len(subsets['test']), time_steps=time_steps,
        windows_per_sample=int(x0.shape[0]) if x0.dim() == 3 else 1,
        pre_window_time_steps=sample_info.get('pre_window_time_steps', time_steps),
        window_length=sample_info.get('window_length'), window_hop=sample_info.get('window_hop'))
    return loaders, metadata
