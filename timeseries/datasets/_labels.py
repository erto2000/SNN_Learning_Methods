from __future__ import annotations
from typing import List
import os, weakref

from torch.utils.data import Subset, ConcatDataset

# simple cache keyed by dataset object identity
_CACHE = weakref.WeakKeyDictionary()

def _labels_mnist_raw(ds) -> List[int]:
    # MNISTRaw wraps torchvision MNIST at ds.ds
    # torchvision stores targets as a tensor
    return [int(x) for x in ds.ds.targets.tolist()]

def _labels_har_raw(ds) -> List[int]:
    # HARDatasetRaw keeps y as a torch.LongTensor
    return [int(x) for x in ds.y.tolist()]

def _labels_sc_raw(ds) -> List[int]:
    """
    SpeechCommandsRaw: derive labels from paths (no audio load).
    - ds._indices includes -1 entries for "silence" (if enabled)
    - The label for non -1 is the parent folder name of the wav path
    """
    sc = ds.ds  # torchaudio.datasets.SPEECHCOMMANDS
    root = getattr(sc, "_path", None) or sc._path
    word_to_idx = ds.word_to_idx
    silence_idx = word_to_idx.get("silence", None)

    # torchaudio keeps a list of relative file paths in sc._walker
    walker = sc._walker
    out: List[int] = []
    for idx in ds._indices:
        if idx == -1:
            # synthetic silence sample
            out.append(int(silence_idx))
        else:
            rel = walker[idx]
            # parent directory name is the class label
            label = os.path.basename(os.path.dirname(os.path.join(root, rel))).lower()
            out.append(int(word_to_idx[label]))
    return out

def get_labels(ds) -> List[int]:
    # cached?
    if ds in _CACHE:
        return _CACHE[ds]

    if hasattr(ds, "__class__"):
        n = ds.__class__.__name__

        # Base cases: our 3 dataset classes
        if n == "MNISTRaw":
            lab = _labels_mnist_raw(ds)
        elif n == "HARDatasetRaw":
            lab = _labels_har_raw(ds)
        elif n == "SpeechCommandsRaw":
            lab = _labels_sc_raw(ds)

        # Meta-datasets
        elif isinstance(ds, Subset):
            base = get_labels(ds.dataset)
            lab = [base[i] for i in ds.indices]
        elif isinstance(ds, ConcatDataset):
            lab = []
            for child in ds.datasets:
                lab.extend(get_labels(child))
        else:
            # Fallback: last resort (still avoids loading x):
            # try to read `targets` or `y` attr if present
            if hasattr(ds, "targets"):
                t = ds.targets
                lab = [int(x) for x in (t.tolist() if hasattr(t, "tolist") else list(t))]
            elif hasattr(ds, "y"):
                y = ds.y
                lab = [int(x) for x in (y.tolist() if hasattr(y, "tolist") else list(y))]
            else:
                # OK, truly unknown: do the slow path (shouldn’t happen for your loaders)
                tmp = []
                for i in range(len(ds)):
                    _, y, _ = ds[i]
                    tmp.append(int(y))
                lab = tmp

    _CACHE[ds] = lab
    return lab
