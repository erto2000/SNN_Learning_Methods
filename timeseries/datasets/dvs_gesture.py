from __future__ import annotations
from typing import Optional, Tuple, List
import torch
from torch.utils.data import Dataset
from ._subsample import stratified_indices_from_labels

# Requires: pip install tonic
try:
    import tonic
    _HAS_TONIC = True
except Exception:
    _HAS_TONIC = False


# Canonical DVSGesture names (11 classes, index-aligned)
_DVS_GESTURE_NAMES: List[str] = [
    "hand_clap",             # 0
    "right_hand_wave",       # 1
    "left_hand_wave",        # 2
    "right_arm_cw",          # 3
    "right_arm_ccw",         # 4
    "left_arm_cw",           # 5
    "left_arm_ccw",          # 6
    "right_hand_cw",         # 7
    "right_hand_ccw",        # 8
    "left_hand_cw",          # 9
    "left_hand_ccw",         # 10
]


class DVSGestureRaw(Dataset):
    """
    IO-only for neuromorphic DVS128 Gesture.
    Returns placeholder x (ignored later) and puts the raw events in info['events'] as Tensor[N,4] (t,x,y,p).
    Time t is in seconds (float32). Use EventToVoxel transform to convert to [T,D].
    """
    def __init__(self, root: str, train: bool = True):
        assert _HAS_TONIC, "tonic is required: pip install tonic"
        self.ds = tonic.datasets.DVSGesture(save_to=root, train=train)

        # Prefer readable class names; fall back to numeric if mismatch ever occurs.
        if isinstance(_DVS_GESTURE_NAMES, list) and len(_DVS_GESTURE_NAMES) == 11:
            self.class_names = list(_DVS_GESTURE_NAMES)
        else:
            self.class_names = [str(c) for c in range(11)]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        events, y = self.ds[i]  # structured array with fields x,y,t,p

        # SAFE conversions (make contiguous + cast once)
        import numpy as np
        t_us = np.ascontiguousarray(events['t']).astype(np.float32, copy=False)
        xs   = np.ascontiguousarray(events['x']).astype(np.float32, copy=False)
        ys   = np.ascontiguousarray(events['y']).astype(np.float32, copy=False)
        ps   = np.ascontiguousarray(events['p']).astype(np.float32, copy=False)

        t    = torch.from_numpy(t_us) / 1e6  # us -> s
        x    = torch.from_numpy(xs)
        ypix = torch.from_numpy(ys)
        p    = torch.from_numpy(ps)

        E = torch.stack([t, x, ypix, p], dim=-1)  # [N,4]

        x_placeholder = torch.zeros((1, 1), dtype=torch.float32)
        info = {
            "id": i,
            "length": 1,
            "events": E,
            "H": 128,
            "W": 128,
        }
        return x_placeholder, int(y), info


class _LabelRemappedSubset(Dataset):
    """Subset that remaps integer labels via old_to_new dict."""
    def __init__(self, base: Dataset, indices: List[int], old_to_new: dict):
        from torch.utils.data import Subset
        self.subset = Subset(base, indices)
        self.old_to_new = old_to_new

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, i):
        x, y, info = self.subset[i]
        return x, self.old_to_new[int(y)], info


def build_dvs_gesture_raw(root: str,
                          max_samples: Optional[int] = None,
                          *,
                          seed: int = 123,
                          min_per_class: int = 10,
                          class_filter: Optional[List[str]] = None) -> Tuple[Dataset, Dataset, List[str], dict]:
    train_raw = DVSGestureRaw(root=root, train=True)
    test_raw  = DVSGestureRaw(root=root, train=False)

    all_class_names = train_raw.class_names  # 11 classes

    # --- Class filter ---
    if class_filter is not None:
        unknown = [c for c in class_filter if c not in all_class_names]
        if unknown:
            raise ValueError(
                f"class_filter contains unknown DVS gesture classes: {unknown}. "
                f"Available: {all_class_names}"
            )
        old_to_new = {all_class_names.index(c): i for i, c in enumerate(class_filter)}
        class_names = list(class_filter)
    else:
        old_to_new = {i: i for i in range(len(all_class_names))}
        class_names = list(all_class_names)

    info = {
        "true_train_total": len(train_raw),
        "true_test_total": len(test_raw),
        "sensor": "DVS128",
        "class_names": class_names,
    }

    # --- Get raw labels ---
    tr_labels_raw = getattr(train_raw.ds, "targets", None)
    te_labels_raw = getattr(test_raw.ds, "targets", None)
    if tr_labels_raw is None:
        tr_labels_raw = [int(train_raw[i][1]) for i in range(len(train_raw))]
    if te_labels_raw is None:
        te_labels_raw = [int(test_raw[i][1]) for i in range(len(test_raw))]
    tr_labels_raw = list(map(int, tr_labels_raw))
    te_labels_raw = list(map(int, te_labels_raw))

    # --- Filter to selected classes ---
    tr_indices = [i for i, y in enumerate(tr_labels_raw) if y in old_to_new]
    te_indices = [i for i, y in enumerate(te_labels_raw) if y in old_to_new]
    tr_labels  = [old_to_new[tr_labels_raw[i]] for i in tr_indices]
    te_labels  = [old_to_new[te_labels_raw[i]] for i in te_indices]

    # --- Stratified subsampling ---
    if max_samples is not None:
        tr_cap = max(1, min(max_samples, len(tr_indices)))
        te_cap = max(1, min(max(max_samples // 4, 2 * len(class_names)), len(te_indices)))

        tr_sub = stratified_indices_from_labels(tr_labels, tr_cap, seed=seed, min_per_class=min_per_class)
        te_sub = stratified_indices_from_labels(te_labels, te_cap, seed=seed, min_per_class=max(3, min_per_class // 2))

        tr_indices = [tr_indices[i] for i in tr_sub]
        te_indices = [te_indices[i] for i in te_sub]

    # --- Build final datasets ---
    if class_filter is not None:
        train: Dataset = _LabelRemappedSubset(train_raw, tr_indices, old_to_new)
        test:  Dataset = _LabelRemappedSubset(test_raw,  te_indices, old_to_new)
    elif max_samples is not None:
        from torch.utils.data import Subset
        train = Subset(train_raw, tr_indices)
        test  = Subset(test_raw,  te_indices)
    else:
        train = train_raw
        test  = test_raw

    return train, test, class_names, info
