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

class DVSGestureRaw(Dataset):
    """
    IO-only for neuromorphic DVS128 Gesture.
    Returns placeholder x (ignored later) and puts the raw events in info['events'] as Tensor[N,4] (t,x,y,p).
    Time t is in seconds (float32). Use EventToVoxel transform to convert to [T,D].
    """
    def __init__(self, root: str, train: bool = True):
        assert _HAS_TONIC, "tonic is required: pip install tonic"
        self.ds = tonic.datasets.DVSGesture(save_to=root, train=train)
        self.class_names = [str(c) for c in range(11)]  # dataset has 11 gesture classes

    def __len__(self): return len(self.ds)

    def __getitem__(self, i):
        events, y = self.ds[i]  # structured array with fields x,y,t,p

        # SAFE conversions (make contiguous + cast once)
        import numpy as np
        t_us = np.ascontiguousarray(events['t']).astype(np.float32, copy=False)
        xs = np.ascontiguousarray(events['x']).astype(np.float32, copy=False)
        ys = np.ascontiguousarray(events['y']).astype(np.float32, copy=False)
        ps = np.ascontiguousarray(events['p']).astype(np.float32, copy=False)

        t = torch.from_numpy(t_us) / 1e6  # us -> s
        x = torch.from_numpy(xs)
        ypix = torch.from_numpy(ys)
        p = torch.from_numpy(ps)

        E = torch.stack([t, x, ypix, p], dim=-1)  # [N,4]

        x_placeholder = torch.zeros((1, 1), dtype=torch.float32)
        info = {"id": i, "length": 1, "events": E, "H": 128, "W": 128}
        return x_placeholder, int(y), info

def build_dvs_gesture_raw(root: str,
                          max_samples: Optional[int] = None,
                          *,
                          seed: int = 123,
                          min_per_class: int = 10) -> Tuple[Dataset, Dataset, List[str], dict]:
    train = DVSGestureRaw(root=root, train=True)
    test  = DVSGestureRaw(root=root, train=False)

    # cache before potentially wrapping in Subset
    class_names = train.class_names

    info = {
        "true_train_total": len(train),
        "true_test_total": len(test),
        "sensor": "DVS128",
        "class_names": class_names,
    }

    if max_samples is not None:
        from torch.utils.data import Subset
        tr_labels = getattr(train.ds, "targets", None)
        te_labels = getattr(test.ds, "targets", None)
        if tr_labels is None:
            tr_labels = [int(train[i][1]) for i in range(len(train))]
        if te_labels is None:
            te_labels = [int(test[i][1]) for i in range(len(test))]

        tr_idx = stratified_indices_from_labels(list(map(int, tr_labels)),
                                                max_samples, seed=seed, min_per_class=min_per_class)
        te_cap = max(1, min(max(max_samples // 4, 2*len(class_names)), len(test)))
        te_idx = stratified_indices_from_labels(list(map(int, te_labels)),
                                                te_cap, seed=seed, min_per_class=max(3, min_per_class//2))

        train = Subset(train, tr_idx)
        test  = Subset(test,  te_idx)

    # return the cached class_names
    return train, test, class_names, info