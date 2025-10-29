from __future__ import annotations
from typing import Dict, List, Optional
import random
from ._labels import get_labels

def _class_buckets_from_labels(labels: List[int]) -> Dict[int, List[int]]:
    buckets: Dict[int, List[int]] = {}
    for i, y in enumerate(labels):
        buckets.setdefault(int(y), []).append(i)
    return buckets

def stratified_indices(
    ds,
    max_total: Optional[int],
    *,
    seed: int = 123,
    min_per_class: int = 0,
    per_class_cap: Optional[int] = None
) -> List[int]:
    """
    Same API as before, but FAST:
    - Builds buckets from cached integer labels (no x loads, no audio I/O)
    - Deterministic given seed
    """
    rng = random.Random(seed)
    labels = get_labels(ds)                 # O(N) simple ints
    buckets = _class_buckets_from_labels(labels)

    # shuffle each bucket
    for k in buckets:
        rng.shuffle(buckets[k])

    # floor
    picked: List[int] = []
    start_ptr = {}
    for k, idxs in buckets.items():
        take = min(min_per_class, len(idxs))
        if per_class_cap is not None:
            take = min(take, per_class_cap)
        picked.extend(idxs[:take])
        start_ptr[k] = take

    if max_total is None:
        rest = []
        for k, idxs in buckets.items():
            end = len(idxs) if per_class_cap is None else min(len(idxs), per_class_cap)
            rest.extend(idxs[start_ptr[k]:end])
        rng.shuffle(rest)
        out = picked + rest
        rng.shuffle(out)
        return out

    # round-robin top-up
    ks = list(buckets.keys())
    while len(picked) < max_total and ks:
        for k in list(ks):
            idxs = buckets[k]
            ptr = start_ptr[k]
            end = len(idxs) if per_class_cap is None else min(len(idxs), per_class_cap)
            if ptr < end:
                picked.append(idxs[ptr])
                start_ptr[k] = ptr + 1
                if len(picked) >= max_total:
                    break
            else:
                ks.remove(k)

    rng.shuffle(picked)
    return picked
