# timeseries/datasets/_subsample.py
from __future__ import annotations
from typing import Dict, List, Optional
import random

def stratified_indices_from_labels(
    labels: List[int],
    max_total: Optional[int],
    *,
    seed: int = 123,
    min_per_class: int = 0,
    per_class_cap: Optional[int] = None
) -> List[int]:
    """
    Build stratified indices from a *label list* (no dataset coupling).

    Args:
      labels: list of class ids, one per sample (len = len(dataset)).
      max_total: cap on total returned indices. If None, return all.
      seed: RNG seed for deterministic shuffling/round-robin.
      min_per_class: guarantee at least this many per class if available.
      per_class_cap: optional per-class hard cap.

    Returns:
      A shuffled list of indices into the original dataset.
    """
    rng = random.Random(seed)

    # bucketize
    buckets: Dict[int, List[int]] = {}
    for i, y in enumerate(labels):
        buckets.setdefault(int(y), []).append(i)

    # shuffle each bucket deterministically
    for k in buckets:
        rng.shuffle(buckets[k])

    # floor (guarantee min_per_class)
    picked: List[int] = []
    start_ptr: Dict[int, int] = {}
    for k, idxs in buckets.items():
        take = min(min_per_class, len(idxs))
        if per_class_cap is not None:
            take = min(take, per_class_cap)
        picked.extend(idxs[:take])
        start_ptr[k] = take

    if max_total is None:
        # take the rest up to per_class_cap, then shuffle all
        rest: List[int] = []
        for k, idxs in buckets.items():
            end = len(idxs) if per_class_cap is None else min(len(idxs), per_class_cap)
            rest.extend(idxs[start_ptr[k]:end])
        rng.shuffle(rest)
        out = picked + rest
        rng.shuffle(out)
        return out

    # round-robin fill-up to max_total
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
