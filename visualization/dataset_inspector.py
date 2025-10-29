# visualization/dataset_inspector.py
from __future__ import annotations
import os, json, math, random
from copy import deepcopy
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Subset, DataLoader

from timeseries.registry import _REGISTRY as DS_REGISTRY, Compose
from timeseries.core import MapDataset
from timeseries.collate import collate_pad

from .dataset_visualization import (
    save_class_distribution, save_length_hist, save_pad_ratio,
    save_examples_har_traces, save_examples_waveforms, save_examples_mnist_grid,
    save_examples_mel_specs, save_examples_spike_raster, save_embeddings_scatter,
    save_pipeline_summary
)

# ──────────────────────────────────────────────────────────────────────────────
def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _set_seed(seed: Optional[int]) -> None:
    if seed is None: return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def _build_raw(dataset: str, root: str, max_samples: Optional[int]) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, List[str]]:
    name = dataset.lower()
    if name not in DS_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset!r}. Registered: {list(DS_REGISTRY)}")
    build_fn = DS_REGISTRY[name]
    return build_fn(root=root, max_samples=max_samples)

def _maybe_fit_pipeline(transform, train_ds) -> Any:
    if transform is None:
        return None
    tf = deepcopy(transform)
    if isinstance(tf, Compose):
        tf.fit(train_ds)
    elif hasattr(tf, "fit"):
        tf.fit(train_ds)
    return tf

def _apply_transform(ds, transform) -> torch.utils.data.Dataset:
    if transform is None:
        return ds
    return MapDataset(ds, transform)

def _subset_stratified(ds, per_class: int, class_names: List[str], seed: int, max_total: Optional[int] = None) -> Subset:
    rng = random.Random(seed)
    # collect indices per class
    buckets: Dict[int, List[int]] = {k: [] for k in range(len(class_names))}
    for i in range(len(ds)):
        try:
            _, y, _ = ds[i]
        except Exception:
            continue
        if y in buckets:
            buckets[y].append(i)
    # sample
    chosen: List[int] = []
    for k, idxs in buckets.items():
        rng.shuffle(idxs)
        chosen.extend(idxs[:per_class])
    if max_total is not None and len(chosen) > max_total:
        rng.shuffle(chosen)
        chosen = chosen[:max_total]
    return Subset(ds, sorted(chosen))

def _collect_lengths(ds) -> List[int]:
    lens = []
    for i in range(len(ds)):
        x, _, _ = ds[i]
        if x.dim() == 2:  # [T,D]
            lens.append(int(x.shape[0]))
        elif x.dim() == 3:  # [S,T,D]
            lens.append(int(x.shape[1]))
        else:
            continue
    return lens

def _class_counts(ds, n_classes: int) -> List[int]:
    counts = [0]*n_classes
    for i in range(len(ds)):
        _, y, _ = ds[i]
        if 0 <= y < n_classes:
            counts[y]+=1
    return counts

def _estimate_pad_ratio(ds, batch_size=128, num_batches=3) -> Tuple[float, Optional[float]]:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_pad, num_workers=0, drop_last=False)
    pad_time_vals, pad_seg_vals = [], []
    n = 0
    for bidx, (_X, _y, info) in enumerate(loader):
        n += 1
        if "time_mask" in info:
            tm = info["time_mask"]
            pad_time = (tm.numel() - tm.sum().item()) / tm.numel()
            pad_time_vals.append(pad_time)
        if "seg_mask" in info:
            sm = info["seg_mask"]
            pad_seg = (sm.numel() - sm.sum().item()) / sm.numel()
            pad_seg_vals.append(pad_seg)
        if n >= num_batches:
            break
    mean_time = float(np.mean(pad_time_vals)) if pad_time_vals else 0.0
    mean_seg  = float(np.mean(pad_seg_vals))  if pad_seg_vals  else None
    return mean_time, mean_seg

def _compute_embeddings(ds, per_class: int, class_names: List[str], seed: int, max_total: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    sel = _subset_stratified(ds, per_class=per_class, class_names=class_names, seed=seed, max_total=max_total)
    Xs, Ys = [], []
    for i in range(len(sel)):
        x, y, _ = sel[i]
        if x.dim() == 2:           # [T,D] -> mean over time
            feat = x.mean(dim=0).cpu().numpy()
        elif x.dim() == 3:         # [S,T,D] -> mean over S then time
            feat = x.mean(dim=1).mean(dim=0).cpu().numpy()
        else:
            continue
        Xs.append(feat)
        Ys.append(y)
    if not Xs:
        return np.zeros((0,2)), np.zeros((0,))
    X = np.stack(Xs, axis=0)
    y = np.array(Ys)
    # PCA via SVD
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt[:2].T
    return Z.astype(np.float32), y

def _is_primitive(v):
    return isinstance(v, (str, int, float, bool, type(None)))

def _summarize_pipeline(transform) -> Dict[str, Any]:
    if transform is None:
        return {"ops": []}

    ops = []
    if isinstance(transform, Compose):
        it = transform.ops
    else:
        it = [transform]

    for op in it:
        name = op.__class__.__name__
        params = {}
        # keep only primitives (and small tuples/lists of primitives)
        for k, v in op.__dict__.items():
            if k.startswith("_"):
                continue
            if _is_primitive(v):
                params[k] = v
            elif isinstance(v, (list, tuple)) and all(_is_primitive(x) for x in v):
                params[k] = list(v)
            else:
                # skip complex objects (e.g., torchaudio transforms), but record their type names
                params[k] = f"<{v.__class__.__name__}>"

        # tiny summaries for any fitted tensors stored privately (e.g., ZScore)
        fit_summary = {}
        for fk, fv in op.__dict__.items():
            if fk.startswith("_") and isinstance(fv, torch.Tensor):
                t = fv.detach().float()
                fit_summary[fk] = {
                    "shape": list(t.shape),
                    "mean": float(t.mean().item()),
                    "std":  float(t.std().item()),
                }

        ops.append({"op": name, "params": params, "fitted": fit_summary})

    return {"ops": ops}

# ──────────────────────────────────────────────────────────────────────────────
def build_dataset_viz(
    *,
    ID: str,
    DATASET: str,
    SPLITS: List[str],
    DATA_ROOT: str = "./data",
    MAX_SAMPLES: Optional[int] = 2000,
    TRANSFORM: Any = None,
    NOTES: str = "",
    SEED: Optional[int] = 123,
    base_dir: str = "results",
    tag: str = "dataset_viz",
) -> Dict[str, Any]:
    """
    Orchestrates a single dataset visualization job.
    """
    _set_seed(SEED)
    out_dir = os.path.join(base_dir, "_dataset_visualization", tag, ID)
    _ensure_dir(out_dir)
    _ensure_dir(os.path.join(out_dir, "corpus"))
    _ensure_dir(os.path.join(out_dir, "examples_raw"))
    _ensure_dir(os.path.join(out_dir, "examples_post"))
    _ensure_dir(os.path.join(out_dir, "embeddings"))

    # 1) Build raw datasets
    train_raw, test_raw, class_names = _build_raw(DATASET, DATA_ROOT, MAX_SAMPLES)
    split_map = {"train": train_raw, "test": test_raw}

    # 2) Fit/apply pipeline (if provided)
    tf = _maybe_fit_pipeline(TRANSFORM, train_raw)
    split_post = {s: _apply_transform(ds, tf) for s, ds in split_map.items()} if tf is not None else {}

    # 3) Corpus stats + figures (raw)
    figs = {}
    selection = {"seed": SEED, "per_class_examples": 3}
    for split in SPLITS:
        ds_raw = split_map[split]
        counts = _class_counts(ds_raw, len(class_names))
        figs[f"class_distribution_{split}"] = save_class_distribution(counts, class_names, os.path.join(out_dir, "corpus", f"class_distribution_{split}.png"))
        lens = _collect_lengths(ds_raw)
        figs[f"length_hist_{split}"] = save_length_hist(lens, os.path.join(out_dir, "corpus", f"length_hist_{split}.png"))

    # 4) Post-pipeline corpus stats + figures
    if tf is not None:
        for split in SPLITS:
            ds_post = split_post[split]
            pad_time, pad_seg = _estimate_pad_ratio(ds_post)
            figs[f"pad_ratio_{split}"] = save_pad_ratio(pad_time, pad_seg, os.path.join(out_dir, "corpus", f"pad_ratio_{split}.png"))
            # if 3D segments exist, we can also histogram segments per sample by probing a small subset
            # (Optional) could be added later.

    # 5) Examples (raw & post) — light, stratified
    for split in SPLITS:
        ds_raw = split_map[split]
        sel = _subset_stratified(ds_raw, per_class=selection["per_class_examples"], class_names=class_names, seed=SEED, max_total=64)
        selection[f"indices_raw_{split}"] = list(sel.indices) if hasattr(sel, "indices") else []
        # RAW examples per dataset type
        if DATASET == "har":
            figs[f"har_traces_{split}"] = save_examples_har_traces(sel, class_names, os.path.join(out_dir, "examples_raw", f"har_traces_{split}.png"))
        elif DATASET == "speech_commands":
            figs[f"waveforms_{split}"] = save_examples_waveforms(sel, class_names, os.path.join(out_dir, "examples_raw", f"waveforms_{split}.png"))
        elif DATASET == "mnist":
            figs[f"mnist_grid_{split}"] = save_examples_mnist_grid(sel, class_names, os.path.join(out_dir, "examples_raw", f"mnist_grid_{split}.png"))

        # POST examples when available
        if tf is not None:
            ds_post = split_post[split]
            selp = _subset_stratified(ds_post, per_class=selection["per_class_examples"], class_names=class_names, seed=SEED, max_total=64)
            selection[f"indices_post_{split}"] = list(selp.indices) if hasattr(selp, "indices") else []
            if DATASET == "har":
                # Overlay a few segments per class (kept compact)
                figs[f"har_segments_{split}"] = save_examples_har_traces(selp, class_names, os.path.join(out_dir, "examples_post", f"har_segments_{split}.png"), overlay_segments=True)
            elif DATASET == "speech_commands":
                figs[f"mel_specs_{split}"] = save_examples_mel_specs(selp, class_names, os.path.join(out_dir, "examples_post", f"mel_specs_{split}.png"))
            elif DATASET == "mnist":
                # If rate-coded spikes: raster; if static-repeat: time strip; we auto-detect by T>1
                figs[f"mnist_time_{split}"] = save_examples_spike_raster(selp, class_names, os.path.join(out_dir, "examples_post", f"mnist_time_{split}.png"))

    # 6) Embeddings (post only, model-agnostic features)
    if tf is not None:
        for split in SPLITS:
            ds_post = split_post[split]
            Z, y = _compute_embeddings(ds_post, per_class=16, class_names=class_names, seed=SEED, max_total=512)
            figs[f"pca_{split}"] = save_embeddings_scatter(Z, y, class_names, os.path.join(out_dir, "embeddings", f"pca_{split}.png"))

    # 7) Pipeline provenance
    pipe_json = save_pipeline_summary(_summarize_pipeline(tf), os.path.join(out_dir, "pipeline_ops.json"))

    # 8) Persist selection + summary
    with open(os.path.join(out_dir, "selection.json"), "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)

    summary = dict(
        id=ID, dataset=DATASET, splits=SPLITS, data_root=DATA_ROOT, max_samples=MAX_SAMPLES,
        notes=NOTES, seed=SEED, class_names=class_names,
        has_transform=tf is not None, output_dir=out_dir
    )
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return dict(out_dir=out_dir, figs=figs, summary_path=summary_path, pipeline_ops=pipe_json)
