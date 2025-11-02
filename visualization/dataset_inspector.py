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
    save_pipeline_summary, save_counts_json,
    save_examples_multichannel_traces, save_examples_voxel_slices,
)

# ──────────────────────────────────────────────────────────────────────────────
def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _set_seed(seed: Optional[int]) -> None:
    if seed is None: return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def _build_raw(dataset: str, root: str, max_samples: Optional[int]) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, List[str], dir]:
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

def _estimate_pad_ratio(ds, max_items: int = 64) -> Tuple[float, Optional[float]]:
    n = len(ds)
    if n == 0:
        return 0.0, None

    # Probe up to max_items items (uniformly spread to avoid scanning the whole ds)
    idxs = list(range(min(max_items, n)))
    if n > max_items:
        # spread across dataset
        step = max(1, n // max_items)
        idxs = [min(i * step, n - 1) for i in range(max_items)]

    dims = None
    Ts, Ss = [], []  # time lengths and segment counts
    for i in idxs:
        try:
            x, _, _ = ds[i]
        except Exception:
            continue
        dims = x.dim()
        if dims == 2:
            Ts.append(int(x.shape[0]))
        elif dims == 3:
            Ss.append(int(x.shape[0]))
            Ts.append(int(x.shape[1]))  # per-segment length after SlidingWindow
        else:
            # ignore unknown
            pass

    if not Ts:
        return 0.0, None

    if dims == 2:
        T_max = max(Ts)
        if T_max <= 0:
            return 0.0, None
        mean_time = float(sum(1.0 - (t / T_max) for t in Ts)) / len(Ts)
        return mean_time, None

    # dims == 3
    S_max = max(Ss) if Ss else 0
    T_max = max(Ts) if Ts else 0
    if S_max <= 0 or T_max <= 0:
        return 0.0, 0.0

    # approximate mean paddings
    mean_seg = float(sum(1.0 - (s / S_max) for s in Ss)) / len(Ss) if Ss else 0.0
    # for time mask, pretend filled cells per sample = S_i * T_i
    mean_time = float(sum(1.0 - ((s * t) / (S_max * T_max)) for s, t in zip(Ss, Ts))) / len(Ss)
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
    Produces:
      - corpus stats/plots (per split + overall)
      - (optional) post-pipeline padding stats
      - example figures (raw and/or post) per dataset family
      - simple embedding PCA (post only)
      - pipeline provenance json
      - selection indices + summary.json

    Returns:
      dict(out_dir, figs, summary_path, pipeline_ops)
    """
    _set_seed(SEED)

    # ── Output folders
    out_dir = os.path.join(base_dir, "dataset_visualization", tag, ID)
    _ensure_dir(out_dir)
    _ensure_dir(os.path.join(out_dir, "corpus"))
    _ensure_dir(os.path.join(out_dir, "examples_raw"))
    _ensure_dir(os.path.join(out_dir, "examples_post"))
    _ensure_dir(os.path.join(out_dir, "embeddings"))

    # ── 1) Raw datasets
    train_raw, test_raw, class_names, info = _build_raw(DATASET, DATA_ROOT, MAX_SAMPLES)
    split_map = {"train": train_raw, "test": test_raw}

    # ── 2) Fit/apply pipeline (if provided)
    tf = _maybe_fit_pipeline(TRANSFORM, train_raw)
    split_post = {s: _apply_transform(ds, tf) for s, ds in split_map.items()} if tf is not None else {}

    # ── 3) Corpus stats + per-split figures (RAW)
    figs: Dict[str, str] = {}
    selection: Dict[str, Any] = {"seed": SEED, "per_class_examples": 3}

    split_sizes: Dict[str, int] = {}
    split_counts: Dict[str, List[int]] = {}

    for split in SPLITS:
        ds_raw = split_map[split]
        split_sizes[split] = len(ds_raw)

        counts = _class_counts(ds_raw, len(class_names))
        split_counts[split] = counts

        # class distribution (bar)
        figs[f"class_distribution_{split}"] = save_class_distribution(
            counts, class_names,
            os.path.join(out_dir, "corpus", f"class_distribution_{split}.png")
        )
        # count tables (json)
        save_counts_json(counts, class_names, os.path.join(out_dir, "corpus", f"class_counts_{split}.json"))

        # length histogram (sequence length in time)
        lens = _collect_lengths(ds_raw)
        figs[f"length_hist_{split}"] = save_length_hist(
            lens, os.path.join(out_dir, "corpus", f"length_hist_{split}.png")
        )

    # OVERALL across requested SPLITS only
    overall_counts = [0] * len(class_names)
    for split in SPLITS:
        cc = split_counts[split]
        for i in range(len(overall_counts)):
            overall_counts[i] += cc[i]

    figs["class_distribution_overall"] = save_class_distribution(
        overall_counts, class_names,
        os.path.join(out_dir, "corpus", "class_distribution_overall.png")
    )
    save_counts_json(overall_counts, class_names, os.path.join(out_dir, "corpus", "class_counts_overall.json"))

    # ── 4) Post-pipeline corpus stats (padding) if available
    if tf is not None:
        for split in SPLITS:
            ds_post = split_post[split]
            pad_time, pad_seg = _estimate_pad_ratio(ds_post)
            figs[f"pad_ratio_{split}"] = save_pad_ratio(
                pad_time, pad_seg, os.path.join(out_dir, "corpus", f"pad_ratio_{split}.png")
            )

    # ── 5) Examples (raw & post) — lightweight, stratified
    for split in SPLITS:
        ds_raw = split_map[split]
        sel = _subset_stratified(
            ds_raw, per_class=selection["per_class_examples"],
            class_names=class_names, seed=SEED, max_total=64
        )
        selection[f"indices_raw_{split}"] = list(sel.indices) if hasattr(sel, "indices") else []

        # RAW per-dataset visuals
        if DATASET == "har":
            figs[f"har_traces_{split}"] = save_examples_har_traces(
                sel, class_names, os.path.join(out_dir, "examples_raw", f"har_traces_{split}.png")
            )
        elif DATASET in ("speech_commands",):
            figs[f"waveforms_{split}"] = save_examples_waveforms(
                sel, class_names, os.path.join(out_dir, "examples_raw", f"waveforms_{split}.png")
            )
        elif DATASET == "mnist":
            figs[f"mnist_grid_{split}"] = save_examples_mnist_grid(
                sel, class_names, os.path.join(out_dir, "examples_raw", f"mnist_grid_{split}.png")
            )
        elif DATASET in ("esc50", "urban8k"):
            figs[f"waveforms_{split}"] = save_examples_waveforms(
                sel, class_names, os.path.join(out_dir, "examples_raw", f"waveforms_{split}.png")
            )
        elif DATASET == "pamap2":
            figs[f"pamap2_traces_{split}"] = save_examples_multichannel_traces(
                sel, class_names, os.path.join(out_dir, "examples_raw", f"pamap2_traces_{split}.png")
            )
        elif DATASET == "mitbih":
            figs[f"ecg_{split}"] = save_examples_waveforms(
                sel, class_names, os.path.join(out_dir, "examples_raw", f"ecg_{split}.png")
            )
        # DVS raw (events) omitted; post handles voxel render

        # POST examples when pipeline exists
        if tf is not None:
            ds_post = split_post[split]
            selp = _subset_stratified(
                ds_post, per_class=selection["per_class_examples"],
                class_names=class_names, seed=SEED, max_total=64
            )
            selection[f"indices_post_{split}"] = list(selp.indices) if hasattr(selp, "indices") else []

            if DATASET == "har":
                figs[f"har_segments_{split}"] = save_examples_har_traces(
                    selp, class_names, os.path.join(out_dir, "examples_post", f"har_segments_{split}.png"),
                    overlay_segments=True
                )
            elif DATASET == "speech_commands":
                figs[f"mel_specs_{split}"] = save_examples_mel_specs(
                    selp, class_names, os.path.join(out_dir, "examples_post", f"mel_specs_{split}.png")
                )
            elif DATASET == "mnist":
                figs[f"mnist_time_{split}"] = save_examples_spike_raster(
                    selp, class_names, os.path.join(out_dir, "examples_post", f"mnist_time_{split}.png")
                )
            elif DATASET in ("esc50", "urban8k"):
                figs[f"mel_specs_{split}"] = save_examples_mel_specs(
                    selp, class_names, os.path.join(out_dir, "examples_post", f"mel_specs_{split}.png")
                )
            elif DATASET == "pamap2":
                figs[f"pamap2_segments_{split}"] = save_examples_multichannel_traces(
                    selp, class_names, os.path.join(out_dir, "examples_post", f"pamap2_segments_{split}.png"),
                    overlay_segments=True
                )
            elif DATASET == "mitbih":
                figs[f"ecg_post_{split}"] = save_examples_waveforms(
                    selp, class_names, os.path.join(out_dir, "examples_post", f"ecg_{split}.png")
                )
            elif DATASET == "dvs_gesture":
                # Expect [T, H*W*(1 or 2)] after EventToVoxel -> Flatten (bins,H,W,ch)->[T,D]
                # Default DVS128 params: H=W=128, bins≈200
                figs[f"dvs_voxels_{split}"] = save_examples_voxel_slices(
                    selp, class_names, os.path.join(out_dir, "examples_post", f"dvs_voxels_{split}.png"),
                    H=128, W=128, bins_hint=200
                )

    # ── 6) Embeddings (post only)
    if tf is not None:
        for split in SPLITS:
            ds_post = split_post[split]
            Z, y = _compute_embeddings(ds_post, per_class=16, class_names=class_names, seed=SEED, max_total=512)
            figs[f"pca_{split}"] = save_embeddings_scatter(
                Z, y, class_names, os.path.join(out_dir, "embeddings", f"pca_{split}.png")
            )

    # ── 7) Pipeline provenance
    pipe_json = save_pipeline_summary(_summarize_pipeline(tf), os.path.join(out_dir, "pipeline_ops.json"))

    # ── 8) Persist selection + summary
    with open(os.path.join(out_dir, "selection.json"), "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)

    summary = dict(
        id=ID,
        dataset=DATASET,
        splits=SPLITS,
        data_root=DATA_ROOT,
        max_samples=MAX_SAMPLES,
        notes=NOTES,
        seed=SEED,
        class_names=class_names,
        has_transform=tf is not None,
        output_dir=out_dir,
        split_sizes=split_sizes,
        overall_total=int(sum(split_sizes.get(s, 0) for s in SPLITS)),
        true_total=info.get('true_train_total', 0) + info.get('true_test_total', 0),
        true_train_total=info.get('true_train_total'),
        true_test_total=info.get('true_test_total'),
    )
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return dict(out_dir=out_dir, figs=figs, summary_path=summary_path, pipeline_ops=pipe_json)
