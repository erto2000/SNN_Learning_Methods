from __future__ import annotations
import os, io, zipfile
from typing import Optional, Tuple, List, Dict, Any
import random
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    import torchaudio
    _HAS_TA = True
except Exception:
    _HAS_TA = False

from ._subsample import stratified_indices_from_labels


class ESC50Raw(Dataset):
    """
    IO-only ESC-50 dataset: returns waveform x:[T,1] at the *original* sample rate.
    No resampling is performed.

    This class now always works on the full ESC-50 dataset; we do not select by folds.
    You can optionally restrict to a subset of classes via `class_filter`.
    The train/test split is handled in `build_esc50_raw`.
    """
    def __init__(
        self,
        root: str,
        class_filter: Optional[List[str]] = None,
    ):
        self.root = root
        self.audio_dir = os.path.join(root, "ESC-50-master", "audio")
        self.meta_path = os.path.join(root, "ESC-50-master", "meta", "esc50.csv")

        # Download if missing
        if not os.path.exists(self.meta_path):
            os.makedirs(root, exist_ok=True)
            url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
            # Lightweight, dependency-free download
            import urllib.request
            zipbytes = urllib.request.urlopen(url).read()
            with zipfile.ZipFile(io.BytesIO(zipbytes)) as z:
                # basic zip-slip guard
                ab_root = os.path.abspath(root)
                for name in z.namelist():
                    dest = os.path.abspath(os.path.join(root, name))
                    if not (dest == ab_root or dest.startswith(ab_root + os.sep)):
                        raise RuntimeError("Unsafe path in zip (zip-slip).")
                z.extractall(root)

        if not os.path.isdir(self.audio_dir) or not os.path.exists(self.meta_path):
            raise RuntimeError(
                f"ESC-50 not found or incomplete under {root}. "
                "Expected 'ESC-50-master/audio' and 'ESC-50-master/meta/esc50.csv'."
            )

        df = pd.read_csv(self.meta_path)

        # Filter by subset of classes if requested
        if class_filter is not None:
            df = df[df["category"].isin(class_filter)].reset_index(drop=True)

        if df.empty:
            raise RuntimeError(
                "No files matched "
                + ("" if class_filter is None else f"class_filter={class_filter}")
                + f" in {self.meta_path}"
            )

        self.rows = df

        # Build class mapping; respect class_filter ordering if provided
        if class_filter is not None:
            # Keep only classes that actually appear in df, in the order of class_filter
            present = set(df["category"].unique().tolist())
            self.class_names = [c for c in class_filter if c in present]
        else:
            self.class_names = sorted(df["category"].unique().tolist())

        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows.iloc[i]

        # ESC-50 audio is flat under ".../audio/"
        fpath = os.path.join(self.audio_dir, row["filename"])
        fpath = os.path.normpath(fpath)

        wav = None
        sr: Optional[int] = None

        # 1) torchaudio path if available
        if _HAS_TA:
            try:
                wav, sr = torchaudio.load(fpath)  # [C,T], float32 or int
            except Exception:
                wav = None

        # 2) soundfile fallback
        if wav is None:
            try:
                import soundfile as sf
            except Exception as e:
                raise RuntimeError(
                    "Failed to load audio: torchaudio not available or failed, "
                    "and 'soundfile' is not installed. Install with: pip install soundfile"
                ) from e
            data, sr = sf.read(fpath, dtype="float32", always_2d=True)  # [T,C]
            wav = torch.from_numpy(data).transpose(0, 1).contiguous()   # [C,T]

        # Convert to mono: [1,T]
        if wav.dim() == 2 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        elif wav.dim() == 1:
            wav = wav.unsqueeze(0)

        # No resampling: keep original sr

        # Final tensor: [T,1], float32
        x = wav.squeeze(0).unsqueeze(-1).to(torch.float32)
        y = int(self.class_to_idx[row["category"]])
        info = {
            "id": i,
            "length": int(x.shape[0]),
            "sample_rate": int(sr),
            "filename": row["filename"],
            "fold": int(row["fold"]),  # kept as metadata only
        }
        return x, y, info


class ESC50Segmented(Dataset):
    """
    Wrapper dataset that turns an ESC50Raw instance into a dataset of
    non-overlapping time segments.

    Each item corresponds to a (file_idx, offset, length) triple.
    """
    def __init__(
        self,
        base: ESC50Raw,
        segments: List[Tuple[int, int, int]],  # (file_idx, offset, length)
    ):
        self.base = base
        self.segments = segments

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int):
        file_idx, offset, length = self.segments[idx]
        x_full, y, info = self.base[file_idx]  # x_full: [T,1]

        # Slice non-overlapping segment
        x_seg = x_full[offset:offset + length]

        seg_info = dict(info)
        seg_info.update(
            {
                "id": int(idx),  # override with segment id
                "segment_offset": int(offset),
                "segment_length": int(length),
                "file_index": int(file_idx),
            }
        )
        return x_seg, y, seg_info


def _build_segments(
    base: ESC50Raw,
    duration: Optional[float],
    silence_threshold: float = 1e-4,
) -> Tuple[List[Tuple[int, int, int]], List[int]]:
    """
    Build list of (file_idx, offset, length_in_samples) segments.

    - If duration is None: one segment per file (full length).
    - Otherwise: segment length = duration * sample_rate (rounded to nearest int)
    """
    segments: List[Tuple[int, int, int]] = []
    labels: List[int] = []

    n_files = len(base)
    for file_idx in range(n_files):
        x, y, info = base[file_idx]
        length = int(info["length"])
        sr = int(info["sample_rate"])

        if duration is None:
            segments.append((file_idx, 0, length))
            labels.append(y)
            continue

        # Compute number of samples per segment
        seg_len = int(round(duration * sr))
        if seg_len <= 0:
            raise ValueError("duration must be positive.")

        n_segs = length // seg_len
        if n_segs == 0:
            continue

        for k in range(n_segs):
            offset = k * seg_len
            seg = x[offset:offset + seg_len]

            if seg.abs().max().item() < silence_threshold:
                continue

            segments.append((file_idx, offset, seg_len))
            labels.append(y)

    if len(segments) == 0:
        raise RuntimeError(
            "No segments were created. This can happen if `duration` is too large "
            "or segments fall below `silence_threshold`."
        )

    return segments, labels


def build_esc50_raw(
    root: str,
    max_samples: Optional[int] = None,
    *,
    test_ratio: float = 0.2,
    seed: int = 123,
    equal_per_class: bool = False,
    class_count: Optional[int] = None,
    class_filter: Optional[List[str]] = None,
    duration: Optional[float] = None,
) -> Tuple[Dataset, Dataset, List[str], dict]:

    if not (0.0 < test_ratio < 1.0):
        raise ValueError("test_ratio must be in (0, 1).")

    tmp_all = ESC50Raw(root=root)
    all_classes = tmp_all.class_names

    if class_filter is not None:
        unknown = [c for c in class_filter if c not in all_classes]
        if unknown:
            raise ValueError(f"class_filter contains unknown ESC-50 classes: {unknown}")
        chosen_filter = list(class_filter)
    elif class_count is not None:
        if class_count > len(all_classes):
            raise ValueError(
                f"class_count={class_count} exceeds total classes {len(all_classes)}"
            )
        chosen_filter = all_classes[:class_count]
    else:
        chosen_filter = None

    base = ESC50Raw(root=root, class_filter=chosen_filter)
    class_names = base.class_names

    # Build segments using duration instead of time_steps
    segments, labels = _build_segments(base, duration=duration)

    rng = random.Random(seed)

    num_classes = len(class_names)
    label_to_indices: Dict[int, List[int]] = {}
    for idx, y in enumerate(labels):
        label_to_indices.setdefault(y, []).append(idx)

    for idxs in label_to_indices.values():
        rng.shuffle(idxs)

    if equal_per_class:
        min_class_count = min(len(v) for v in label_to_indices.values())
        if max_samples is not None:
            max_per_class = max_samples // num_classes
            n_per_class = min(min_class_count, max_per_class) if max_per_class > 0 else min_class_count
        else:
            n_per_class = min_class_count

        selected_indices = []
        for y, idxs in label_to_indices.items():
            selected_indices.extend(idxs[:n_per_class])
    else:
        if max_samples is not None and max_samples < len(labels):
            selected_indices = stratified_indices_from_labels(labels, max_samples, seed=seed, min_per_class=1)
        else:
            selected_indices = list(range(len(labels)))

    selected_indices = sorted(selected_indices)
    segments = [segments[i] for i in selected_indices]
    labels = [labels[i] for i in selected_indices]

    label_to_selected: Dict[int, List[int]] = {}
    for idx, y in enumerate(labels):
        label_to_selected.setdefault(y, []).append(idx)

    train_segment_ids = []
    test_segment_ids = []
    rng_split = random.Random(seed + 1)

    for y, idxs in label_to_selected.items():
        rng_split.shuffle(idxs)
        n = len(idxs)
        n_test = max(1, int(round(test_ratio * n)))
        test_segment_ids.extend(idxs[:n_test])
        train_segment_ids.extend(idxs[n_test:])

    train_segment_ids.sort()
    test_segment_ids.sort()

    train_segments = [segments[i] for i in train_segment_ids]
    test_segments = [segments[i] for i in test_segment_ids]

    train = ESC50Segmented(base, train_segments)
    test = ESC50Segmented(base, test_segments)

    try:
        _, _, first_info = base[0]
        sr = first_info.get("sample_rate")
    except Exception:
        sr = None

    info = {
        "true_total_segments": len(segments),
        "true_train_total": len(train),
        "true_test_total": len(test),
        "sample_rate": sr,
        "class_names": class_names,
        "test_ratio": test_ratio,
        "equal_per_class": equal_per_class,
        "duration": duration,
    }

    return train, test, class_names, info
