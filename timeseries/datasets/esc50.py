from __future__ import annotations
import os, io, zipfile
from typing import Optional, Tuple, List
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
    IO-only: returns waveform x:[T,1] at target_sr (default 16000).
    Uses official folds 1..5; choose which to train/test in the builder.
    Will auto-download the dataset (GitHub zip) into:
        {root}/ESC-50-master/{audio,meta}
    """
    def __init__(
        self,
        root: str,
        folds: List[int],
        target_sr: int = 16000,
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
        df = df[df["fold"].isin(folds)].reset_index(drop=True)

        # NEW: filter by subset of classes if requested
        if class_filter is not None:
            df = df[df["category"].isin(class_filter)].reset_index(drop=True)

        if df.empty:
            raise RuntimeError(
                f"No files matched folds={folds}"
                + ("" if class_filter is None else f" and class_filter={class_filter}")
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
        self.target_sr = int(target_sr)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        import math
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

        # Resample to target_sr if needed
        target_sr = self.target_sr
        if int(sr) != target_sr:
            if _HAS_TA:
                wav = torchaudio.functional.resample(wav, int(sr), target_sr)
            else:
                # Minimal linear resample using interpolate
                T_old = wav.shape[-1]
                T_new = int(math.ceil(T_old * (target_sr / float(sr))))
                wav = torch.nn.functional.interpolate(
                    wav.unsqueeze(0), size=T_new, mode="linear", align_corners=False
                ).squeeze(0)
            sr = target_sr

        # Final tensor: [T,1], float32
        x = wav.squeeze(0).unsqueeze(-1).to(torch.float32)
        y = int(self.class_to_idx[row["category"]])
        info = {
            "id": i,
            "length": int(x.shape[0]),
            "sample_rate": int(sr),
            "filename": row["filename"],
            "fold": int(row["fold"]),
        }
        return x, y, info


def build_esc50_raw(
    root: str,
    max_samples: Optional[int] = None,
    *,
    train_folds: List[int] = [1, 2, 3, 4],
    test_folds:  List[int] = [5],
    target_sr: int = 16000,
    seed: int = 123,
    min_per_class: int = 3,
    class_count: Optional[int] = None,
) -> Tuple[Dataset, Dataset, List[str], dict]:
    # If we want to restrict classes, first figure out the global sorted class list
    class_filter: Optional[List[str]] = None
    if class_count is not None:
        # Build a temporary dataset over all folds to get canonical class ordering
        all_folds = sorted(set(train_folds) | set(test_folds))
        tmp = ESC50Raw(root=root, folds=all_folds, target_sr=target_sr)
        all_classes = tmp.class_names  # already sorted

        if class_count > len(all_classes):
            raise ValueError(
                f"class_count={class_count} is larger than total classes={len(all_classes)}"
            )

        # Deterministic subset: first class_count classes
        class_filter = all_classes[:class_count]

    # Now build train/test using the same class_filter (or None for all)
    train = ESC50Raw(
        root=root,
        folds=list(train_folds),
        target_sr=target_sr,
        class_filter=class_filter,
    )
    test  = ESC50Raw(
        root=root,
        folds=list(test_folds),
        target_sr=target_sr,
        class_filter=class_filter,
    )

    # keep a copy BEFORE any Subset wrapping
    class_names = train.class_names  # either full list or restricted list

    info = {
        "true_train_total": len(train),
        "true_test_total": len(test),
        "sample_rate": target_sr,
        "class_names": class_names,
    }

    if max_samples is not None:
        from torch.utils.data import Subset

        tr_labels = [int(train.class_to_idx[c]) for c in train.rows["category"]]
        te_labels = [int(test .class_to_idx[c]) for c in test .rows["category"]]

        tr_idx = stratified_indices_from_labels(
            tr_labels, max_samples, seed=seed, min_per_class=min_per_class
        )
        te_cap = max(1, min(max(max_samples // 4, 2 * len(class_names)), len(test)))
        te_idx = stratified_indices_from_labels(
            te_labels, te_cap, seed=seed, min_per_class=max(1, min_per_class // 2)
        )

        train = Subset(train, tr_idx)
        test  = Subset(test,  te_idx)

    # return the saved class_names, not train.class_names (which may be a Subset)
    return train, test, class_names, info