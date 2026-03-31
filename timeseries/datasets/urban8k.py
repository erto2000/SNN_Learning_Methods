from __future__ import annotations
import os, pandas as pd, tarfile, random
from typing import Optional, Tuple, List
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset

try:
    import torchaudio
    _HAS_TA = True
except Exception:
    _HAS_TA = False

from ._subsample import stratified_indices_from_labels

# ----------------------------------------------------------------------------- #
# Helpers: dataset check + download
# ----------------------------------------------------------------------------- #

def _have_urban8k(root: str) -> bool:
    audio_root = os.path.join(root, "UrbanSound8K", "audio")
    meta_path  = os.path.join(root, "UrbanSound8K", "metadata", "UrbanSound8K.csv")
    if not (os.path.isfile(meta_path) and os.path.isdir(audio_root)):
        return False
    # Check we have fold dirs
    for d in os.listdir(audio_root):
        if os.path.isdir(os.path.join(audio_root, d)) and d.startswith("fold"):
            return True
    return False


def _safe_extract(tarobj, path="."):
    # Prevent path traversal vulnerability
    def is_within_directory(directory, target):
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
        return os.path.commonprefix([abs_directory, abs_target]) == abs_directory
    for member in tarobj.getmembers():
        member_path = os.path.join(path, member.name)
        if not is_within_directory(path, member_path):
            raise Exception("Attempted Path Traversal in Tar File")
    tarobj.extractall(path)


def _download_urban8k(root: str, *, max_retries: int = 5, timeout_connect: int = 15, timeout_read: int = 60):
    """
    Robust UrbanSound8K download with progress + resume.
    Primary: Zenodo; Fallback: Kaggle (requires kaggle CLI and credentials).
    """
    import math, sys
    from contextlib import suppress
    os.makedirs(root, exist_ok=True)

    zenodo_url = "https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz?download=1"
    tgz_path   = os.path.join(root, "UrbanSound8K.tar.gz")
    tmp_path   = tgz_path + ".part"

    def _human(n):  # bytes -> nice string
        for u in ["B","KB","MB","GB","TB"]:
            if n < 1024: return f"{n:.1f}{u}"
            n /= 1024
        return f"{n:.1f}PB"

    def _requests_session():
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        s = requests.Session()
        retry = Retry(
            total=max_retries,
            connect=max_retries,
            read=max_retries,
            backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET", "HEAD"])
        )
        s.mount("https://", HTTPAdapter(max_retries=retry))
        s.mount("http://",  HTTPAdapter(max_retries=retry))
        s.headers.update({"User-Agent": "python"})
        return s

    def _progress_dl(url: str) -> bool:
        import requests, time
        s = _requests_session()

        # ask server for size
        with s.head(url, allow_redirects=True, timeout=(timeout_connect, timeout_read)) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", "0") or 0)

        # resume if tmp exists
        pos = 0
        if os.path.exists(tmp_path):
            pos = os.path.getsize(tmp_path)
            if total and pos > total:
                pos = 0  # start fresh if weird
        headers = {}
        if pos:
            headers["Range"] = f"bytes={pos}-"

        chunk = 1024 * 1024  # 1 MB
        t0 = time.time()
        with s.get(url, stream=True, headers=headers, timeout=(timeout_connect, timeout_read)) as r:
            if r.status_code not in (200, 206):
                r.raise_for_status()
            mode = "ab" if pos else "wb"
            with open(tmp_path, mode) as f:
                downloaded = pos
                # progress every ~0.5s
                next_tick = 0.0
                for b in r.iter_content(chunk_size=chunk):
                    if not b:
                        continue
                    f.write(b)
                    downloaded += len(b)
                    now = time.time()
                    if now >= next_tick:
                        next_tick = now + 0.5
                        if total:
                            pct = 100.0 * downloaded / total
                            rate = downloaded / max(1e-9, (now - t0))
                            sys.stdout.write(f"\r[Urban8K] Downloading: {pct:5.1f}% "
                                             f"({_human(downloaded)}/{_human(total)}) "
                                             f"at {_human(rate)}/s")
                        else:
                            sys.stdout.write(f"\r[Urban8K] Downloading: {_human(downloaded)}")
                        sys.stdout.flush()
        # finalize
        if total and os.path.getsize(tmp_path) != total:
            # some servers omit content-length for Range; accept if >= total
            if os.path.getsize(tmp_path) < total:
                return False
        if os.path.exists(tgz_path):
            os.remove(tgz_path)
        os.replace(tmp_path, tgz_path)
        print("\n[Urban8K] Download complete.")
        return True

    def _kaggle_fallback() -> bool:
        """
        Try Kaggle mirror if available:
        'urbansound8k' dataset slug varies; commonly 'urbansound8k/urbansound8k'
        Requires: pip install kaggle && set KAGGLE_USERNAME/KAGGLE_KEY or kaggle.json.
        """
        with suppress(Exception):
            import shutil, subprocess
            if shutil.which("kaggle") is None:
                return False
            # This downloads into current dir; then move/rename
            print("[Urban8K] Trying Kaggle fallback...")
            # A common slug is 'urbansound8k/urbansound8k'; adjust if your org mirror differs.
            cmd = ["kaggle", "datasets", "download", "-d", "urbansound8k/urbansound8k", "-f", "UrbanSound8K.tar.gz", "-p", root]
            subprocess.check_call(cmd)
            return os.path.exists(tgz_path)
        return False

    # Attempt primary with resume
    ok = _progress_dl(zenodo_url)
    if not ok:
        # clean partial if we will try mirror
        with suppress(Exception):
            os.remove(tmp_path)
        ok = _kaggle_fallback()
    if not ok or not os.path.exists(tgz_path):
        raise RuntimeError(
            "UrbanSound8K download failed.\n"
            "Tried Zenodo (with resume) and Kaggle fallback. "
            "If you're behind a corporate proxy/firewall, try manual download and place the file here:\n"
            f"  {tgz_path}\n"
        )

    # Extract with safety + small progress
    print("[Urban8K] Extracting (this can take a few minutes)...")
    with tarfile.open(tgz_path, mode="r:gz") as tar:
        members = tar.getmembers()
        total_members = len(members)
        for i, m in enumerate(members, 1):
            # path traversal guard
            member_path = os.path.join(root, m.name)
            abs_directory = os.path.abspath(root)
            abs_target = os.path.abspath(member_path)
            if not os.path.commonprefix([abs_directory, abs_target]) == abs_directory:
                raise Exception("Attempted Path Traversal in Tar File")
            tar.extract(m, path=root)
            if i % 200 == 0 or i == total_members:
                sys.stdout.write(f"\r[Urban8K] Extracted {i}/{total_members}")
                sys.stdout.flush()
    print("\n[Urban8K] Extract done.")

    if not _have_urban8k(root):
        raise RuntimeError(
            f"UrbanSound8K extraction incomplete in {root}.\n"
            "Expected: UrbanSound8K/audio/fold*/... and UrbanSound8K/metadata/UrbanSound8K.csv"
        )

# ----------------------------------------------------------------------------- #
# UrbanSound8K Raw Dataset
# ----------------------------------------------------------------------------- #

class Urban8KRaw(Dataset):
    """
    IO-only: UrbanSound8K.

    Returns waveform x:[T,1] at the original dataset sample rate (no resampling).
    Optional `duration` (in seconds) will center-crop or zero-pad each clip.
    """
    def __init__(self, root: str, folds: List[int], duration: Optional[float] = None, class_filter: Optional[List[str]] = None):
        assert _HAS_TA, "torchaudio required for UrbanSound8K."

        # Download/prepare dataset if missing
        if not _have_urban8k(root):
            print("[Urban8K] Dataset not found. Downloading...")
            _download_urban8k(root)

        self.root = root
        self.audio_root = os.path.join(root, "UrbanSound8K", "audio")
        self.meta_path  = os.path.join(root, "UrbanSound8K", "metadata", "UrbanSound8K.csv")
        self.duration = duration

        df = pd.read_csv(self.meta_path)
        df = df[df["fold"].isin(folds)].reset_index(drop=True)

        if class_filter is not None:
            unknown = [c for c in class_filter if c not in df["class"].values]
            if unknown:
                raise ValueError(f"class_filter contains unknown UrbanSound8K classes: {unknown}")
            df = df[df["class"].isin(class_filter)].reset_index(drop=True)
            if df.empty:
                raise ValueError(f"No samples found for class_filter={class_filter} in the selected folds.")
            present = set(df["class"].unique())
            self.class_names = [c for c in class_filter if c in present]
            name_to_idx = {c: i for i, c in enumerate(self.class_names)}
            df = df.copy()
            df["classID"] = df["class"].map(name_to_idx)
        else:
            classes = df[["classID", "class"]].drop_duplicates().sort_values("classID")
            self.class_names = classes["class"].tolist()

        self.rows = df

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        fpath = os.path.join(self.audio_root, f"fold{int(r['fold'])}", r["slice_file_name"])
        wav, sr = torchaudio.load(fpath)       # [C, T]
        wav = wav.mean(dim=0, keepdim=True)    # mono -> [1, T]

        # Optional fixed duration in seconds
        if self.duration is not None:
            num_samples = int(round(self.duration * sr))
            if num_samples > 0:
                T = wav.shape[1]
                if T > num_samples:
                    # Center crop
                    start = (T - num_samples) // 2
                    wav = wav[:, start:start + num_samples]
                elif T < num_samples:
                    # Zero-pad at the end
                    pad = num_samples - T
                    wav = F.pad(wav, (0, pad))

        x = wav.squeeze(0).unsqueeze(-1).to(torch.float32)  # [T,1]
        y = int(r["classID"])
        info = {
            "id": i,
            "length": x.shape[0],
            "sample_rate": sr,               # original file sample rate
            "filename": r["slice_file_name"],
            "fold": int(r["fold"]),
        }
        return x, y, info

# ----------------------------------------------------------------------------- #
# Factory
# ----------------------------------------------------------------------------- #

def build_urban8k_raw(
    root: str,
    max_samples: Optional[int] = None,
    *,
    test_ratio: float = 0.2,
    seed: int = 123,
    equal_per_class: bool = False,
    duration: Optional[float] = None,
    class_filter: Optional[List[str]] = None,
) -> Tuple[Dataset, Dataset, List[str], dict]:
    """
    Build UrbanSound8K datasets from a single combined pool (folds 1–10).

    When equal_per_class=True:
        - For each class, we select the same number of samples (if possible),
        - Then split *within that class* into train/test with test_ratio.
        => Both train and test end up balanced across classes (up to rounding).
    """

    full = Urban8KRaw(root=root, folds=list(range(1, 11)), duration=duration, class_filter=class_filter)

    class_names = full.class_names
    num_classes = len(class_names)
    n_total = len(full)
    labels = [int(c) for c in full.rows["classID"]]

    rng = random.Random(seed)

    # ------------------------------------------------------------------ #
    # equal_per_class=True  → class-wise balancing + class-wise split
    # ------------------------------------------------------------------ #
    if equal_per_class:
        # Build per-class index lists
        indices_per_class: dict[int, List[int]] = {cid: [] for cid in range(num_classes)}
        for idx, lab in enumerate(labels):
            if lab in indices_per_class:
                indices_per_class[lab].append(idx)

        # Decide how many per class
        if max_samples is not None:
            per_class_target = max_samples // num_classes
            per_class_target = max(1, per_class_target)
        else:
            # Use the minimum class count so all classes can contribute equally
            per_class_target = min(len(v) for v in indices_per_class.values())

        train_indices: List[int] = []
        test_indices:  List[int] = []

        for cid, idxs in indices_per_class.items():
            if not idxs:
                continue
            idxs = idxs[:]         # copy
            rng.shuffle(idxs)

            k = min(per_class_target, len(idxs))
            selected = idxs[:k]

            if k <= 1:
                # If only one sample, put it in train
                train_indices.extend(selected)
                continue

            n_test = int(round(test_ratio * k))
            n_test = max(1, min(k - 1, n_test))  # keep at least one in each split

            test_indices.extend(selected[:n_test])
            train_indices.extend(selected[n_test:])

        used_indices = sorted(set(train_indices) | set(test_indices))

    # ------------------------------------------------------------------ #
    # equal_per_class=False → stratified subset, then global split
    # ------------------------------------------------------------------ #
    else:
        all_indices = list(range(n_total))

        if max_samples is not None and max_samples < n_total:
            subset_indices = stratified_indices_from_labels(
                labels,
                max_samples,
                seed=seed,
                min_per_class=1,
            )
            all_indices = subset_indices

        used_indices = all_indices[:]
        rng.shuffle(used_indices)

        if len(used_indices) <= 1:
            train_indices = used_indices
            test_indices = []
        else:
            n_test = int(round(test_ratio * len(used_indices)))
            n_test = max(1, min(len(used_indices) - 1, n_test))
            test_indices = used_indices[:n_test]
            train_indices = used_indices[n_test:]

    # Build datasets
    train_ds = Subset(full, train_indices)
    test_ds  = Subset(full, test_indices)

    # Info (UrbanSound8K original SR)
    info = {
        "total_examples": n_total,
        "used_examples": len(used_indices),
        "num_train": len(train_indices),
        "num_test": len(test_indices),
        "test_ratio": test_ratio,
        "class_names": class_names,
        "num_classes": num_classes,
        "sample_rate": 44_100,
        "duration": duration,
        "max_samples": max_samples,
        "equal_per_class": equal_per_class,
    }

    return train_ds, test_ds, class_names, info
