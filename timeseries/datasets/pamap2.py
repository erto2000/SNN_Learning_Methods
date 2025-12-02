from __future__ import annotations
import os, io, zipfile, urllib.request, time, shutil
from typing import Optional, Tuple, List, Dict
import numpy as np
import torch
from torch.utils.data import Dataset
from ._subsample import stratified_indices_from_labels

_PAMAP_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip"
_PAMAP_ROOTFOLDER = "PAMAP2_Dataset"

# Activity map from official metadata
_PAMAP_ACTIVITIES: Dict[int, str] = {
    1: "lying", 2: "sitting", 3: "standing", 4: "walking",
    5: "running", 6: "cycling", 7: "Nordic walking", 9: "watching TV",
    10: "computer work", 11: "car driving", 12: "ascending stairs",
    13: "descending stairs", 16: "vacuum cleaning", 17: "ironing",
    18: "folding laundry", 19: "house cleaning", 20: "playing soccer",
    24: "rope jumping"
}
_IGNORE_LABEL = 0  # "other"/null label in raw stream

# ----- Column layout (per README): keep acc16g(3)+gyro(3)+mag(3) for hand/chest/ankle = 27 dims -----
def _imu_block_indices(base: int):
    acc16 = [base+1, base+2, base+3]
    gyro  = [base+10, base+11, base+12]
    mag   = [base+13, base+14, base+15]
    return acc16 + gyro + mag

# After time(0), act(1), hr(2) => hand at 3, chest at 20, ankle at 37
_HAND_BASE, _CHEST_BASE, _ANKLE_BASE = 3, 20, 37
_PAMAP_KEEP_COLS = (
    _imu_block_indices(_HAND_BASE)
    + _imu_block_indices(_CHEST_BASE)
    + _imu_block_indices(_ANKLE_BASE)
)  # 27 dims

# --------------------------- Download / Extract utils ---------------------------

def _have_pamap2(root: str) -> bool:
    base = os.path.join(root, _PAMAP_ROOTFOLDER)
    prot = os.path.join(base, "Protocol")
    return os.path.isdir(prot) and any(
        f.startswith("subject") and f.endswith(".dat") for f in os.listdir(prot)
    )

def _safe_unzip(zip_bytes: bytes, out_dir: str):
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        # basic zip slip protection
        ab_out = os.path.abspath(out_dir)
        for m in z.namelist():
            dst_path = os.path.abspath(os.path.join(out_dir, m))
            if not (dst_path == ab_out or dst_path.startswith(ab_out + os.sep)):
                raise RuntimeError("Unsafe path in zip (zip-slip).")
        z.extractall(out_dir)

def _download_with_retries(url: str, *, timeout: int = 60, max_retries: int = 3, backoff: int = 3) -> bytes:
    last_err = None
    # Try requests streaming (if available)
    try:
        import requests
        for attempt in range(1, max_retries+1):
            try:
                with requests.get(url, stream=True, timeout=timeout) as r:
                    r.raise_for_status()
                    chunks = []
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            chunks.append(chunk)
                    return b"".join(chunks)
            except Exception as e:
                last_err = e
                time.sleep(backoff * attempt)
    except Exception as e:
        last_err = e  # requests missing or failed setup; fall back to urllib

    # urllib fallback
    for attempt in range(1, max_retries+1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "python"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as e:
            last_err = e
            time.sleep(backoff * attempt)

    raise RuntimeError(f"Failed to download: {url}. Last error: {last_err}")

def _ensure_pamap2(root: str) -> str:
    out_dir = os.path.join(root, _PAMAP_ROOTFOLDER)
    if _have_pamap2(root):
        return out_dir

    os.makedirs(root, exist_ok=True)
    print("[PAMAP2] Downloading dataset (UCI) ...")
    try:
        zip_bytes = _download_with_retries(_PAMAP_URL, timeout=90, max_retries=4, backoff=4)
    except Exception as e:
        raise RuntimeError(
            "Could not download PAMAP2 automatically (network/timeouts).\n"
            f"Manual download URL:\n  {_PAMAP_URL}\n"
            f"Then extract into: {root}\n"
            f"Expected: {root}/{_PAMAP_ROOTFOLDER}/Protocol/subject*.dat"
        ) from e

    print("[PAMAP2] Extracting ...")
    _safe_unzip(zip_bytes, root)

    # Some archives may use "PAMAP2 Dataset" (space). Normalize name.
    alt_dir = os.path.join(root, "PAMAP2 Dataset")
    if not os.path.isdir(out_dir) and os.path.isdir(alt_dir):
        try:
            os.rename(alt_dir, out_dir)
        except Exception:
            shutil.copytree(alt_dir, out_dir, dirs_exist_ok=True)
            shutil.rmtree(alt_dir, ignore_errors=True)

    if not _have_pamap2(root):
        raise RuntimeError(
            f"PAMAP2 extraction looks incomplete in {root}. "
            f"Ensure '{_PAMAP_ROOTFOLDER}/Protocol/subject*.dat' exist."
        )
    return out_dir

def _discover_subject_ids(base_dir: str) -> List[int]:
    prot = os.path.join(base_dir, "Protocol")
    ids = []
    for f in os.listdir(prot):
        if f.startswith("subject") and f.endswith(".dat"):
            num = "".join(ch for ch in f if ch.isdigit())
            if num:
                ids.append(int(num))
    return sorted(ids)

def _maybe_map_subjects(req: List[int], available: List[int]) -> List[int]:
    """If user passed [1..9] but available are [101..109], map by +100."""
    if not req or not available:
        return req
    if max(available) >= 100 and max(req) < 100:
        return [s + 100 for s in req]
    return req

# --------------------------------- Dataset -------------------------------------

class PAMAP2Raw(Dataset):
    """
    IO-only: each sample is a contiguous segment of a single activity from a subject file.
    Returns x:[T, D=27] (float32), y:int (activity id remapped to 0..C-1), info: dict.

    If time_steps is not None, long segments are split into non-overlapping
    windows of length time_steps, each becoming one sample.
    """
    def __init__(self,
                 root: str,
                 split_subjects: List[int],
                 min_len: int = 200,
                 time_steps: Optional[int] = None):
        data_dir = _ensure_pamap2(root)
        prot_dir = os.path.join(data_dir, "Protocol")
        files = [os.path.join(prot_dir, f) for f in os.listdir(prot_dir)
                 if f.startswith("subject") and f.endswith(".dat")]

        self.samples: List[Tuple[torch.Tensor, int, Dict]] = []
        present: Dict[int, str] = {}
        self.time_steps = time_steps

        # NaN handling helper
        def _clean_nan(a: np.ndarray) -> np.ndarray:
            # Replace crazy values & NaNs with 0 (z-score later)
            return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

        for fp in files:
            sid = int(''.join([c for c in os.path.basename(fp) if c.isdigit()]) or -1)
            if sid not in split_subjects:
                continue
            try:
                arr = np.loadtxt(fp)  # (N, 54)
            except Exception:
                continue
            if arr.ndim != 2 or arr.shape[1] < 54:
                continue

            arr = _clean_nan(arr)
            labels = arr[:, 1].astype(int)
            X = arr[:, _PAMAP_KEEP_COLS].astype(np.float32)  # (N,27)

            # Segment into contiguous runs of constant activity
            start = 0
            N = len(labels)
            while start < N:
                y_raw = int(labels[start])
                if y_raw == _IGNORE_LABEL:
                    start += 1
                    continue
                end = start + 1
                while end < N and int(labels[end]) == y_raw:
                    end += 1

                seg = X[start:end]
                seg_len = seg.shape[0]

                # Skip very short segments
                if seg_len < max(1, int(min_len)):
                    start = end
                    continue

                present.setdefault(y_raw, _PAMAP_ACTIVITIES.get(y_raw, f"class_{y_raw}"))

                if self.time_steps is None:
                    # Original behavior: whole contiguous segment is one sample
                    t = torch.from_numpy(seg).to(torch.float32)  # [T,27]
                    info = {
                        "id": len(self.samples),
                        "subject": sid,
                        "length": int(t.shape[0]),
                        "activity_raw": y_raw
                    }
                    self.samples.append((t, y_raw, info))
                else:
                    # New behavior: split into equal-length windows
                    T = int(self.time_steps)
                    if T <= 0:
                        raise ValueError("time_steps must be positive if not None.")

                    num_chunks = seg_len // T  # drop remainder by default
                    for k in range(num_chunks):
                        chunk = seg[k * T:(k + 1) * T]
                        if chunk.shape[0] != T:
                            continue  # safety check; should be exact

                        t = torch.from_numpy(chunk).to(torch.float32)  # [T,27]
                        info = {
                            "id": len(self.samples),
                            "subject": sid,
                            "length": int(t.shape[0]),
                            "activity_raw": y_raw,
                            "segment_index": k,  # index within this contiguous run
                        }
                        self.samples.append((t, y_raw, info))

                start = end

        # Finalize class mapping
        self.class_ids = sorted(present.keys())
        self.class_names = [_PAMAP_ACTIVITIES.get(cid, f"class_{cid}") for cid in self.class_ids]
        self.cid_to_y = {cid: i for i, cid in enumerate(self.class_ids)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        x, cid, info = self.samples[i]
        y = self.cid_to_y[cid]
        info = {**info, "label_name": self.class_names[y]}
        return x, y, info

# ---------------------------------- Factory ------------------------------------

def build_pamap2_raw(
    root: str,
    max_samples: Optional[int] = None,
    *,
    min_len: int = 100,
    seed: int = 123,
    equal_per_class: bool = False,
    time_steps: Optional[int] = None,
    test_ratio: float = 0.2,
) -> Tuple[Dataset, Dataset, List[str], dict]:
    """
    Build PAMAP2 train/test datasets from a single unified pool of all subjects.

    - The dataset is first built over *all* available subjects in the Protocol folder.
    - Then it is split into train/test using `test_ratio`.

    If equal_per_class is True:
      - A single full dataset is built.
      - Exactly 'per_class' samples per class are chosen globally (optionally capped by max_samples).
      - For each class, samples are split into train/test according to 'test_ratio'.
      - Both train and test are balanced over classes (up to rounding).

    If equal_per_class is False:
      - A single full dataset is built.
      - Optionally, a stratified subsample of size max_samples is selected.
      - This pool is then split into train/test according to 'test_ratio'.
    """

    if not (0.0 < test_ratio < 1.0):
        raise ValueError("test_ratio must be in (0, 1).")

    rng = np.random.default_rng(seed)

    # Discover all available subject IDs under PAMAP2_Dataset/Protocol
    base_dir = os.path.join(root, _PAMAP_ROOTFOLDER) if _have_pamap2(root) else root
    prot_dir = os.path.join(base_dir, "Protocol")
    available = _discover_subject_ids(base_dir) if os.path.isdir(prot_dir) else []

    if not available:
        raise RuntimeError(
            "No PAMAP2 subjects found. Make sure the dataset is extracted to:\n"
            f"  {root}/{_PAMAP_ROOTFOLDER}/Protocol/subject*.dat"
        )

    # Build a single unified dataset over all subjects
    full = PAMAP2Raw(
        root=root,
        split_subjects=available,
        min_len=min_len,
        time_steps=time_steps,
    )

    if len(full) == 0:
        raise RuntimeError(
            "PAMAP2 dataset is empty. "
            "Try lowering min_len or checking that subject .dat files are valid."
        )

    # Labels over the full dataset (in remapped space 0..C-1)
    all_labels = np.array([int(full[i][1]) for i in range(len(full))], dtype=int)
    classes = np.unique(all_labels)
    if classes.size == 0:
        raise RuntimeError("No classes found in dataset.")

    num_classes = int(classes.size)
    class_names = getattr(full, "class_names", [])

    from torch.utils.data import Subset

    # -------------------------------------------------------------------------
    # Path 1: equal_per_class == True -> balanced per class, then split
    # -------------------------------------------------------------------------
    if equal_per_class:
        # Determine per-class budget, possibly capped by max_samples
        class_counts = [int(np.sum(all_labels == c)) for c in classes]
        if max_samples is None:
            # Use all data but capped by the smallest class
            per_class = min(class_counts)
        else:
            if max_samples < num_classes:
                raise RuntimeError(
                    f"max_samples={max_samples} is too small to allocate at least "
                    f"one sample per class for {num_classes} classes."
                )
            budget_per_class = max_samples // num_classes
            per_class = min(budget_per_class, *class_counts)

        if per_class <= 0:
            raise RuntimeError(
                "Not enough samples per class to enforce equal_per_class "
                "with the given data and max_samples."
            )

        train_idx: List[int] = []
        test_idx: List[int] = []

        for c in classes:
            c_idx = np.where(all_labels == c)[0]
            # guaranteed per_class <= len(c_idx)
            chosen = rng.choice(c_idx, size=per_class, replace=False)
            rng.shuffle(chosen)

            # How many test samples for this class?
            n_test = int(round(test_ratio * per_class))
            if n_test <= 0 and per_class > 1:
                n_test = 1
            if n_test >= per_class and per_class > 1:
                n_test = per_class - 1

            test_idx.extend(chosen[:n_test].tolist())
            train_idx.extend(chosen[n_test:].tolist())

        rng.shuffle(train_idx)
        rng.shuffle(test_idx)

        train = Subset(full, train_idx)
        test = Subset(full, test_idx)

        info = {
            "true_total": len(full),
            "true_train_total": len(train),
            "true_test_total": len(test),
            "input_dim": 27,
            "class_names": class_names,
            "per_class": per_class,
            "test_ratio": test_ratio,
        }
        return train, test, class_names, info

    # -------------------------------------------------------------------------
    # Path 2: equal_per_class == False -> optionally stratified subsample, then split
    # -------------------------------------------------------------------------

    # If max_samples is specified, choose a stratified subset of the full dataset.
    if max_samples is not None:
        from ._subsample import stratified_indices_from_labels

        total = min(max_samples, len(full))
        if total <= 0:
            raise RuntimeError("max_samples is too small (no samples would be selected).")

        chosen = stratified_indices_from_labels(all_labels.tolist(), total, seed=seed)
        chosen = np.array(chosen, dtype=int)
    else:
        chosen = np.arange(len(full), dtype=int)

    # Shuffle chosen indices and split by test_ratio
    rng.shuffle(chosen)
    n_total = len(chosen)
    n_test = int(round(test_ratio * n_total))
    if n_test <= 0 and n_total > 1:
        n_test = 1
    if n_test >= n_total and n_total > 1:
        n_test = n_total - 1

    test_idx = chosen[:n_test]
    train_idx = chosen[n_test:]

    train = Subset(full, train_idx.tolist())
    test = Subset(full, test_idx.tolist())

    info = {
        "true_total": len(full),
        "true_train_total": len(train),
        "true_test_total": len(test),
        "input_dim": 27,
        "class_names": class_names,
        "test_ratio": test_ratio,
        "max_samples_used": len(chosen),
    }

    return train, test, class_names, info
