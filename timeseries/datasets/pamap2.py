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
    """
    def __init__(self, root: str, split_subjects: List[int], min_len: int = 200):
        data_dir = _ensure_pamap2(root)
        prot_dir = os.path.join(data_dir, "Protocol")
        files = [os.path.join(prot_dir, f) for f in os.listdir(prot_dir)
                 if f.startswith("subject") and f.endswith(".dat")]

        self.samples: List[Tuple[torch.Tensor, int, Dict]] = []
        present: Dict[int, str] = {}

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
                if seg.shape[0] >= max(1, int(min_len)):
                    present.setdefault(y_raw, _PAMAP_ACTIVITIES.get(y_raw, f"class_{y_raw}"))
                    t = torch.from_numpy(seg).to(torch.float32)  # [T,27]
                    info = {
                        "id": len(self.samples),
                        "subject": sid,
                        "length": int(t.shape[0]),
                        "activity_raw": y_raw
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

def build_pamap2_raw(root: str,
                     max_samples: Optional[int] = None,
                     *,
                     # Use REAL protocol IDs by default
                     train_subjects: List[int] = [101,102,103,104,105,106,107,108],
                     test_subjects:  List[int] = [109],
                     min_len: int = 200,
                     seed: int = 123,
                     min_per_class: int = 3) -> Tuple[Dataset, Dataset, List[str], dict]:

    # Normalize subjects in case user passes 1..9
    base_dir = os.path.join(root, _PAMAP_ROOTFOLDER) if _have_pamap2(root) else root
    available = _discover_subject_ids(base_dir) if os.path.isdir(os.path.join(base_dir, "Protocol")) else []
    train_subjects = _maybe_map_subjects(list(train_subjects), available)
    test_subjects  = _maybe_map_subjects(list(test_subjects),  available)

    train = PAMAP2Raw(root=root, split_subjects=train_subjects, min_len=min_len)
    test  = PAMAP2Raw(root=root, split_subjects=test_subjects,  min_len=min_len)

    # Guard against empty splits (prevents ZScore.fit crash)
    if len(train) == 0:
        raise RuntimeError(
            "PAMAP2 train split is empty. "
            "Check subject IDs (use 101–109 for Protocol) and/or lower min_len."
        )

    class_names = getattr(train, "class_names", [])

    info = {
        "true_train_total": len(train),
        "true_test_total": len(test),
        "input_dim": 27,
        "class_names": class_names
    }

    # Optional stratified subsample (fast via prebuilt samples list)
    if max_samples is not None:
        from torch.utils.data import Subset
        tr_labels = [int(train[i][1]) for i in range(len(train))]
        te_labels = [int(test[i][1])  for i in range(len(test))]
        tr_idx = stratified_indices_from_labels(tr_labels, max_samples, seed=seed, min_per_class=min_per_class)
        te_cap = max(1, min(max(max_samples // 4, 2*len(class_names)), len(test)))
        te_idx = stratified_indices_from_labels(te_labels, te_cap, seed=seed, min_per_class=max(1, min_per_class//2))
        train = Subset(train, tr_idx)
        test  = Subset(test,  te_idx)

    return train, test, class_names, info
