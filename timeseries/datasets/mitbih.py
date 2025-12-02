# timeseries/datasets/mitbih.py
from __future__ import annotations
from typing import Optional, Tuple, List, Dict
import os
import time
import shutil
import numpy as np
import torch
from torch.utils.data import Dataset

from ._subsample import stratified_indices_from_labels  # still available if needed elsewhere

# Requires: pip install wfdb
try:
    import wfdb
    _HAS_WFDB = True
except Exception:
    _HAS_WFDB = False

# Recommended patient-wise split (DS1 train, DS2 test) per literature
# DS1 (train): records with odd numbers from 101..234 excluding some; DS2 (test): even
_DS1 = [101,106,108,109,112,114,115,116,118,119,122,124,201,203,205,207,208,209,215,220,223,230]
_DS2 = [100,103,105,111,113,117,121,123,200,202,210,212,213,214,219,221,222,228,231,232,233,234]

# Map beat symbols to AAMI classes (N, S, V, F, Q)
_AAMI_MAP: Dict[str, str] = {
    # N
    'N':'N', 'L':'N', 'R':'N', 'e':'N', 'j':'N',
    # S
    'A':'S', 'a':'S', 'J':'S', 'S':'S',
    # V
    'V':'V', 'E':'V',
    # F
    'F':'F',
    # Q (unknown/paced/others)
    'Q':'Q', '/':'Q', 'f':'Q', '?':'Q', 'P':'Q'
}
# Explicit extras seen in MIT-BIH annotations
for s in ['B','x','|','p','t','u','~','*','D','S','T','+','!','[',']','"','@','=']:
    _AAMI_MAP.setdefault(s, 'Q')

_AAMI_NAMES = ['N','S','V','F','Q']
_AAMI_TO_INT = {c: i for i, c in enumerate(_AAMI_NAMES)}

# Keep everything for this dataset under a dedicated subfolder
_MITBIH_DIR = "mitbih"   # -> {root}/mitbih/


def _have_files(dir_path: str, rec_str: str) -> bool:
    return (
        os.path.isfile(os.path.join(dir_path, f"{rec_str}.dat")) and
        os.path.isfile(os.path.join(dir_path, f"{rec_str}.hea")) and
        os.path.isfile(os.path.join(dir_path, f"{rec_str}.atr"))
    )


def _read_record(path_no_ext: str):
    """
    Read a record (local or remote) using wfdb rdsamp/rdann.
    Returns: x (np.float32), fs (int), rlocs (np.ndarray[int]), symbols (List[str])
    """
    sig, fields = wfdb.rdsamp(path_no_ext)     # (np.ndarray, dict)
    ann = wfdb.rdann(path_no_ext, "atr")
    x = sig[:, 0].astype(np.float32)           # lead 0
    fs = int(fields["fs"])
    rlocs = ann.sample.astype(int)
    symbols = ann.symbol
    return x, fs, rlocs, symbols


def _download_record_files(dst_dir: str, rec_str: str, *, max_retries: int = 5,
                           timeout_connect: int = 10, timeout_read: int = 30):
    """
    Download {rec}.dat/.hea/.atr from PhysioNet with resume, progress, and retries.
    Shows progress even on slow links so it never looks "stuck".
    """
    import sys, time
    from contextlib import suppress

    files = [f"{rec_str}.dat", f"{rec_str}.hea", f"{rec_str}.atr"]
    base_url = "https://physionet.org/files/mitdb/1.0.0/"

    os.makedirs(dst_dir, exist_ok=True)

    def _human(n):
        for u in ["B", "KB", "MB", "GB", "TB"]:
            if n < 1024:
                return f"{n:.1f}{u}"
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
        s.mount("http://", HTTPAdapter(max_retries=retry))
        s.headers.update({"User-Agent": "python"})
        return s

    def _progress_get(url: str, outpath: str) -> bool:
        """
        Resume-aware streaming download with visible progress.
        """
        import requests
        s = _requests_session()
        # discover size (if server provides it)
        with s.head(url, allow_redirects=True, timeout=(timeout_connect, timeout_read)) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length") or 0)

        tmp = outpath + ".part"
        pos = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        headers = {"Range": f"bytes={pos}-"} if pos else {}

        chunk = 1024 * 1024
        t0 = time.time()
        with s.get(url, stream=True, headers=headers, timeout=(timeout_connect, timeout_read)) as r:
            if r.status_code not in (200, 206):
                r.raise_for_status()
            mode = "ab" if pos else "wb"
            downloaded = pos
            with open(tmp, mode) as f:
                next_tick = 0.0
                for b in r.iter_content(chunk_size=chunk):
                    if not b:
                        continue
                    f.write(b)
                    downloaded += len(b)
                    now = time.time()
                    if now >= next_tick:
                        next_tick = now + 0.3
                        if total:
                            pct = 100.0 * downloaded / total
                            rate = downloaded / max(1e-6, (now - t0))
                            sys.stdout.write(
                                f"\r[MITBIH]   {os.path.basename(outpath):10s}  {pct:5.1f}% "
                                f"({_human(downloaded)}/{_human(total)}) at {_human(rate)}/s"
                            )
                        else:
                            sys.stdout.write(
                                f"\r[MITBIH]   {os.path.basename(outpath):10s}  {_human(downloaded)}"
                            )
                        sys.stdout.flush()

        # finalize
        if total and os.path.getsize(tmp) < total:
            return False
        with suppress(Exception):
            if os.path.exists(outpath):
                os.remove(outpath)
        os.replace(tmp, outpath)
        sys.stdout.write("\n")
        return True

    for fname in files:
        url = base_url + fname
        out = os.path.join(dst_dir, fname)
        if os.path.isfile(out):
            print(f"[MITBIH]   {fname:10s} already present")
            continue

        print(f"[MITBIH] Downloading {fname} from PhysioNet...")
        ok = False
        # retry loop (our requests session already retries on transient errors,
        # but we also retry the whole transfer if the file remained partial)
        for attempt in range(1, max_retries + 1):
            if _progress_get(url, out):
                ok = True
                break
            sleep_s = 1.5 * attempt
            print(f"[MITBIH]   retrying in {sleep_s:.1f}s (attempt {attempt}/{max_retries})")
            time.sleep(sleep_s)

        if not ok:
            # clean partial
            with suppress(Exception):
                os.remove(out + ".part")
            raise RuntimeError(f"Failed to download {url} -> {out}")


def _fetch_record(root: str, rec: int):
    """
    Locate or download record files for `rec`, returning waveform/ann.
    Priority:
      1) {root}/mitbih/
      2) {root}/mitbih/1.0.0/
      3) {root}/mitbih/mitdb/    (backward compat)
      4) {root}/                  (legacy fallback if previously dumped into root)
      5) Remote read (if local files not found)
    New downloads are ALWAYS placed into {root}/mitbih/.
    """
    if not _HAS_WFDB:
        raise RuntimeError("wfdb is required: pip install wfdb")

    rec_str = f"{rec}"
    base = os.path.join(root, _MITBIH_DIR)

    candidates = [
        os.path.join(base),               # {root}/mitbih/
        os.path.join(base, "1.0.0"),      # {root}/mitbih/1.0.0/
        os.path.join(base, "mitdb"),      # {root}/mitbih/mitdb/  (backward compat)
        root,                             # fallback: legacy dump into {root}
    ]

    # 1) Local search
    for d in candidates:
        if _have_files(d, rec_str):
            return _read_record(os.path.join(d, rec_str))

    # 2) Download into {root}/mitbih
    os.makedirs(base, exist_ok=True)
    _download_record_files(base, rec_str)

    # 3) Try local again (freshly downloaded into base)
    if _have_files(base, rec_str):
        return _read_record(os.path.join(base, rec_str))

    # 4) Final fallback: direct remote read (wfdb can read from URLs)
    try:
        return _read_record(f"https://physionet.org/files/mitdb/1.0.0/{rec_str}")
    except Exception as e:
        raise FileNotFoundError(
            f"Could not locate or fetch record {rec_str}. "
            f"Tried local {candidates} and PhysioNet."
        ) from e


def _beats_from_record(x: np.ndarray, fs: int, rlocs: np.ndarray, symbols: List[str],
                       win_left: int, win_right: int):
    out = []
    for r, s in zip(rlocs, symbols):
        cls = _AAMI_TO_INT.get(_AAMI_MAP.get(s, 'Q'), _AAMI_TO_INT['Q'])
        a = max(0, r - win_left)
        b = min(len(x), r + win_right)
        seg = np.zeros((win_left + win_right,), dtype=np.float32)
        seg[:b-a] = x[a:b]
        out.append((torch.from_numpy(seg).unsqueeze(-1), cls))  # [T,1], int
    return out


class MITBIHRaw(Dataset):
    """
    IO-only: beat-centered windows x:[T,1] around R-peaks, label=AAMI class (0..4).
    """
    def __init__(self, root: str, records: List[int], win_samples: int = 360):
        if not _HAS_WFDB:
            raise RuntimeError("wfdb is required: pip install wfdb")

        self.samples: List[Tuple[torch.Tensor, int, Dict]] = []
        self.win_left = win_samples // 2
        self.win_right = win_samples - self.win_left

        for rec in records:
            x, fs, rlocs, syms = _fetch_record(root, rec)
            beats = _beats_from_record(x, fs, rlocs, syms, self.win_left, self.win_right)
            for i, (seg, y) in enumerate(beats):
                info = {
                    "id": len(self.samples),
                    "record": rec,
                    "length": int(seg.shape[0]),
                    "fs": fs,
                }
                self.samples.append((seg, y, info))

        # By default, 5-class AAMI
        self.class_names = list(_AAMI_NAMES)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        x, y, info = self.samples[i]
        return x.to(torch.float32), int(y), info


def build_mitbih_raw(
    root: str,
    max_samples: Optional[int] = None,
    *,
    records: Optional[List[int]] = None,
    test_ratio: float = 0.2,
    win_samples: int = 360,
    seed: int = 123,
    two_class: bool = False,
    equal_per_class: bool = False,
) -> Tuple[Dataset, Dataset, List[str], dict]:
    """
    Factory:
      - Builds a single MITBIHRaw dataset from given records (or all DS1+DS2 by default).
      - Splits it into train/test according to `test_ratio` using stratified splitting.
      - Optionally:
          * `two_class=True`: collapse to 2 classes: 0 = normal (N), 1 = others (S,V,F,Q).
          * `equal_per_class=True`: use the same number of samples per class (balanced),
            then perform stratified split so train/test remain class-balanced.
      - Stores/reads data under {root}/mitbih/.
    """
    rng = np.random.RandomState(seed)

    # Use all standard DS1+DS2 records by default
    if records is None:
        records = sorted(set(_DS1 + _DS2))

    # 1) Build full dataset
    full = MITBIHRaw(root=root, records=list(records), win_samples=win_samples)

    # 2) Two-class conversion (normal vs others) if requested
    if two_class:
        new_samples = []
        for seg, y, info in full.samples:
            # 0 = N (normal), 1 = others (S,V,F,Q)
            new_y = 0 if y == _AAMI_TO_INT['N'] else 1
            # Optionally keep original label info (not strictly necessary)
            info = dict(info)
            info["orig_aami_label"] = _AAMI_NAMES[y]
            new_samples.append((seg, new_y, info))
        full.samples = new_samples
        full.class_names = ['N', 'O']  # normal vs others
    else:
        full.class_names = list(_AAMI_NAMES)

    num_classes = len(full.class_names)

    # 3) Prepare labels and per-class indices
    labels = [int(y) for (_, y, _) in full.samples]
    indices_by_class: Dict[int, List[int]] = {c: [] for c in range(num_classes)}
    for idx, y in enumerate(labels):
        if y in indices_by_class:
            indices_by_class[y].append(idx)
        else:
            # In case some unexpected label sneaks in, put it into "others"
            # but this should not normally happen.
            last_cls = num_classes - 1
            indices_by_class.setdefault(last_cls, []).append(idx)

    # Shuffle indices within each class
    for c in indices_by_class:
        rng.shuffle(indices_by_class[c])

    # 4) Optional subsampling with/without equal_per_class
    if equal_per_class:
        # Same number of samples per class overall
        min_count = min(len(v) for v in indices_by_class.values() if len(v) > 0)
        if max_samples is not None:
            # Target per class from max_samples
            target_per_class = max(1, max_samples // num_classes)
            per_class = min(min_count, target_per_class)
        else:
            per_class = min_count

        base_indices: List[int] = []
        for c, idxs in indices_by_class.items():
            take = min(per_class, len(idxs))
            base_indices.extend(idxs[:take])
    else:
        # Use all samples, or a random subset of full dataset if max_samples is given
        all_indices = list(range(len(full)))
        rng.shuffle(all_indices)
        if max_samples is not None:
            base_indices = all_indices[:max_samples]
        else:
            base_indices = all_indices

    # 5) Stratified train/test split from base_indices
    per_class_base: Dict[int, List[int]] = {c: [] for c in range(num_classes)}
    for idx in base_indices:
        y = labels[idx]
        if y not in per_class_base:
            # again, safeguard
            y = num_classes - 1
        per_class_base[y].append(idx)

    train_indices: List[int] = []
    test_indices: List[int] = []

    for c, idxs in per_class_base.items():
        if not idxs:
            continue
        rng.shuffle(idxs)
        n_c = len(idxs)
        # number of test samples from this class
        n_test_c = int(round(test_ratio * n_c))
        # Ensure at least 1 test if possible and at least 1 train if class has >1 sample
        if n_test_c <= 0 and n_c > 1:
            n_test_c = 1
        if n_test_c >= n_c and n_c > 1:
            n_test_c = n_c - 1

        test_indices.extend(idxs[:n_test_c])
        train_indices.extend(idxs[n_test_c:])

    # 6) Build Subset datasets
    from torch.utils.data import Subset
    train = Subset(full, train_indices)
    test = Subset(full, test_indices)

    info = {
        "total_samples": len(full),
        "train_samples": len(train),
        "test_samples": len(test),
        "sample_rate": 360,
        "window": win_samples,
        "class_names": list(full.class_names),
        "two_class": two_class,
        "equal_per_class": equal_per_class,
        "test_ratio": test_ratio,
    }

    return train, test, list(full.class_names), info
