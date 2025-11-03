# timeseries/datasets/mitbih.py
from __future__ import annotations
from typing import Optional, Tuple, List, Dict
import os
import time
import shutil
import numpy as np
import torch
from torch.utils.data import Dataset

from ._subsample import stratified_indices_from_labels

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
_AAMI_TO_INT = {c:i for i,c in enumerate(_AAMI_NAMES)}

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


def _download_record_files(dst_dir: str, rec_str: str, *, max_retries: int = 5, timeout_connect: int = 10, timeout_read: int = 30):
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
        for u in ["B","KB","MB","GB","TB"]:
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
            allowed_methods=frozenset(["GET","HEAD"])
        )
        s.mount("https://", HTTPAdapter(max_retries=retry))
        s.mount("http://",  HTTPAdapter(max_retries=retry))
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
                            sys.stdout.write(f"\r[MITBIH]   {os.path.basename(outpath):10s}  {pct:5.1f}% "
                                             f"({_human(downloaded)}/{_human(total)}) at {_human(rate)}/s")
                        else:
                            sys.stdout.write(f"\r[MITBIH]   {os.path.basename(outpath):10s}  {_human(downloaded)}")
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

def _fetch_record(root: str, rec: int, *, local_only: bool = False):
    """
    Locate or (optionally) download record files for `rec`, returning waveform/ann.
    Priority:
      1) {root}/mitbih/
      2) {root}/mitbih/1.0.0/
      3) {root}/mitbih/mitdb/    (backward compat)
      4) {root}/                  (legacy fallback if previously dumped into root)
      5) Remote read (if not local_only)
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

    # 2) Download into {root}/mitbih (unless local_only)
    if local_only:
        raise FileNotFoundError(
            f"Record {rec_str} not found locally under {base}. "
            "Set local_only=False to allow downloading."
        )

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
    def __init__(self, root: str, records: List[int], win_samples: int = 360, *, local_only: bool = False):
        if not _HAS_WFDB:
            raise RuntimeError("wfdb is required: pip install wfdb")

        self.samples: List[Tuple[torch.Tensor,int,Dict]] = []
        self.win_left = win_samples // 2
        self.win_right = win_samples - self.win_left

        for rec in records:
            x, fs, rlocs, syms = _fetch_record(root, rec, local_only=local_only)
            beats = _beats_from_record(x, fs, rlocs, syms, self.win_left, self.win_right)
            for i, (seg, y) in enumerate(beats):
                info = {"id": len(self.samples), "record": rec, "length": int(seg.shape[0]), "fs": fs}
                self.samples.append((seg, y, info))

        self.class_names = _AAMI_NAMES

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        x, y, info = self.samples[i]
        return x.to(torch.float32), int(y), info


def build_mitbih_raw(root: str,
                     max_samples: Optional[int] = None,
                     *,
                     train_records: List[int] = _DS1,
                     test_records:  List[int] = _DS2,
                     win_samples: int = 360,
                     seed: int = 123,
                     min_per_class: int = 50,
                     local_only: bool = False) -> Tuple[Dataset, Dataset, List[str], dict]:
    """
    Factory:
      - Builds MITBIHRaw for train/test using record lists.
      - Stores/reads data under {root}/mitbih/.
      - If local_only=True, will error if records are missing locally (no download).
    """
    train = MITBIHRaw(root=root, records=list(train_records),
                      win_samples=win_samples, local_only=local_only)
    test  = MITBIHRaw(root=root, records=list(test_records),
                      win_samples=win_samples, local_only=local_only)

    info = {
        "true_train_total": len(train),
        "true_test_total": len(test),
        "sample_rate": 360,
        "window": win_samples,
        "class_names": train.class_names
    }

    if max_samples is not None:
        from torch.utils.data import Subset
        tr_labels = [int(train[i][1]) for i in range(len(train))]
        te_labels = [int(test[i][1])  for i in range(len(test))]

        tr_idx = stratified_indices_from_labels(tr_labels, max_samples,
                                                seed=seed, min_per_class=min_per_class)
        te_cap = max(1, min(max(max_samples // 4, 5*len(train.class_names)), len(test)))
        te_idx = stratified_indices_from_labels(te_labels, te_cap,
                                                seed=seed, min_per_class=max(5, min_per_class//2))

        train = Subset(train, tr_idx)
        test  = Subset(test,  te_idx)

    return train, test, _AAMI_NAMES, info
