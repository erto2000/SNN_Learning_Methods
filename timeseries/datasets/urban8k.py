from __future__ import annotations
import os, pandas as pd, shutil, tarfile, time, io
from typing import Optional, Tuple, List
import torch
from torch.utils.data import Dataset

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
    IO-only: UrbanSound8K. Returns waveform x:[T,1] at target_sr.
    Folds 1–8 train, 9–10 test by default.
    """
    def __init__(self, root: str, folds: List[int], target_sr: int = 16000):
        assert _HAS_TA, "torchaudio required for UrbanSound8K."

        # Download/prepare dataset if missing
        if not _have_urban8k(root):
            print("[Urban8K] Dataset not found. Downloading...")
            _download_urban8k(root)

        self.root = root
        self.audio_root = os.path.join(root, "UrbanSound8K", "audio")
        self.meta_path  = os.path.join(root, "UrbanSound8K", "metadata", "UrbanSound8K.csv")

        df = pd.read_csv(self.meta_path)
        df = df[df["fold"].isin(folds)].reset_index(drop=True)
        self.rows = df

        classes = df[["classID","class"]].drop_duplicates().sort_values("classID")
        self.class_names = classes["class"].tolist()
        self.target_sr = int(target_sr)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        fpath = os.path.join(self.audio_root, f"fold{int(r['fold'])}", r["slice_file_name"])
        wav, sr = torchaudio.load(fpath)
        wav = wav.mean(dim=0, keepdim=True)  # mono

        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)

        x = wav.squeeze(0).unsqueeze(-1).to(torch.float32)  # [T,1]
        y = int(r["classID"])
        info = {
            "id": i, "length": x.shape[0], "sample_rate": self.target_sr,
            "filename": r["slice_file_name"], "fold": int(r["fold"])
        }
        return x, y, info


# ----------------------------------------------------------------------------- #
# Factory
# ----------------------------------------------------------------------------- #

def build_urban8k_raw(root: str,
                      max_samples: Optional[int] = None,
                      *,
                      train_folds: List[int] = list(range(1,9)),
                      test_folds:  List[int] = [9,10],
                      target_sr: int = 16000,
                      seed: int = 123,
                      min_per_class: int = 3) -> Tuple[Dataset, Dataset, List[str], dict]:

    train = Urban8KRaw(root=root, folds=list(train_folds), target_sr=target_sr)
    test  = Urban8KRaw(root=root, folds=list(test_folds),  target_sr=target_sr)

    # capture BEFORE any Subset wrapping
    class_names = train.class_names

    info = {
        "true_train_total": len(train),
        "true_test_total": len(test),
        "sample_rate": target_sr,
        "class_names": class_names,
    }

    if max_samples is not None:
        from torch.utils.data import Subset
        tr_labels = [int(c) for c in train.rows["classID"]]
        te_labels = [int(c) for c in test.rows["classID"]]

        tr_idx = stratified_indices_from_labels(tr_labels, max_samples,
                                                seed=seed, min_per_class=min_per_class)
        te_cap = max(1, min(max(max_samples // 4, 2*len(class_names)), len(test)))
        te_idx = stratified_indices_from_labels(te_labels, te_cap,
                                                seed=seed, min_per_class=max(1, min_per_class//2))

        train = Subset(train, tr_idx)
        test  = Subset(test,  te_idx)

    # return the captured class_names, not train.class_names
    return train, test, class_names, info