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


def _download_urban8k(root: str, *, max_retries: int = 3, timeout: int = 60):
    """
    Robust UrbanSound8K download via Zenodo.
    Falls back to manual instructions if fails.
    """
    os.makedirs(root, exist_ok=True)
    url = "https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz?download=1"
    tgz_path = os.path.join(root, "UrbanSound8K.tar.gz")

    def _stream_download() -> bool:
        try:
            import requests
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(tgz_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            return True
        except Exception:
            return False

    def _urllib_download() -> bool:
        try:
            import urllib.request
            req = urllib.request.Request(url, headers={"User-Agent": "python"})
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(tgz_path, "wb") as out:
                shutil.copyfileobj(resp, out)
            return True
        except Exception:
            return False

    # Retry logic
    ok = False
    for attempt in range(1, max_retries + 1):
        if _stream_download() or _urllib_download():
            ok = True
            break
        time.sleep(2 * attempt)  # backoff

    if not ok:
        raise RuntimeError(
            "UrbanSound8K download failed due to network issues.\n"
            "Please download manually:\n"
            "  https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz\n"
            f"Place it in: {root} and extract so structure is:\n"
            "UrbanSound8K/\n"
            "  audio/fold1 ... fold10\n"
            "  metadata/UrbanSound8K.csv\n"
        )

    # Extract
    with tarfile.open(tgz_path, mode="r:gz") as tar:
        _safe_extract(tar, path=root)

    if not _have_urban8k(root):
        raise RuntimeError(
            f"UrbanSound8K extraction incomplete in {root}.\n"
            "Ensure folder is: UrbanSound8K/audio/fold*/... and metadata/"
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