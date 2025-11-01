from __future__ import annotations
import os, pandas as pd
from typing import Optional, Tuple, List
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
    Uses official folds: 1..5. You pick which to train/test in the builder.
    """
    def __init__(self, root: str, folds: List[int], target_sr: int = 16000):
        assert _HAS_TA, "torchaudio required for ESC-50."
        self.root = root
        self.audio_dir = os.path.join(root, "ESC-50-master", "audio")
        self.meta_path = os.path.join(root, "ESC-50-master", "meta", "esc50.csv")
        if not os.path.exists(self.meta_path):
            # lightweight git download if not present
            os.makedirs(root, exist_ok=True)
            import urllib.request, zipfile, io
            url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
            zipbytes = urllib.request.urlopen(url).read()
            with zipfile.ZipFile(io.BytesIO(zipbytes)) as z:
                z.extractall(root)

        df = pd.read_csv(self.meta_path)
        df = df[df["fold"].isin(folds)].reset_index(drop=True)
        self.rows = df
        self.class_names = sorted(df["category"].unique().tolist())
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.target_sr = int(target_sr)

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        import os, math
        import torch
        row = self.rows.iloc[i]

        # ESC-50 audio is flat under ".../audio/"
        fpath = os.path.join(self.audio_dir, row["filename"])

        # 1) Try torchaudio
        wav = None;
        sr = None
        if _HAS_TA:
            try:
                import torchaudio
                wav, sr = torchaudio.load(os.path.normpath(fpath))  # [C,T]
            except Exception:
                wav = None

        # 2) Fallback to soundfile if torchaudio failed or isn't available
        if wav is None:
            try:
                import soundfile as sf
            except Exception as e:
                raise RuntimeError(
                    "Failed to load audio: torchaudio failed and soundfile not installed. "
                    "Install with: pip install soundfile"
                ) from e
            data, sr = sf.read(os.path.normpath(fpath), dtype="float32", always_2d=True)  # [T,C]
            wav = torch.from_numpy(data).transpose(0, 1).contiguous()  # [C,T]

        # mono
        if wav.dim() == 2 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)  # [1,T]
        elif wav.dim() == 1:
            wav = wav.unsqueeze(0)  # [1,T]

        # resample to target_sr
        target_sr = self.target_sr
        if sr != target_sr:
            if _HAS_TA:
                import torchaudio
                wav = torchaudio.functional.resample(wav, sr, target_sr)
            else:
                # lightweight linear resample
                T_old = wav.shape[-1]
                T_new = int(math.ceil(T_old * (target_sr / float(sr))))
                wav = torch.nn.functional.interpolate(
                    wav.unsqueeze(0), size=T_new, mode="linear", align_corners=False
                ).squeeze(0)
            sr = target_sr

        x = wav.squeeze(0).unsqueeze(-1).to(torch.float32)  # [T,1]
        y = int(self.class_to_idx[row["category"]])
        info = {
            "id": i,
            "length": int(x.shape[0]),
            "sample_rate": sr,
            "filename": row["filename"],
            "fold": int(row["fold"]),
        }
        return x, y, info

def build_esc50_raw(root: str,
                    max_samples: Optional[int] = None,
                    *,
                    train_folds: List[int] = [1,2,3,4],
                    test_folds:  List[int] = [5],
                    target_sr: int = 16000,
                    seed: int = 123,
                    min_per_class: int = 3) -> Tuple[Dataset, Dataset, List[str], dict]:

    train = ESC50Raw(root=root, folds=list(train_folds), target_sr=target_sr)
    test  = ESC50Raw(root=root, folds=list(test_folds),  target_sr=target_sr)

    info = {"true_train_total": len(train), "true_test_total": len(test),
            "sample_rate": target_sr, "class_names": train.class_names}

    # Optional stratified subsample (fast via metadata)
    if max_samples is not None:
        from torch.utils.data import Subset
        tr_labels = [int(train.class_to_idx[c]) for c in train.rows["category"]]
        te_labels = [int(test.class_to_idx[c])  for c in test.rows["category"]]
        tr_idx = stratified_indices_from_labels(tr_labels, max_samples, seed=seed, min_per_class=min_per_class)
        te_cap = max(1, min(max(max_samples // 4, 2*len(train.class_names)), len(test)))
        te_idx = stratified_indices_from_labels(te_labels, te_cap, seed=seed, min_per_class=max(1, min_per_class//2))
        train = Subset(train, tr_idx)
        test  = Subset(test,  te_idx)

    return train, test, train.class_names, info
