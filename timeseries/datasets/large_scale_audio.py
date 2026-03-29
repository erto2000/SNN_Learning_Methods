from __future__ import annotations
import os
import zipfile
import random
import shutil
import urllib.request
import urllib.parse
import ssl
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import torch
from torch.utils.data import Dataset

from ._subsample import stratified_indices_from_labels

try:
    import torchaudio
    _HAS_TA = True
except Exception:
    _HAS_TA = False

_DATASET_DIR = "large_scale_audio"
_FIGSHARE_ARTICLE_ID = 19291472
_CLASS_NAMES = ["emergency_vehicle_siren", "road_noise"]


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi  # type: ignore
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        try:
            return ssl.create_default_context()
        except Exception:
            return ssl._create_unverified_context()


def _download_url(url: str, dst: str, *, timeout: int = 120):
    req = urllib.request.Request(url, headers={"User-Agent": "python"})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as r, open(dst, "wb") as f:
            shutil.copyfileobj(r, f)
    except ssl.SSLError:
        with urllib.request.urlopen(req, timeout=timeout, context=ssl._create_unverified_context()) as r, open(dst, "wb") as f:
            shutil.copyfileobj(r, f)


def _safe_extract_zip(zip_path: str, out_dir: str):
    with zipfile.ZipFile(zip_path, "r") as zf:
        ab_root = os.path.abspath(out_dir)
        for name in zf.namelist():
            dest = os.path.abspath(os.path.join(out_dir, name))
            if not (dest == ab_root or dest.startswith(ab_root + os.sep)):
                raise RuntimeError("Unsafe path in zip (zip-slip).")
        zf.extractall(out_dir)


def _have_dataset(root: str) -> bool:
    ds_dir = os.path.join(root, _DATASET_DIR)
    if not os.path.isdir(ds_dir):
        return False
    return any(Path(ds_dir).rglob("*.wav"))


def _figshare_files(article_id: int) -> List[Dict[str, str]]:
    api = f"https://api.figshare.com/v2/articles/{article_id}"
    req = urllib.request.Request(api, headers={"User-Agent": "python", "Accept": "application/json"})
    import json
    try:
        with urllib.request.urlopen(req, timeout=60, context=_ssl_context()) as r:
            meta = json.loads(r.read().decode("utf-8"))
    except ssl.SSLError:
        with urllib.request.urlopen(req, timeout=60, context=ssl._create_unverified_context()) as r:
            meta = json.loads(r.read().decode("utf-8"))

    files: List[Dict[str, str]] = []
    for f in meta.get("files", []):
        url = f.get("download_url")
        if not url:
            continue
        name = f.get("name") or os.path.basename(urllib.parse.urlparse(url).path) or "download.bin"
        files.append({"name": str(name), "url": str(url)})
    return files


def _ensure_large_scale_audio(root: str) -> str:
    ds_dir = os.path.join(root, _DATASET_DIR)
    if _have_dataset(root):
        return ds_dir

    os.makedirs(ds_dir, exist_ok=True)
    files = _figshare_files(_FIGSHARE_ARTICLE_ID)
    if not files:
        raise RuntimeError("Could not resolve Figshare file URLs for Large-Scale Audio Dataset.")

    for i, item in enumerate(files):
        name = item["name"]
        url = item["url"]
        local_name = os.path.join(ds_dir, name)
        if not os.path.exists(local_name):
            print(f"[large_scale_audio] Downloading file {i + 1}/{len(files)}: {name} ...")
            _download_url(url, local_name)

        is_zip_name = name.lower().endswith(".zip")
        if is_zip_name or zipfile.is_zipfile(local_name):
            print(f"[large_scale_audio] Extracting archive {i + 1}/{len(files)}: {name} ...")
            _safe_extract_zip(local_name, ds_dir)
        else:
            print(f"[large_scale_audio] Kept non-archive file: {name}")

    if not _have_dataset(root):
        raise RuntimeError(
            "Large-Scale Audio Dataset download/extraction finished, but no WAV files were found. "
            "Figshare may have changed file packaging; inspect the downloaded files under the dataset directory."
        )
    return ds_dir


class LargeScaleAudioRaw(Dataset):
    """
    IO-only loader for the Large-Scale Audio Dataset for Emergency Vehicle Sirens and Road Noises.
    Returns waveform x:[T,1] and binary label.
    """
    def __init__(self, root: str):
        self.root = _ensure_large_scale_audio(root)
        self.class_names = list(_CLASS_NAMES)
        self.items: List[Tuple[str, int]] = []

        for wav in Path(self.root).rglob("*.wav"):
            rel = str(wav).lower()
            if "emergency" in rel or "siren" in rel or "ambulance" in rel:
                y = 0
            elif "road" in rel or "noise" in rel or "traffic" in rel:
                y = 1
            else:
                continue
            self.items.append((str(wav), y))

        if not self.items:
            raise RuntimeError(
                f"No labeled WAV files found under {self.root}. Expected folders containing siren/emergency/ambulance or road/noise/traffic in their names."
            )

        self.items.sort()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        fpath, y = self.items[i]
        wav = None
        sr = None
        if _HAS_TA:
            try:
                wav, sr = torchaudio.load(fpath)
            except Exception:
                wav = None
        if wav is None:
            try:
                import soundfile as sf
            except Exception as e:
                raise RuntimeError("Failed to load audio. Install torchaudio or soundfile.") from e
            data, sr = sf.read(fpath, dtype="float32", always_2d=True)
            wav = torch.from_numpy(data).transpose(0, 1).contiguous()

        if wav.dim() == 2 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        elif wav.dim() == 1:
            wav = wav.unsqueeze(0)

        x = wav.squeeze(0).unsqueeze(-1).to(torch.float32)
        info = {
            "id": i,
            "length": int(x.shape[0]),
            "sample_rate": int(sr),
            "filename": os.path.basename(fpath),
            "path": fpath,
        }
        return x, int(y), info


class _SegmentedAudio(Dataset):
    def __init__(self, base: LargeScaleAudioRaw, segments: List[Tuple[int, int, int]]):
        self.base = base
        self.segments = segments
        self.class_names = getattr(base, "class_names", None)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx: int):
        file_idx, start, seg_len = self.segments[idx]
        x, y, info = self.base[file_idx]
        seg = x[start:start + seg_len]
        out_info = dict(info)
        out_info.update({
            "id": idx,
            "file_index": int(file_idx),
            "segment_offset": int(start),
            "segment_length": int(seg_len),
        })
        return seg, y, out_info


def _build_segments(base: LargeScaleAudioRaw, duration: Optional[float]) -> Tuple[List[Tuple[int, int, int]], List[int]]:
    segments: List[Tuple[int, int, int]] = []
    labels: List[int] = []
    for i in range(len(base)):
        _, y, info = base[i]
        T = int(info["length"])
        sr = int(info["sample_rate"])
        if duration is None:
            segments.append((i, 0, T))
            labels.append(y)
            continue
        seg_len = max(1, int(round(duration * sr)))
        n = T // seg_len
        if n == 0:
            continue
        for k in range(n):
            segments.append((i, k * seg_len, seg_len))
            labels.append(y)
    if not segments:
        raise RuntimeError("No audio segments were created. Try a shorter duration or use duration=None.")
    return segments, labels


def build_large_scale_audio_raw(
    root: str,
    max_samples: Optional[int] = None,
    *,
    seed: int = 123,
    duration: Optional[float] = 3.0,
    test_ratio: float = 0.2,
    equal_per_class: bool = False,
) -> Tuple[Dataset, Dataset, List[str], dict]:
    base = LargeScaleAudioRaw(root=root)
    segments, labels = _build_segments(base, duration=duration)

    rng = random.Random(seed)
    by_label: Dict[int, List[int]] = {}
    for i, y in enumerate(labels):
        by_label.setdefault(int(y), []).append(i)
    for idxs in by_label.values():
        rng.shuffle(idxs)

    if equal_per_class:
        n_per = min(len(v) for v in by_label.values())
        if max_samples is not None:
            n_per = min(n_per, max_samples // max(1, len(base.class_names)))
        selected = []
        for idxs in by_label.values():
            selected.extend(idxs[:n_per])
        selected.sort()
    else:
        if max_samples is not None and max_samples < len(labels):
            selected = stratified_indices_from_labels(labels, max_samples, seed=seed, min_per_class=1)
            selected.sort()
        else:
            selected = list(range(len(labels)))

    segments = [segments[i] for i in selected]
    labels = [labels[i] for i in selected]

    by_label = {}
    for i, y in enumerate(labels):
        by_label.setdefault(int(y), []).append(i)

    train_ids: List[int] = []
    test_ids: List[int] = []
    split_rng = random.Random(seed + 1)
    for idxs in by_label.values():
        split_rng.shuffle(idxs)
        n_test = max(1, int(round(len(idxs) * test_ratio)))
        test_ids.extend(idxs[:n_test])
        train_ids.extend(idxs[n_test:])

    train = _SegmentedAudio(base, [segments[i] for i in sorted(train_ids)])
    test = _SegmentedAudio(base, [segments[i] for i in sorted(test_ids)])
    info = {
        "true_total_files": len(base),
        "true_total_segments": len(segments),
        "duration_seconds": duration,
    }
    return train, test, list(base.class_names), info
