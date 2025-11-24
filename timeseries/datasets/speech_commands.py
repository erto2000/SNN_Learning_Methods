# timeseries/datasets/speech_commands.py
from __future__ import annotations
from typing import Optional, List, Tuple
import os, random, glob
import torch
from torch.utils.data import Dataset

from ._subsample import stratified_indices_from_labels

try:
    import torchaudio
    from torchaudio.datasets import SPEECHCOMMANDS
    _HAS_TA = True
except Exception:
    _HAS_TA = False


class SpeechCommandsRaw(Dataset):
    """
    IO-only: returns waveform [T,1] at 16k, padded/truncated to max_seconds.
    No log-mel, no z-score here.

    Optional class control:
      - classes: explicit list of class names to keep (case-insensitive).
      - class_count: randomly choose this many classes (from all available).
    """
    def __init__(
        self,
        subset: str,
        root: str,
        max_seconds: float = 1.0,
        include_silence: bool = True,
        classes: Optional[List[str]] = None,
        class_count: Optional[int] = None,
        class_seed: int = 0,
    ):
        assert _HAS_TA, "torchaudio is required for Speech Commands."
        self.ds = SPEECHCOMMANDS(root=root, download=True, subset=subset)
        self.sample_rate = 16000
        self.max_len = int(max_seconds * self.sample_rate)
        self.include_silence = bool(include_silence)

        root_path = getattr(self.ds, "_path", None) or self.ds._path

        # All available labels from directory structure
        labels = sorted(
            [
                d
                for d in os.listdir(root_path)
                if os.path.isdir(os.path.join(root_path, d)) and not d.startswith("_")
            ]
        )

        # Optionally add "silence" label if background noise directory is present
        if (
            os.path.isdir(os.path.join(root_path, "_background_noise_"))
            and "silence" not in labels
            and include_silence
        ):
            labels.append("silence")

        # --------- Class subset selection ---------
        # 1) Explicit list of classes (case-insensitive filter)
        if classes is not None:
            wanted = {c.lower() for c in classes}
            labels = [l for l in labels if l.lower() in wanted]
            if not labels:
                raise ValueError(
                    "No classes remain after applying 'classes' filter. "
                    f"Requested: {classes}"
                )

        # 2) Or a random subset via class_count
        elif class_count is not None and class_count > 0 and class_count < len(labels):
            rng = random.Random(class_seed)
            # sample from labels as they are (directory names), then sort for stability
            labels = sorted(rng.sample(labels, class_count))

        # Final class list & mapping
        self.class_names = labels
        self.word_to_idx = {w.lower(): i for i, w in enumerate(self.class_names)}

        # Index list; may include -1 to represent synthetic "silence" samples.
        # Build indices ONLY for files whose label remains in word_to_idx.
        self._indices: List[int] = []
        self._file_labels: List[str] = []

        walker = self.ds._walker  # list of relative file paths
        for j, rel in enumerate(walker):
            label = os.path.basename(os.path.dirname(os.path.join(root_path, rel))).lower()
            self._file_labels.append(label)
            if label in self.word_to_idx:
                self._indices.append(j)

        # Build a bank of silence waveforms from background noise (optional)
        self._silence_bank: List[torch.Tensor] = []
        if include_silence and ("silence" in self.word_to_idx):
            try:
                noise_dir = os.path.join(root_path, "_background_noise_")
                if os.path.isdir(noise_dir):
                    for nf in glob.glob(os.path.join(noise_dir, "*.wav")):
                        wav, nsr = torchaudio.load(nf)
                        wav = wav.mean(dim=0, keepdim=True)
                        if nsr != self.sample_rate:
                            wav = torchaudio.functional.resample(
                                wav, nsr, self.sample_rate
                            )
                        self._silence_bank.append(wav.squeeze(0).clone())
                if self._silence_bank:
                    # extend indices with -1 entries (each maps to a synthetic silence sample)
                    self._indices.extend([-1] * len(self._silence_bank))
            except Exception:
                # If anything goes wrong, just skip synthetic silence; core dataset still works
                self._silence_bank = []

    def __len__(self) -> int:
        return len(self._indices)

    def _prepare(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 2:  # [C,T] -> mono
            wav = wav.mean(dim=0)
        if wav.numel() > self.max_len:
            wav = wav[:self.max_len]
        elif wav.numel() < self.max_len:
            wav = torch.nn.functional.pad(wav, (0, self.max_len - wav.numel()))
        return wav.unsqueeze(-1).to(torch.float32)  # [T,1]

    def __getitem__(self, i):
        idx = self._indices[i]
        if idx == -1 and self._silence_bank:
            wav = random.choice(self._silence_bank)
            x = self._prepare(wav)
            y = self.word_to_idx["silence"]
            info = {"id": i, "length": x.shape[0], "sample_rate": self.sample_rate}
            return x, y, info

        waveform, sr, label, *_ = self.ds[idx]
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        x = self._prepare(waveform)
        y = self.word_to_idx.get(label.lower(), 0)
        info = {"id": idx, "length": x.shape[0], "sample_rate": self.sample_rate}
        return x, y, info


def build_sc_raw(
    root: str,
    max_samples: Optional[int] = None,
    include_silence: bool = False,
    *,
    seed: int = 123,
    min_per_class: int = 6,
    classes: Optional[List[str]] = None,
    class_count: Optional[int] = None,
    class_seed: int = 0,
) -> Tuple[Dataset, Dataset, List[str], dict]:
    """
    Returns:
      - train: training+validation concatenated
      - test: official testing split
    Both are optionally subsampled STRATIFIED (not head-sliced) using only label lists.

    Additional options:
      - classes: explicit list of class names to keep (case-insensitive).
      - class_count: pick a random subset of that many classes.
    """
    os.makedirs(root, exist_ok=True)
    train = SpeechCommandsRaw(
        "training",
        root=root,
        include_silence=include_silence,
        classes=classes,
        class_count=class_count,
        class_seed=class_seed,
    )
    valid = SpeechCommandsRaw(
        "validation",
        root=root,
        include_silence=include_silence,
        classes=classes,
        class_count=class_count,
        class_seed=class_seed,
    )
    test = SpeechCommandsRaw(
        "testing",
        root=root,
        include_silence=include_silence,
        classes=classes,
        class_count=class_count,
        class_seed=class_seed,
    )

    from torch.utils.data import ConcatDataset, Subset

    full_train = ConcatDataset([train, valid])

    info = {
        "true_train_total": len(full_train),
        "true_test_total": len(test),
    }

    if max_samples is not None:
        max_tr = max(1, min(max_samples, len(full_train)))
        # keep test smaller but ensure at least a few per class (handles silence too)
        max_te = max(
            1,
            min(
                max(max_samples // 4, 6 * len(train.class_names)),
                len(test),
            ),
        )

        # ------- FAST label derivation without loading audio -------
        # Use the dataset's own indices and cached labels to map to class ids.
        def _labels_sc_split(ds: SpeechCommandsRaw) -> List[int]:
            word_to_idx = ds.word_to_idx
            silence_idx = word_to_idx.get("silence", None)
            out: List[int] = []
            for idx in ds._indices:
                if idx == -1:
                    # synthetic silence sample
                    if silence_idx is not None:
                        out.append(int(silence_idx))
                else:
                    label = ds._file_labels[idx]
                    out.append(int(word_to_idx[label]))
            return out

        # labels for ConcatDataset([train, valid]) == concatenate lists
        tr_labels = _labels_sc_split(train) + _labels_sc_split(valid)
        te_labels = _labels_sc_split(test)

        tr_idx = stratified_indices_from_labels(
            tr_labels, max_tr, seed=seed, min_per_class=min_per_class
        )
        te_idx = stratified_indices_from_labels(
            te_labels,
            max_te,
            seed=seed,
            min_per_class=max(3, min_per_class // 2),
        )

        full_train = Subset(full_train, tr_idx)
        test = Subset(test, te_idx)

    return full_train, test, train.class_names, info
