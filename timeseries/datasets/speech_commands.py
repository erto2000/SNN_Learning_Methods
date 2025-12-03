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
    IO-only: returns waveform [T,1] at the original Speech Commands sampling rate
    (16 kHz), padded/truncated to max_seconds.
    No log-mel, no z-score here.

    Optional class control:
      - class_count: randomly choose this many classes (from all available).
    """
    def __init__(
        self,
        subset: str,
        root: str,
        max_seconds: float = 1.0,
        include_silence: bool = True,
        class_count: Optional[int] = None,
        class_seed: int = 0,
    ):
        assert _HAS_TA, "torchaudio is required for Speech Commands."
        self.ds = SPEECHCOMMANDS(root=root, download=True, subset=subset)

        # Original SpeechCommands sampling rate is 16 kHz
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
        # Random subset via class_count
        if class_count is not None and class_count > 0 and class_count < len(labels):
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
                        # Do NOT resample. Only keep if sampling rate matches.
                        if nsr != self.sample_rate:
                            continue
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
        # No resampling – assume SpeechCommands is already at 16 kHz.
        x = self._prepare(waveform)
        y = self.word_to_idx.get(label.lower(), 0)
        info = {"id": idx, "length": x.shape[0], "sample_rate": sr}
        return x, y, info


def build_sc_raw(
    root: str,
    max_samples: Optional[int] = None,
    include_silence: bool = False,
    *,
    seed: int = 123,
    class_count: Optional[int] = None,
    equal_per_class: bool = False,
) -> Tuple[Dataset, Dataset, List[str], dict]:
    """
    Returns:
      - train: training+validation concatenated
      - test: official testing split
    Both are optionally subsampled STRATIFIED (not head-sliced) using only label lists.

    Options:
      - class_count: pick a random subset of that many classes.
      - equal_per_class: if True, return train and test with the same number of
        samples per class (balanced). When this is enabled, max_samples is ignored.
    """
    os.makedirs(root, exist_ok=True)
    # Use 'seed' also as class_seed for reproducible class selection
    train = SpeechCommandsRaw(
        "training",
        root=root,
        include_silence=include_silence,
        class_count=class_count,
        class_seed=seed,
    )
    valid = SpeechCommandsRaw(
        "validation",
        root=root,
        include_silence=include_silence,
        class_count=class_count,
        class_seed=seed,
    )
    test = SpeechCommandsRaw(
        "testing",
        root=root,
        include_silence=include_silence,
        class_count=class_count,
        class_seed=seed,
    )

    from torch.utils.data import ConcatDataset, Subset
    from collections import defaultdict

    full_train = ConcatDataset([train, valid])

    info = {
        "true_train_total": len(full_train),
        "true_test_total": len(test),
    }

    # ------- FAST label derivation without loading audio -------
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

    # ---------- Equal-per-class balancing (ignores max_samples) ----------
    if equal_per_class:
        def _balanced_indices(labels: List[int], seed: int) -> List[int]:
            rng = random.Random(seed)
            per_class = defaultdict(list)
            for i, y in enumerate(labels):
                per_class[y].append(i)
            # Same number of samples per class
            n_per_class = min(len(v) for v in per_class.values())
            idxs: List[int] = []
            for _, inds in per_class.items():
                rng.shuffle(inds)
                idxs.extend(inds[:n_per_class])
            idxs.sort()
            return idxs

        tr_idx = _balanced_indices(tr_labels, seed)
        te_idx = _balanced_indices(te_labels, seed + 1)

        full_train = Subset(full_train, tr_idx)
        test = Subset(test, te_idx)

        info["balanced_train_per_class"] = len(tr_idx) // len(train.class_names)
        info["balanced_test_per_class"] = len(te_idx) // len(train.class_names)

        return full_train, test, train.class_names, info

    # ---------- Stratified subsampling with max_samples ----------
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

        # Use a fixed internal min_per_class (no user-facing arg any more)
        internal_min_per_class_train = 6
        internal_min_per_class_test = 3

        tr_idx = stratified_indices_from_labels(
            tr_labels, max_tr, seed=seed, min_per_class=internal_min_per_class_train
        )
        te_idx = stratified_indices_from_labels(
            te_labels,
            max_te,
            seed=seed,
            min_per_class=internal_min_per_class_test,
        )

        full_train = Subset(full_train, tr_idx)
        test = Subset(test, te_idx)

    return full_train, test, train.class_names, info
