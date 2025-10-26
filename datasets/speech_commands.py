# datasets/speech_commands.py
from typing import Optional, Tuple, List
import os, random, glob
import torch
from torch.utils.data import Dataset

try:
    import torchaudio
    from torchaudio.datasets import SPEECHCOMMANDS
    _HAS_TA = True
except Exception:
    _HAS_TA = False

class _SCWrapper(Dataset):
    """
    Returns log-mel features [T,mels] as raw per-sample tensors.
    No segmentation/padding here.
    """
    def __init__(self, subset: str, root: str, mels: int = 64,
                 win_len: int = 25, hop_len: int = 10,
                 target_words: Optional[List[str] | str] = "auto",
                 include_silence: bool = True, max_seconds: float = 1.0):
        assert _HAS_TA, "torchaudio is required for Speech Commands."
        self.ds = SPEECHCOMMANDS(root=root, download=True, subset=subset)
        self.sample_rate = 16000
        self.mels = mels
        self.max_len = int(max_seconds * self.sample_rate)

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate, n_fft=1024,
            win_length=int(win_len * self.sample_rate / 1000),
            hop_length=int(hop_len * self.sample_rate / 1000),
            n_mels=mels
        )
        self.amplog = torchaudio.transforms.AmplitudeToDB()
        self.include_silence = bool(include_silence)

        # labels
        if target_words is None or target_words == "auto":
            root_path = getattr(self.ds, "_path", None) or self.ds._path
            labels = sorted([d for d in os.listdir(root_path)
                             if os.path.isdir(os.path.join(root_path, d)) and not d.startswith("_")])
            if os.path.isdir(os.path.join(root_path, "_background_noise_")) and "silence" not in labels:
                labels.append("silence")
            self.target_words = labels
        elif isinstance(target_words, list):
            self.target_words = target_words
        else:
            raise ValueError("target_words must be 'auto', a list, or None.")

        self.word_to_idx = {w.lower(): i for i, w in enumerate(self.target_words)}

        # Indices + silence bank
        self._indices = list(range(len(self.ds)))
        self._silence_bank: List[torch.Tensor] = []
        if self.include_silence and ("silence" in self.word_to_idx):
            try:
                noise_dir = os.path.join(getattr(self.ds, "_path", ""), "_background_noise_")
                if os.path.isdir(noise_dir):
                    for nf in glob.glob(os.path.join(noise_dir, "*.wav")):
                        wav, nsr = torchaudio.load(nf)  # [C,N]
                        wav = wav.mean(dim=0, keepdim=True)
                        if nsr != self.sample_rate:
                            wav = torchaudio.functional.resample(wav, nsr, self.sample_rate)
                        wav = wav.squeeze(0)
                        if wav.numel() >= self.max_len:
                            for _ in range(6):
                                start = random.randint(0, wav.numel() - self.max_len)
                                self._silence_bank.append(wav[start:start+self.max_len].clone())
                if self._silence_bank:
                    self._indices.extend([-1] * len(self._silence_bank))
            except Exception:
                self._silence_bank = []

    def __len__(self): return len(self._indices)

    def _wav_to_logmel(self, wav: torch.Tensor) -> torch.Tensor:
        mel = self.melspec(wav.unsqueeze(0))
        return self.amplog(mel).squeeze(0).transpose(0, 1)  # [T,mels]

    def __getitem__(self, i):
        idx = self._indices[i]
        if idx == -1 and self._silence_bank:
            wav = self._silence_bank[i % len(self._silence_bank)]
            logmel = self._wav_to_logmel(wav)
            y = self.word_to_idx["silence"]
            return logmel, y

        waveform, sr, label, *_ = self.ds[idx]
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        wav = waveform[0]

        if wav.numel() > self.max_len:
            wav = wav[:self.max_len]
        elif wav.numel() < self.max_len:
            wav = torch.nn.functional.pad(wav, (0, self.max_len - wav.numel()))

        logmel = self._wav_to_logmel(wav)
        y = self.word_to_idx.get(label.lower(), None)
        if y is None:
            # extremely rare if 'auto' missed a folder — just pick next
            return self.__getitem__((i + 1) % len(self))
        return logmel, y

def build_sc(root: str, max_samples: Optional[int] = None, mels: int = 64, include_silence: bool = True, **_):
    assert _HAS_TA, "torchaudio is required for Speech Commands."
    os.makedirs(root, exist_ok=True)

    train = _SCWrapper("training",   root=root, mels=mels, include_silence=include_silence)
    valid = _SCWrapper("validation", root=root, mels=mels, include_silence=include_silence)
    test  = _SCWrapper("testing",    root=root, mels=mels, include_silence=include_silence)

    full_train = torch.utils.data.ConcatDataset([train, valid])

    if max_samples is not None:
        max_tr = max(1, min(max_samples, len(full_train)))
        max_te = max(1, min(max_samples // 4 if (max_samples and max_samples > 4) else 1, len(test)))
        full_train = torch.utils.data.Subset(full_train, range(max_tr))
        test = torch.utils.data.Subset(test, range(max_te))

    class_names = train.target_words
    return full_train, test, class_names
