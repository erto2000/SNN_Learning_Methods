# datasets/collate.py
from typing import Optional, List, Tuple
import torch
import torch.nn.functional as F
from .transforms import split_into_segments, pad_time_to

def make_segment_collate(sample_length: Optional[int], stride: Optional[int]):
    """
    Collate only. Applies post-processing (segmenting + padding) to samples.
    Returns X:[B,S,T,D], y:[B].
    """
    L, S = sample_length, stride

    def _collate(batch: List[Tuple[torch.Tensor, int]]):
        all_segments: List[List[torch.Tensor]] = []
        ys: List[int] = []
        for x, y in batch:
            if x.dim() == 1:
                x = x.unsqueeze(-1)
            segs = split_into_segments(x, L, S)
            all_segments.append(segs)
            ys.append(int(y))

        # infer S_max and T_pad
        S_max = max(len(segs) for segs in all_segments) if all_segments else 1
        if L is not None and L > 0:
            T_pad = L
        else:
            T_pad = 1
            for segs in all_segments:
                for s in segs:
                    T_pad = max(T_pad, s.shape[0])

        D = all_segments[0][0].shape[1] if all_segments and all_segments[0] else 1
        B = len(batch)
        X = torch.zeros((B, S_max, T_pad, D), dtype=torch.float32)

        for b, segs in enumerate(all_segments):
            for s_idx, s in enumerate(segs):
                X[b, s_idx] = pad_time_to(s, T_pad).to(torch.float32)

        y = torch.tensor(ys, dtype=torch.long)
        return X, y

    return _collate

def flatten_segments(X: torch.Tensor, y: torch.Tensor):
    """
    X:[B,S,T,D] -> (X_segs:[Nseg,T,D], y_segs:[Nseg], sample_ids:[Nseg], seg_mask:[B,S])
    """
    B, S, T, D = X.shape
    seg_mask = (X.abs().sum(dim=(2, 3)) > 0)
    b_idx, s_idx = torch.where(seg_mask)
    if b_idx.numel() == 0:
        return X.new_zeros((0, T, D)), y.new_zeros((0,), dtype=torch.long), b_idx, seg_mask
    X_segs = X[b_idx, s_idx]
    y_segs = y[b_idx]
    sample_ids = b_idx
    return X_segs, y_segs, sample_ids, seg_mask

def majority_vote(preds_seg: torch.Tensor, sample_ids: torch.Tensor, num_classes: int, B: int):
    """Per-sample hard vote from per-segment preds."""
    if preds_seg.numel() == 0:
        return torch.zeros(B, dtype=torch.long, device=preds_seg.device)
    counts = torch.zeros(B, num_classes, device=preds_seg.device)
    one_hot = F.one_hot(preds_seg, num_classes=num_classes).float()
    counts.index_add_(0, sample_ids, one_hot)
    return counts.argmax(dim=1)
