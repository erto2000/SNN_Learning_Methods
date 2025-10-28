# utils/segments.py
from typing import Iterator, Tuple
import torch
import torch.nn.functional as F

@torch.no_grad()
def iter_pieces(loader, device) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]:
    """
    Yields:
      Xp:[Nseg,T,D], yp:[Nseg], sample_ids:[Nseg], B:int(original batch size)
    Works for:
      - unsegmented: X:[B,T,D] -> Nseg=B, sample_ids=[0..B-1]
      - segmented:   X:[B,S,T,D] + seg_mask -> flatten valid segments
    """
    for X, y, extra in loader:
        if X.dim() == 3:
            # [B,T,D]
            B, T, D = X.shape
            Xp = X.to(device)                      # [B,T,D]
            yp = y.to(device)                      # [B]
            sample_ids = torch.arange(B, device=device, dtype=torch.long)
            yield Xp, yp, sample_ids, B
        elif X.dim() == 4:
            # [B,S,T,D] with optional seg_mask
            seg_mask = extra.get("seg_mask")
            if seg_mask is None:
                seg_mask = torch.ones(X.shape[:2], dtype=torch.bool)
            B, S, T, D = X.shape
            valid = seg_mask                       # [B,S]
            b_idx, s_idx = torch.where(valid)
            if b_idx.numel() == 0:
                continue
            Xp = X[b_idx, s_idx].to(device)        # [Nseg,T,D]
            yp = y[b_idx].to(device)               # [Nseg]
            sample_ids = b_idx.to(device)          # [Nseg]
            yield Xp, yp, sample_ids, B
        else:
            raise ValueError(f"Unexpected X.dim()={X.dim()}")

@torch.no_grad()
def majority_vote(preds_seg: torch.Tensor, sample_ids: torch.Tensor, num_classes: int, B: int) -> torch.Tensor:
    """preds_seg:[Nseg], sample_ids:[Nseg] -> per-sample preds:[B] by hard vote."""
    if preds_seg.numel() == 0:
        return torch.zeros(B, dtype=torch.long, device=preds_seg.device)
    counts = torch.zeros(B, num_classes, device=preds_seg.device)
    one_hot = F.one_hot(preds_seg, num_classes=num_classes).float()
    counts.index_add_(0, sample_ids, one_hot)
    return counts.argmax(dim=1)
