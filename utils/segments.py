from typing import Iterator, Tuple, Optional
import torch
import torch.nn.functional as F

@torch.no_grad()
def iter_pieces(
        loader,
        device,
        *,
        chunk_segments: bool = False,
        dtype: Optional[torch.dtype] = None,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]:
    """
    Yields:
      Xp:[Nseg,T,D], yp:[Nseg], sample_ids:[Nseg], B:int (original batch size)

    Default (chunk_segments=False): old behavior (yields all windows at once).
    Training (chunk_segments=True): windows are yielded in chunks of size B.

    dtype: if not None, cast X to this dtype.
    """
    for X, y, extra in loader:
        target_dtype = dtype or X.dtype

        # ----- No windowing: [B,T,D] -----
        if X.dim() == 3:
            B = X.shape[0]
            Xp = X.to(device=device, dtype=target_dtype)
            yp = y.to(device)
            sample_ids = torch.arange(B, device=device, dtype=torch.long)
            yield Xp, yp, sample_ids, B
            continue

        # ----- Windowed: [B,S,T,D] -----
        if X.dim() == 4:
            seg_mask = extra.get("seg_mask")
            if seg_mask is None:
                seg_mask = torch.ones(X.shape[:2], dtype=torch.bool)

            B, S, T, D = X.shape
            b_idx, s_idx = torch.where(seg_mask)
            if b_idx.numel() == 0:
                continue

            Xf = X[b_idx, s_idx].to(device=device, dtype=target_dtype)  # [Nseg,T,D]
            yf = y[b_idx].to(device)                                    # [Nseg]
            sid = b_idx.to(device)                                      # [Nseg]

            if not chunk_segments:
                yield Xf, yf, sid, B
            else:
                # chunk windows so each step sees <= B windows
                Nseg = Xf.shape[0]
                for i in range(0, Nseg, B):
                    j = min(i + B, Nseg)
                    yield Xf[i:j], yf[i:j], sid[i:j], B
            continue

        raise ValueError(f"Unexpected X.dim()={X.dim()}")


@torch.no_grad()
def majority_vote(preds_seg: torch.Tensor, sample_ids: torch.Tensor, num_classes: int, B: int) -> torch.Tensor:
    if preds_seg.numel() == 0:
        return torch.zeros(B, dtype=torch.long, device=preds_seg.device)
    counts = torch.zeros(B, num_classes, device=preds_seg.device)
    one_hot = F.one_hot(preds_seg, num_classes=num_classes).float()
    counts.index_add_(0, sample_ids, one_hot)
    return counts.argmax(dim=1)
