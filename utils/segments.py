import torch
import torch.nn.functional as F


def _batch_pieces(X, y, extra, device, chunk_segments, dtype, include_weights=False):
    target_dtype = dtype or X.dtype
    B = X.shape[0]
    if X.dim() == 3:
        time_mask = extra.get("time_mask")
        horizon = int(time_mask.sum(dim=1).max()) if time_mask is not None else X.shape[1]
        mask = time_mask[:, :horizon].to(device) if time_mask is not None else None
        if mask is not None and bool(mask.all()):
            mask = None
        piece = (X[:, :horizon].to(device=device, dtype=target_dtype), y.to(device),
                 torch.arange(B, device=device, dtype=torch.long), B, mask)
        yield piece + (torch.ones(B, device=device),) if include_weights else piece
        return
    if X.dim() != 4:
        raise ValueError(f"Unexpected X.dim()={X.dim()}")
    seg_mask = extra.get("seg_mask")
    if seg_mask is None:
        seg_mask = torch.ones(X.shape[:2], dtype=torch.bool, device=X.device)
    b_idx, s_idx = torch.where(seg_mask)
    time_mask = extra.get("time_mask")
    lengths = (time_mask.sum(dim=-1) if time_mask is not None else
               torch.full(X.shape[:2], X.shape[2], device=X.device))
    lengths = lengths * seg_mask
    totals = lengths.sum(dim=1)
    if bool((totals <= 0).any()):
        raise ValueError("Every recording must contain at least one valid timestep")
    # A recording contributes total weight one, even when windows overlap.
    # Do not normalize again inside a chunk: that would restore tiny tails to
    # the influence of a full window when they happen to fill the last chunk.
    weights = lengths.float() / totals[:, None]
    # Independent windows retain sample IDs for voting. Mixed lengths stay in
    # the same minibatch instead of creating extra updates for short tails.
    chunk = B if chunk_segments else max(1, len(b_idx))
    for offset in range(0, len(b_idx), chunk):
        bi, si = b_idx[offset:offset+chunk], s_idx[offset:offset+chunk]
        mask = time_mask[bi, si] if time_mask is not None else None
        horizon = int(mask.sum(dim=1).max()) if mask is not None else X.shape[2]
        mask = mask[:, :horizon].to(device) if mask is not None else None
        if mask is not None and bool(mask.all()):
            mask = None
        piece = (X[bi, si, :horizon].to(device=device, dtype=target_dtype),
                 y[bi].to(device), bi.to(device), B, mask)
        yield piece + (weights[bi, si].to(device),) if include_weights else piece


@torch.no_grad()
def iter_piece_batches(loader, device, *, chunk_segments=False, dtype=None, source_length_profile=None,
                       include_weights=False):
    """Yield one iterator of masked piece batches per original sample batch.

    Keep this boundary during evaluation so all windows, including partial
    tails, contribute to one sample-level vote rather than separate votes.
    """
    for X, y, extra in loader:
        if source_length_profile is not None:
            infos = extra.get("infos", [{} for _ in range(len(y))])
            for index, info in enumerate(infos):
                length = info.get("pre_window_time_steps")
                if length is None:
                    mask = extra.get("time_mask")
                    length = int(mask[index].sum()) if X.dim() == 3 and mask is not None else X.shape[-2]
                source_length_profile[int(length)] += 1
        yield _batch_pieces(X, y, extra, device, chunk_segments, dtype, include_weights)


@torch.no_grad()
def iter_pieces(loader, device, *, chunk_segments=False, dtype=None, source_length_profile=None,
                include_weights=False):
    """Yield Xp:[N,T,D], labels, sample IDs, original B and optional time mask.

    Training chunks contain at most B windows, with their real lengths retained
    in the mask. Full sequences keep the original minibatch intact. When
    include_weights is true, a sixth item gives recording-normalized length
    weights for learning.
    """
    for pieces in iter_piece_batches(loader, device, chunk_segments=chunk_segments, dtype=dtype,
                                     source_length_profile=source_length_profile,
                                     include_weights=include_weights):
        yield from pieces


@torch.no_grad()
def majority_vote(preds_seg: torch.Tensor, sample_ids: torch.Tensor, num_classes: int, B: int,
                  weights: torch.Tensor = None) -> torch.Tensor:
    if preds_seg.numel() == 0:
        return torch.zeros(B, dtype=torch.long, device=preds_seg.device)
    counts = torch.zeros(B, num_classes, device=preds_seg.device)
    one_hot = F.one_hot(preds_seg, num_classes=num_classes).float()
    if weights is not None:
        weights = weights.to(device=one_hot.device, dtype=one_hot.dtype)
        if weights.shape != preds_seg.shape or not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
            raise ValueError("Vote weights must be finite, nonnegative and match predictions")
        one_hot = one_hot * weights[:, None]
    counts.index_add_(0, sample_ids, one_hot)
    return counts.argmax(dim=1)
