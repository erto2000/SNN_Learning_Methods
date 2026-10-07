import torch
import torch.nn.functional as F


def _batch_pieces(X, y, extra, device, chunk_segments, dtype):
    target_dtype = dtype or X.dtype
    B = X.shape[0]
    if X.dim() == 3:
        # Preserve the existing full-sequence batching convention.
        yield (X.to(device=device, dtype=target_dtype), y.to(device),
               torch.arange(B, device=device, dtype=torch.long), B)
        return
    if X.dim() != 4:
        raise ValueError(f"Unexpected X.dim()={X.dim()}")
    seg_mask = extra.get("seg_mask")
    if seg_mask is None:
        seg_mask = torch.ones(X.shape[:2], dtype=torch.bool)
    b_idx, s_idx = torch.where(seg_mask)
    time_mask = extra.get("time_mask")
    lengths = (time_mask[b_idx, s_idx].sum(dim=1) if time_mask is not None
               else torch.full_like(b_idx, X.shape[2]))
    # Equal-length groups ensure neither forward dynamics nor gradients see
    # padded time steps. Each window keeps its original sample ID for voting.
    for length in sorted(lengths.unique().tolist(), reverse=True):
        if length <= 0:
            raise ValueError("A valid window must contain at least one time step")
        positions = torch.where(lengths == length)[0]
        chunk = B if chunk_segments else max(1, len(positions))
        for offset in range(0, len(positions), chunk):
            take = positions[offset:offset+chunk]
            bi, si = b_idx[take], s_idx[take]
            yield (X[bi, si, :length].to(device=device, dtype=target_dtype),
                   y[bi].to(device), bi.to(device), B)


@torch.no_grad()
def iter_piece_batches(loader, device, *, chunk_segments=False, dtype=None, source_length_profile=None):
    """Yield one iterator of actual-length pieces per original sample batch.

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
        yield _batch_pieces(X, y, extra, device, chunk_segments, dtype)


@torch.no_grad()
def iter_pieces(loader, device, *, chunk_segments=False, dtype=None, source_length_profile=None):
    """Yield Xp:[N,T,D], labels, original sample IDs and original batch size.

    Windowed pieces are grouped by actual temporal length. Training chunks
    contain at most B windows. Storage padding never reaches a learner.
    """
    for pieces in iter_piece_batches(loader, device, chunk_segments=chunk_segments, dtype=dtype,
                                     source_length_profile=source_length_profile):
        yield from pieces


@torch.no_grad()
def majority_vote(preds_seg: torch.Tensor, sample_ids: torch.Tensor, num_classes: int, B: int) -> torch.Tensor:
    if preds_seg.numel() == 0:
        return torch.zeros(B, dtype=torch.long, device=preds_seg.device)
    counts = torch.zeros(B, num_classes, device=preds_seg.device)
    one_hot = F.one_hot(preds_seg, num_classes=num_classes).float()
    counts.index_add_(0, sample_ids, one_hot)
    return counts.argmax(dim=1)
