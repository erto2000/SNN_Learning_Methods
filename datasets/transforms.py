# datasets/transforms.py
from typing import List, Optional
import torch

def split_into_segments(x: torch.Tensor, L: Optional[int], stride: Optional[int]) -> List[torch.Tensor]:
    """
    x: [T,D] or [T]  -> returns list of segments [t_i,D] (t_i<=L if L>0).
    If L is None: returns [x].
    """
    if x.dim() == 1:
        x = x.unsqueeze(-1)
    T = x.shape[0]
    if L is None or L <= 0:
        return [x]
    hop = int(stride) if (stride is not None and stride > 0) else L
    segs, start = [], 0
    while start < T:
        segs.append(x[start:start+L])
        start += hop
    return segs

def pad_time_to(x: torch.Tensor, T_pad: int) -> torch.Tensor:
    """Right-pad [T,D] to [T_pad,D]."""
    T, D = x.shape
    if T == T_pad:
        return x
    out = x.new_zeros((T_pad, D))
    out[:T] = x
    return out
