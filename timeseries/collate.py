# timeseries/collate.py
from __future__ import annotations
from typing import List, Tuple, Dict, Any
import torch

def _pad_time_to(x: torch.Tensor, T_pad: int) -> torch.Tensor:
    # x: [T,D]
    T, D = x.shape
    if T == T_pad: return x
    out = x.new_zeros((T_pad, D))
    out[:T] = x
    return out

def collate_pad(batch: List[Tuple[torch.Tensor, int, Dict[str, Any]]]):
    """
    Supports x of shape [T,D] (2D) OR [S,T,D] (3D, e.g., after SlidingWindow).
    Returns:
      - if 2D:    X:[B,T_pad,D],   y:[B], info={"time_mask":[B,T_pad], "infos":[...] }
      - if 3D:    X:[B,S_pad,T_pad,D], y:[B],
                  info={"seg_mask":[B,S_pad], "time_mask":[B,S_pad,T_pad], "infos":[...] }
    """
    xs, ys, infos = zip(*batch)
    y = torch.tensor(ys, dtype=torch.long)

    dim = xs[0].dim()
    if dim == 2:
        # [T,D]
        T_pad = max(x.shape[0] for x in xs)
        D = xs[0].shape[1]
        B = len(xs)
        X = torch.zeros((B, T_pad, D), dtype=torch.float32)
        mask = torch.zeros((B, T_pad), dtype=torch.bool)
        for b, x in enumerate(xs):
            T = x.shape[0]
            X[b, :T] = x
            mask[b, :T] = True
        return (X, y, {"time_mask": mask, "infos": infos})

    elif dim == 3:
        # [S,T,D]
        S_pad = max(x.shape[0] for x in xs)
        T_pad = max(x.shape[1] for x in xs)
        D = xs[0].shape[2]
        B = len(xs)
        X = torch.zeros((B, S_pad, T_pad, D), dtype=torch.float32)
        seg_mask = torch.zeros((B, S_pad), dtype=torch.bool)
        time_mask = torch.zeros((B, S_pad, T_pad), dtype=torch.bool)
        for b, x in enumerate(xs):
            S, T, _ = x.shape
            X[b, :S, :T] = x
            seg_mask[b, :S] = True
            time_mask[b, :S, :T] = True
        return (X, y, {"seg_mask": seg_mask, "time_mask": time_mask, "infos": infos})

    else:
        raise ValueError(f"Unsupported x.dim()={dim}; expected 2 or 3.")
