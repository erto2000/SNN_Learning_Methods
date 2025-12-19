# utils/time_gating.py
import torch
from typing import Literal

def make_time_weights(
    T: int,
    *,
    device=None,
    dtype=torch.float32,
    start_u: float = 0.5,
    mode: Literal["hard", "linear", "sigmoid", "cosine"] = "hard",
    ramp_u: float = 0.0,
    sharpness: float = 20.0,
) -> torch.Tensor:
    """
    Returns w:[T] in [0,1] for normalized time u=t/(T-1).
      - hard:   0 for u < start_u, 1 otherwise
      - linear: ramp 0->1 across [start_u, start_u+ramp_u]
      - cosine: smooth ramp across [start_u, start_u+ramp_u]
      - sigmoid: smooth step centered in the ramp interval

    If ramp_u==0, behaves like hard threshold even if mode != hard.
    """
    if T <= 1:
        return torch.ones(T, device=device, dtype=dtype)

    u = torch.linspace(0.0, 1.0, T, device=device, dtype=dtype)

    start_u = float(max(0.0, min(1.0, start_u)))
    ramp_u  = float(max(0.0, min(1.0, ramp_u)))

    if mode == "hard" or ramp_u == 0.0:
        return (u >= start_u).to(dtype)

    end_u = min(1.0, start_u + ramp_u)
    x = (u - start_u) / max(1e-8, (end_u - start_u))  # ramp in "ramp coords"

    if mode == "linear":
        return x.clamp(0.0, 1.0)

    if mode == "cosine":
        x01 = x.clamp(0.0, 1.0)
        return 0.5 - 0.5 * torch.cos(torch.pi * x01)

    if mode == "sigmoid":
        mid = 0.5 * (start_u + end_u)
        return torch.sigmoid(sharpness * (u - mid)).to(dtype)

    raise ValueError(f"Unknown time gating mode: {mode}")
