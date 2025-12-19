# utils/time_gating.py
import torch
from typing import Literal, Optional

def make_time_weights(
    T: int,
    device=None,
    dtype=torch.float32,
    *,
    start_u: float = 0.5,                 # normalized start in [0,1]
    mode: Literal["hard", "linear", "sigmoid", "cosine"] = "hard",
    ramp_u: float = 0.0,                  # normalized ramp width (for linear/sigmoid/cosine)
    sharpness: float = 20.0,              # for sigmoid
) -> torch.Tensor:
    """
    Returns w:[T] in [0,1]
      - hard:   0 for u < start_u, 1 otherwise
      - linear: ramp from 0->1 across [start_u, start_u + ramp_u]
      - sigmoid: smooth step centered near start_u
      - cosine: smooth ramp 0->1 across ramp interval
    """
    if T <= 1:
        return torch.ones(T, device=device, dtype=dtype)

    u = torch.linspace(0.0, 1.0, T, device=device, dtype=dtype)

    start_u = float(max(0.0, min(1.0, start_u)))
    ramp_u = float(max(0.0, min(1.0, ramp_u)))

    if mode == "hard" or ramp_u == 0.0:
        return (u >= start_u).to(dtype)

    end_u = min(1.0, start_u + ramp_u)
    x = (u - start_u) / max(1e-8, (end_u - start_u))  # normalize ramp to [0,1]

    if mode == "linear":
        return x.clamp(0.0, 1.0)

    if mode == "cosine":
        x01 = x.clamp(0.0, 1.0)
        return 0.5 - 0.5 * torch.cos(torch.pi * x01)

    if mode == "sigmoid":
        # center transition roughly inside the ramp interval
        mid = (start_u + end_u) * 0.5
        return torch.sigmoid(sharpness * (u - mid)).to(dtype)

    raise ValueError(f"Unknown mode: {mode}")
