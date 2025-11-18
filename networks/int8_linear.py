# networks/int8_linear.py
from __future__ import annotations
import torch
import torch.nn.functional as F
import torch.nn as nn


class Int8Linear(nn.Module):
    """
    Inference-only wrapper around nn.Linear with int8 weight storage
    and float compute: y = x @ (W_int8*scale)^T + b

    - No gradients.
    - Can do per-tensor or per-channel scaling.
    """
    def __init__(self, linear: nn.Linear, per_channel: bool = True):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.per_channel = bool(per_channel)

        # Copy bias as-is (still float32, cheap)
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
        else:
            self.bias = None

        # Quantize weights
        w = linear.weight.detach().clone()  # [out_features, in_features]
        w_device = w.device

        if per_channel:
            # per-output-channel scale: [out_features, 1]
            max_abs = w.abs().amax(dim=1, keepdim=True)  # [out_features,1]
            scale = (max_abs / 127.0).clamp_min(1e-8)
        else:
            max_abs = w.abs().max()
            scale = (max_abs / 127.0).clamp_min(1e-8)

        # int8 weights in [-128, 127]
        w_int8 = torch.round(w / scale).clamp_(-128, 127).to(torch.int8)

        # Register as buffers (no grad, saved with the model)
        self.register_buffer("weight_int8", w_int8)
        self.register_buffer("scale", scale.to(w_device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize on the fly to match input dtype
        w_f = self.weight_int8.to(dtype=x.dtype) * self.scale.to(dtype=x.dtype)
        return F.linear(x, w_f, self.bias)

def convert_linear_to_int8(module: nn.Module, per_channel: bool = True) -> nn.Module:
    """
    Recursively walk a module and replace nn.Linear with Int8Linear.
    Operates in-place and returns the same root module.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, Int8Linear(child, per_channel=per_channel))
        else:
            convert_linear_to_int8(child, per_channel=per_channel)
    return module
