# networks/specs.py
from dataclasses import dataclass
from typing import List, Optional, Literal

NormType = Optional[Literal["layernorm", "batchnorm"]]
HeadType = Optional[Literal["logits", "lif"]]  # None => no head (e.g., FF)

@dataclass
class LayerSpec:
    dim_in: int
    dim_out: int
    recurrent: bool = False
    norm:   NormType = None   # "layernorm" | "batchnorm" | None

@dataclass
class NetConfig:
    layers: List[LayerSpec]
    beta: float = 0.9
    spike_grad: str = "fast_sigmoid"
    slope: float = 25.0
    threshold: float = 1.0
    head: Optional[HeadType] = "logits"  # "logits" | "lif" | None
    init: str = "default"