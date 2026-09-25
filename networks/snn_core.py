# networks/snn_core.py
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.init as init
import snntorch as snn
from snntorch import surrogate
from .specs import NetConfig, LayerSpec

def _resolve_surrogate(name: str, slope: float):

    table = {
        # Common surrogates
        "fast_sigmoid": surrogate.fast_sigmoid(slope),
        "sigmoid": surrogate.sigmoid(),
        "atan": surrogate.atan(),
        "triangular": surrogate.triangular(),

        # STE variants
        "ste": surrogate.straight_through_estimator(),
        "straight_through_estimator": surrogate.straight_through_estimator(),

        # Spike rate / escape noise
        "spike_rate_escape": surrogate.spike_rate_escape(),
        "SpikeRateEscape": surrogate.SpikeRateEscape(),

        # Sparse options
        "sparse_fast_sigmoid": surrogate.SparseFastSigmoid(slope),
        "SparseFastSigmoid": surrogate.SparseFastSigmoid(slope),

        # Operators
        "lso": surrogate.LSO(),
        "LSO": surrogate.LSO(),
        "sso": surrogate.SSO(),
        "SSO": surrogate.SSO(),
        "sfs": surrogate.SFS(),
        "SFS": surrogate.SFS(),
        "leaky_spike_operator": surrogate.LeakySpikeOperator(),
        "LeakySpikeOperator": surrogate.LeakySpikeOperator(),

        # Stochastic
        "stochastic": surrogate.StochasticSpikeOperator(),
        "StochasticSpikeOperator": surrogate.StochasticSpikeOperator(),
    }

    return table.get(name.lower(), surrogate.fast_sigmoid(slope))

class SNNState:
    def __init__(self, mems: List[torch.Tensor], spikes: List[Optional[torch.Tensor]]):
        self.mems = mems
        self.spikes = spikes  # per-layer last spikes if recurrence is active, else None

class SNNCore(nn.Module):
    """
    Generic SNN backbone:
      [ (Linear -> Norm? -> LIF) x L ] -> optional head ("logits" Linear | "lif")
    No aggregation; step-by-step only. Learners decide time handling.
    """
    def __init__(self, cfg: NetConfig, n_classes: int):
        super().__init__()
        self.cfg = cfg
        self.n_classes = n_classes

        sg = _resolve_surrogate(cfg.spike_grad, cfg.slope)
        self.fcs   = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.lifs  = nn.ModuleList()
        self.Wrecs = nn.ParameterList()
        self.Wrec_flags: List[bool] = []

        for ls in cfg.layers:
            self.fcs.append(nn.Linear(ls.dim_in, ls.dim_out, bias=ls.bias))
            if ls.norm == "layernorm":
                self.norms.append(nn.LayerNorm(ls.dim_out))
            elif ls.norm == "batchnorm":
                self.norms.append(nn.BatchNorm1d(ls.dim_out, affine=True))
            elif ls.norm == "rmsnorm":
                self.norms.append(nn.RMSNorm(ls.dim_out))
            else:
                self.norms.append(nn.Identity())
            self.lifs.append(snn.Leaky(beta=cfg.beta, spike_grad=sg, threshold=cfg.threshold))

            # BP learns active recurrence through autograd; manual learners
            # can disable autograd and supply their own recurrent gradients.
            self.Wrec_flags.append(ls.recurrent)
            Wrec = nn.Parameter(torch.zeros(ls.dim_out, ls.dim_out), requires_grad=ls.recurrent)
            self.Wrecs.append(Wrec)

        last_dim = cfg.layers[-1].dim_out
        if cfg.head == "logits":
            self.head = nn.Linear(last_dim, n_classes, bias=cfg.head_bias)
            self.head_lif = None
        elif cfg.head == "lif":
            self.head = nn.Linear(last_dim, n_classes, bias=False)
            self.head_lif = snn.Leaky(beta=cfg.beta, spike_grad=sg, threshold=cfg.threshold)
        else:
            self.head = None
            self.head_lif = None

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if cfg.init == 'default':
                    continue
                if cfg.init == "kaiming_uniform":
                    init.kaiming_uniform_(m.weight, a=0.0, mode="fan_in", nonlinearity="linear")
                elif cfg.init == "kaiming_normal":
                    init.kaiming_normal_(m.weight, a=0.0, mode="fan_in", nonlinearity="linear")
                elif cfg.init == "xavier_uniform":
                    init.xavier_uniform_(m.weight)
                elif cfg.init == "xavier_normal":
                    init.xavier_normal_(m.weight)
                elif cfg.init == "normal":
                    init.normal_(m.weight, mean=0.0, std=0.02)
                elif cfg.init == "uniform":
                    init.uniform_(m.weight, a=-0.1, b=0.1)
                elif cfg.init == "constant":
                    init.constant_(m.weight, 0.01)
                elif cfg.init == "orthogonal":
                    init.orthogonal_(m.weight, gain=1.0)
                elif cfg.init == "sparse":
                    init.sparse_(m.weight, sparsity=0.1, std=0.01)
                else:
                    raise ValueError(f"Unknown init: {cfg.init}")
                if m.bias is not None:
                    init.zeros_(m.bias)

    def init_state(self, N: int, device, dtype) -> Tuple[SNNState, Optional[torch.Tensor]]:
        mems = [torch.zeros(N, fc.out_features, device=device, dtype=dtype) for fc in self.fcs]
        spks = [
            torch.zeros(N, fc.out_features, device=device, dtype=dtype) if self.Wrec_flags[i] else None
            for i, fc in enumerate(self.fcs)
        ]
        head_mem = torch.zeros(N, self.n_classes, device=device, dtype=dtype) if self.head_lif else None
        return SNNState(mems, spks), head_mem

    def forward_step(
        self,
        x_t: torch.Tensor,
        state: SNNState,
        head_mem: Optional[torch.Tensor] = None,
        need_pre: bool = False,
    ):
        """
        x_t: [N, Din0]
        need_pre=True -> returns per-layer pre-activations (post-norm, pre-LIF)
        returns:
          last_hidden_spikes [N,H_L], head_out (None|[N,K]), new_state, new_head_mem,
          layer_spikes list([N,H_l]), optionally layer_pres list([N,H_l])
        """

        h = x_t
        new_mems, new_spks, layer_spikes, pres = [], [], [], []
        for i, (fc, norm, lif) in enumerate(zip(self.fcs, self.norms, self.lifs)):
            pre = fc(h)
            if self.Wrec_flags[i]:
                prev_spk = state.spikes[i]
                if prev_spk is not None:
                    pre = pre + prev_spk @ self.Wrecs[i]
            pre = norm(pre)
            if need_pre:
                pres.append(pre)

            spk, mem = lif(pre, state.mems[i])

            # make sure LIF outputs match pre's dtype (important for fp16/bf16 eval)
            if spk.dtype != pre.dtype:
                spk = spk.to(pre.dtype)
            if mem.dtype != pre.dtype:
                mem = mem.to(pre.dtype)

            new_mems.append(mem)
            # preserve last spikes if recurrence is being used for this layer
            new_spks.append(spk if self.Wrec_flags[i] else None)
            layer_spikes.append(spk)
            h = spk

        if self.head is None:
            out = None
            return h, out, SNNState(new_mems, new_spks), None, layer_spikes, (pres if need_pre else None)

        if self.head_lif is None:
            out = self.head(h)  # logits
            return h, out, SNNState(new_mems, new_spks), None, layer_spikes, (pres if need_pre else None)
        else:
            pre_o = self.head(h)
            spk_o, head_mem = self.head_lif(pre_o, head_mem)

            # keep head LIF outputs in same dtype as pre_o
            if spk_o.dtype != pre_o.dtype:
                spk_o = spk_o.to(pre_o.dtype)
            if head_mem.dtype != pre_o.dtype:
                head_mem = head_mem.to(pre_o.dtype)

            return h, spk_o, SNNState(new_mems, new_spks), head_mem, layer_spikes, (pres if need_pre else None)
