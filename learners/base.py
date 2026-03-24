# learners/base.py
from abc import ABC, abstractmethod
import torch
from networks.specs import NetConfig


class BaseLearner(ABC):
    """
    Minimal, window-agnostic learner base.
    Contract:
      - forward(X, return_activity=False) -> logits OR (logits, activity)
      - train_step(X,y) -> dict with loss/acc
      - predict_batch(X) -> [B] class indices
    """
    def __init__(self, net_cfg: NetConfig, meta: dict, device: torch.device):
        self.cfg = net_cfg
        self.meta = meta
        self.device = device
        self.model = self._build_model().to(device)

    @abstractmethod
    def _build_model(self) -> torch.nn.Module:
        ...

    @abstractmethod
    def forward(self, X: torch.Tensor, return_activity: bool = False):
        """X:[B,T,D] -> logits:[B,K] OR (logits:[B,K], activity:dict)."""
        ...

    @abstractmethod
    def train_step(self, X: torch.Tensor, y: torch.Tensor) -> dict:
        """Single-batch update. Returns logs like {'loss': float, 'acc': float[%]}."""
        ...

    # --------- memory estimates ---------
    def get_static_memory_bytes(self, fp_bytes: int = 4) -> int:
        """
        Theoretical static memory for network
        """
        # ----- read dims from the actual network -----
        fcs = getattr(self.model, "fcs", None)
        if fcs is None or len(fcs) == 0:
            return 0

        # layer widths
        d0 = fcs[0].in_features
        d = [fc.out_features for fc in fcs]  # [d1..dL]
        L = len(d)

        # recurrence flags
        r_flags = getattr(self.model, "Wrec_flags", [False] * L)
        r_flags = list(r_flags)

        # number of classes C (prefer meta, fall back to model.n_classes)
        C = int(self.meta.get("n_classes", getattr(self.model, "n_classes", 0)))

        # ----- theoretical param counts -----
        # feedforward weights: sum d_l d_{l-1}
        Nf = d[0] * d0 + sum(d[l] * d[l - 1] for l in range(1, L))

        # recurrent weights: sum r_l d_l^2
        Nr = sum((d[l] * d[l]) for l in range(L) if r_flags[l])

        # output head weights: C d_L if head exists
        Nout = C * d[-1] if getattr(self.model, "head", None) is not None else 0

        Nstatic = Nf + Nr + Nout
        return Nstatic * fp_bytes

    def get_training_memory_bytes(self, batch: int, time_steps: int, fp_bytes: int = 4) -> int:
        """
        Theoretical training-time memory (activations, algorithm buffers, etc.)
        Default: 0 (override in subclasses).
        """
        return 0

    def _layer_fanouts(self) -> list[int]:
        """
        Per hidden layer:
          - next hidden layer size
          - plus recurrent size if recurrent
          - plus output head size for last hidden layer
          - for FF-like headless models, count the final goodness readout as fanout=1
        """
        fcs = getattr(self.model, "fcs", None)
        if fcs is None or len(fcs) == 0:
            return []

        r_flags = list(getattr(self.model, "Wrec_flags", [False] * len(fcs)))
        fanouts = []

        for l, fc in enumerate(fcs):
            fanout = 0

            # next hidden layer
            if l < len(fcs) - 1:
                fanout += fcs[l + 1].out_features

            # recurrent fanout
            if r_flags[l]:
                fanout += fc.out_features

            # normal classifier head
            if l == len(fcs) - 1 and getattr(self.model, "head", None) is not None:
                fanout += int(self.meta["n_classes"])

            # headless final layer (e.g. FF): goodness/readout usage
            if l == len(fcs) - 1 and getattr(self.model, "head", None) is None and fanout == 0:
                fanout += 1

            fanouts.append(int(fanout))

        return fanouts

    def _make_activity_dict(
        self,
        *,
        layer_spike_counts: list[float],
        num_samples: int,
        num_timesteps: int,
        extra: dict | None = None,
    ) -> dict:
        fcs = getattr(self.model, "fcs", [])
        layer_sizes = [fc.out_features for fc in fcs]
        fanouts = self._layer_fanouts()

        total_spike_count = float(sum(layer_spike_counts))
        synops = float(sum(sc * fo for sc, fo in zip(layer_spike_counts, fanouts)))
        num_neurons = int(sum(layer_sizes))
        num_neuron_slots = float(max(1, num_samples * num_timesteps * max(1, num_neurons)))
        firing_rate = total_spike_count / num_neuron_slots

        out = {
            "total_spike_count": total_spike_count,
            "firing_rate": firing_rate,
            "synaptic_operations": synops,
            "num_neurons": num_neurons,
            "num_samples": int(num_samples),
            "num_timesteps": int(num_timesteps),
            "num_neuron_slots": num_neuron_slots,
            "layer_spike_counts": [float(x) for x in layer_spike_counts],
            "layer_fanouts": [int(x) for x in fanouts],
        }
        if extra:
            out.update(extra)
        return out

    @torch.no_grad()
    def predict_batch(self, X: torch.Tensor) -> torch.Tensor:
        """Greedy prediction: X:[B,T,D] -> LongTensor[B]."""
        self.model.eval()
        X = X.to(self.device)
        out = self.forward(X, return_activity=False)
        logits = out[0] if isinstance(out, tuple) else out
        return logits.argmax(dim=1)