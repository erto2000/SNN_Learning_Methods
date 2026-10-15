# learners/base.py
from abc import ABC, abstractmethod
import torch
from networks.specs import NetConfig


class BaseLearner(ABC):
    """
    Minimal, window-agnostic learner base.
    Contract:
      - forward(X, return_activity=False, time_mask=None) -> logits OR (logits, activity)
      - train_step(X,y,time_mask=None,sample_weight=None) -> dict with loss/acc
      - predict_batch(X,time_mask=None) -> [B] class indices
    """
    def __init__(self, net_cfg: NetConfig, meta: dict, device: torch.device):
        self.cfg = net_cfg
        self.meta = meta
        self.device = device
        self.model = self._build_model().to(device)
        self._batch_gradients = None

    def begin_recording_batch(self):
        """Defer optimizer updates while sequential windows form gradients."""
        self._batch_gradients = {}

    @torch.no_grad()
    def _optimizer_step(self, optimizer):
        if self._batch_gradients is None:
            optimizer.step()
            return
        gradients = self._batch_gradients.setdefault(optimizer, {})
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    if param in gradients:
                        gradients[param].add_(param.grad)
                    else:
                        gradients[param] = param.grad.detach().clone()

    @torch.no_grad()
    def end_recording_batch(self):
        gradients_by_optimizer = self._batch_gradients
        self._batch_gradients = None
        if gradients_by_optimizer is None:
            return
        for optimizer, gradients in gradients_by_optimizer.items():
            optimizer.zero_grad(set_to_none=True)
            snapshots = {p: p.detach().clone() for p in gradients} if getattr(self, 'max_rel_step', None) is not None else {}
            for param, gradient in gradients.items():
                param.grad = gradient
            optimizer.step()
            for param, old in snapshots.items():
                delta = param - old
                limit = self.max_rel_step * old.norm()
                if old.norm() > 0 and delta.norm() > limit:
                    param.copy_(old + delta * limit / (delta.norm() + 1e-8))
        if hasattr(self, '_clamp_rec'):
            self._clamp_rec()

    @abstractmethod
    def _build_model(self) -> torch.nn.Module:
        ...

    @abstractmethod
    def forward(self, X: torch.Tensor, return_activity: bool = False, time_mask=None):
        """X:[B,T,D] -> logits:[B,K] OR (logits:[B,K], activity:dict)."""
        ...

    @abstractmethod
    def train_step(self, X: torch.Tensor, y: torch.Tensor, time_mask=None, sample_weight=None) -> dict:
        """Single-batch update. Returns logs like {'loss': float, 'acc': float[%]}."""
        ...

    @staticmethod
    def _prepare_sample_weight(X: torch.Tensor, sample_weight):
        """Keep each window's recording-level share when averaging a batch.

        The caller supplies valid length divided by the recording's total
        window lengths. Do not renormalize these shares within a minibatch:
        that would give a batch containing only short tails full influence.
        """
        if sample_weight is None:
            return None
        sample_weight = torch.as_tensor(sample_weight, device=X.device, dtype=X.dtype)
        if sample_weight.shape != (X.shape[0],):
            raise ValueError("sample_weight must have shape [B]")
        if not bool(torch.isfinite(sample_weight).all()) or bool((sample_weight < 0).any()):
            raise ValueError("sample_weight must contain finite nonnegative values")
        return None if bool((sample_weight == 1).all()) else sample_weight

    @staticmethod
    def _prepare_time_mask(X: torch.Tensor, time_mask):
        """Validate recorded lengths; real zero-valued inputs remain valid."""
        if X.ndim != 3 or X.shape[1] < 1:
            raise ValueError("Expected a nonempty sequence batch [B,T,D]")
        if time_mask is None:
            return None
        if time_mask.dtype != torch.bool or time_mask.shape != X.shape[:2]:
            raise ValueError("time_mask must be boolean with shape [B,T]")
        time_mask = time_mask.to(device=X.device)
        if not bool(time_mask[:, 0].all()):
            raise ValueError("Every sequence must contain at least one valid time step")
        if bool((time_mask[:, 1:] & ~time_mask[:, :-1]).any()):
            raise ValueError("time_mask must contain a valid prefix followed by padding")
        return None if bool(time_mask.all()) else time_mask

    def _cost_dimensions(self) -> dict:
        fcs = getattr(self.model, "fcs", None)
        if fcs is None or len(fcs) == 0:
            return {
                "C": int(self.meta.get("n_classes", 0)),
                "d0": int(self.meta.get("input_dim", 0)),
                "tilde_d0": int(self.meta.get("input_dim", 0)),
                "d": [], "tilde_inputs": [], "a": [],
                "A": 0, "U": 0, "V": 0, "F": 0, "P": 0, "Y": 0,
                "f_l": [], "p_l": [], "L": 0,
            }

        d = [int(fc.out_features) for fc in fcs]
        tilde_inputs = [int(fcs[0].in_features)] + d[:-1]
        a = [din * dout for din, dout in zip(tilde_inputs, d)]
        C = int(self.meta.get("n_classes", getattr(self.model, "n_classes", 0)))
        d0 = int(self.meta.get("input_dim", tilde_inputs[0]))
        V = C * d[-1] if getattr(self.model, "head", None) is not None else 0
        f_l = [a_l + din + 2 * dout for a_l, din, dout in zip(a, tilde_inputs, d)]
        p_l = [a_l + din + dout for a_l, din, dout in zip(a, tilde_inputs, d)]
        Y = 0 if V == 0 else V + d[-1] + C

        return {
            "C": C,
            "d0": d0,
            "tilde_d0": tilde_inputs[0],
            "d": d,
            "tilde_inputs": tilde_inputs,
            "a": a,
            "A": int(sum(a)),
            "U": int(sum(d)),
            "V": int(V),
            "F": int(sum(f_l)),
            "P": int(sum(p_l)),
            "Y": int(Y),
            "f_l": [int(x) for x in f_l],
            "p_l": [int(x) for x in p_l],
            "L": len(d),
        }

    def _cost_components(self, batch: int, time_steps: int, dims: dict) -> dict:
        raise NotImplementedError(f"{type(self).__name__} does not define estimated costs.")

    def estimate_costs(
        self,
        batch: int,
        time_steps: int,
        fp_bytes: int = 4,
        alpha: float = 1.0,
        beta: float = 1.0,
        num_windows: float = 1.0,
    ) -> dict:
        if batch < 1 or time_steps < 1 or num_windows <= 0:
            raise ValueError("Batch size, time steps and number of windows must be positive")
        if alpha < 0 or beta < 0:
            raise ValueError("Estimated time coefficients must be nonnegative")
        dims = self._cost_dimensions()
        comps = self._cost_components(batch, time_steps, dims)

        memory_components = {k: int(v) for k, v in comps.get("memory", {}).items()}
        compute_components = {k: int(v) for k, v in comps.get("compute", {}).items()}
        access_components = {k: int(v) for k, v in comps.get("access", {}).items()}
        # Repeated windows multiply total work; their working memory is reused.
        compute_components = {k: v * num_windows for k, v in compute_components.items()}
        access_components = {k: v * num_windows for k, v in access_components.items()}

        memory_scalars = int(sum(memory_components.values()))
        compute_scalars = sum(compute_components.values())
        access_scalars = sum(access_components.values())
        time_proxy = float(alpha) * compute_scalars + float(beta) * access_scalars

        return {
            "method": type(self).__name__,
            "model_version": "algorithmic-window-v3",
            "work_scope": "per_batch_of_original_sequences",
            "memory_scope": "peak_sequential_window_working_set",
            "windows_per_sample": num_windows,
            "batch_size": int(batch),
            "time_steps": int(time_steps),
            "fp_bytes": int(fp_bytes),
            "definitions": dims,
            "memory": {
                "components": memory_components,
                "total_scalars": memory_scalars,
                "total_bytes": memory_scalars * int(fp_bytes),
            },
            "compute": {
                "components": compute_components,
                "total_scalars": compute_scalars,
            },
            "access": {
                "components": access_components,
                "total_scalars": access_scalars,
            },
            "time_proxy": {
                "alpha": float(alpha),
                "beta": float(beta),
                "value": time_proxy,
            },
        }

    def get_param_memory_bytes(self, fp_bytes: int = 4) -> int:
        total = 0
        fcs = getattr(self.model, "fcs", None)
        if fcs is not None:
            for fc in fcs:
                total += int(fc.weight.numel())
                if getattr(fc, "bias", None) is not None:
                    total += int(fc.bias.numel())

        Wrecs = getattr(self.model, "Wrecs", None)
        r_flags = list(getattr(self.model, "Wrec_flags", []))
        if Wrecs is not None:
            for i, W in enumerate(Wrecs):
                if i < len(r_flags) and r_flags[i]:
                    total += int(W.numel())

        head = getattr(self.model, "head", None)
        if head is not None:
            total += int(head.weight.numel())
            if getattr(head, "bias", None) is not None:
                total += int(head.bias.numel())

        return total * fp_bytes

    def get_input_memory_bytes(self, batch: int, time_steps: int, fp_bytes: int = 4) -> int:
        fcs = getattr(self.model, "fcs", None)
        if fcs is None or len(fcs) == 0:
            return 0

        d0 = fcs[0].in_features
        return batch * time_steps * d0 * fp_bytes

    def get_inference_memory_bytes(self, batch: int, time_steps: int, fp_bytes: int = 4) -> int:
        fcs = getattr(self.model, "fcs", None)
        if fcs is None or len(fcs) == 0:
            return self.get_input_memory_bytes(batch, time_steps, fp_bytes=fp_bytes) + \
                self.get_param_memory_bytes(fp_bytes=fp_bytes)

        Hs = [fc.out_features for fc in fcs]

        N_input = self.get_input_memory_bytes(batch, time_steps, fp_bytes=fp_bytes)
        N_param = self.get_param_memory_bytes(fp_bytes=fp_bytes)
        N_state = batch * sum(Hs) * fp_bytes

        return N_input + N_param + N_state

    def get_training_memory_bytes(self, batch: int, time_steps: int, fp_bytes: int = 4) -> int:
        return int(self.estimate_costs(batch, time_steps, fp_bytes=fp_bytes)["memory"]["total_bytes"])

    def _layer_fanouts(self) -> list[int]:
        """
        Per hidden layer:
          - next hidden layer size
          - plus recurrent size if recurrent
          - plus output head size for last hidden layer
          - for FF-like headless models, count final goodness readout as fanout=1
        """
        fcs = getattr(self.model, "fcs", None)
        if fcs is None or len(fcs) == 0:
            return []

        r_flags = list(getattr(self.model, "Wrec_flags", [False] * len(fcs)))
        fanouts = []

        for l, fc in enumerate(fcs):
            fanout = 0

            if l < len(fcs) - 1:
                fanout += fcs[l + 1].out_features

            if r_flags[l]:
                fanout += fc.out_features

            if l == len(fcs) - 1 and getattr(self.model, "head", None) is not None:
                fanout += int(self.meta["n_classes"])

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
        valid_steps: int | None = None,
        extra: dict | None = None,
    ) -> dict:
        fcs = getattr(self.model, "fcs", [])
        layer_sizes = [fc.out_features for fc in fcs]
        fanouts = self._layer_fanouts()

        total_spike_count = float(sum(layer_spike_counts))
        synops = float(sum(sc * fo for sc, fo in zip(layer_spike_counts, fanouts)))

        num_neurons = int(sum(layer_sizes))
        sample_steps = int(valid_steps) if valid_steps is not None else num_samples * num_timesteps
        num_neuron_slots = float(max(1, sample_steps * max(1, num_neurons)))
        firing_rate = total_spike_count / num_neuron_slots

        input_dim = int(fcs[0].in_features) if len(fcs) > 0 else 0
        first_hidden = int(layer_sizes[0]) if layer_sizes else 0

        input_mac_ops = float(sample_steps * input_dim * first_hidden)
        neuron_updates = float(sample_steps * num_neurons)

        out = {
            "total_spike_count": total_spike_count,
            "firing_rate": firing_rate,
            "synaptic_operations": synops,
            "num_neurons": num_neurons,
            "num_samples": int(num_samples),
            "num_timesteps": int(num_timesteps),
            "valid_timesteps": sample_steps,
            "num_neuron_slots": num_neuron_slots,
            "layer_spike_counts": [float(x) for x in layer_spike_counts],
            "layer_fanouts": [int(x) for x in fanouts],
            "input_mac_ops": input_mac_ops,
            "neuron_updates": neuron_updates,
            "input_dim": input_dim,
        }

        if extra:
            out.update(extra)
        return out

    @torch.no_grad()
    def predict_batch(self, X: torch.Tensor, time_mask=None) -> torch.Tensor:
        self.model.eval()
        X = X.to(self.device)
        out = self.forward(X, return_activity=False, time_mask=time_mask)
        logits = out[0] if isinstance(out, tuple) else out
        return logits.argmax(dim=1)
