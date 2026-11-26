"""Evaluate the training cost formulas on the current model and data."""
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path

import torch

from experiment_config import DATASET_CONFIGS, DEFAULT
from timeseries.registry import get_dataloaders
from timeseries.transforms import window_spans
from utils.common import set_seed
from utils.costs import training_costs
from utils.runner import _make_learner
from utils.training import build_cfg, str_to_dtype
from utils.window_search import _has_fixed_horizon


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ConfiguredCosts:
    config: dict
    metadata: dict
    length_counts: Counter
    _learners: dict = field(default_factory=dict, init=False, repr=False)

    @property
    def hidden_sizes(self):
        return tuple(self.config["HIDDEN_SIZES"])

    @property
    def full_sequence_steps(self):
        return max(self.length_counts)

    @property
    def original_samples(self):
        return sum(self.length_counts.values())

    @property
    def fp_bytes(self):
        return str_to_dtype(self.config["DTYPE"]).itemsize

    def dimensions(self, hidden_sizes=None):
        sizes = self.hidden_sizes if hidden_sizes is None else hidden_sizes
        return dict(batch_size=self.config["BATCH_SIZE"],
                    input_features=self.metadata["input_dim"],
                    classes=self.metadata["n_classes"], hidden_sizes=sizes,
                    fp_bytes=self.fp_bytes)

    def learner(self, method, hidden_sizes=None):
        sizes = self.hidden_sizes if hidden_sizes is None else tuple(hidden_sizes)
        key = (method, sizes)
        if key not in self._learners:
            config = {**self.config, "LEARNER": method, "HIDDEN_SIZES": sizes}
            spec = build_cfg(self.metadata["input_dim"], self.metadata["n_classes"], config)
            # Use the same architecture and learner setup as training, without
            # a forward pass or optimizer step. Parameter values do not enter the model.
            with torch.random.fork_rng(devices=[]):
                learner = _make_learner(spec, self.metadata, torch.device("cpu"), config)
                learner.model.to(dtype=str_to_dtype(config["DTYPE"]))
                self._learners[key] = learner
        return self._learners[key]

    def estimate(self, method, steps, hidden_sizes=None):
        cost = self.learner(method, hidden_sizes).estimate_costs(
            batch=self.config["BATCH_SIZE"], time_steps=steps, fp_bytes=self.fp_bytes)
        return dict(memory_bytes=cost["memory"]["total_bytes"],
                    compute_scalars=cost["compute"]["total_scalars"],
                    access_scalars=cost["access"]["total_scalars"])

    def export_metadata(self):
        return dict(**self.dimensions(), dtype=self.config["DTYPE"],
                    full_sequence_steps=self.full_sequence_steps,
                    original_samples=self.original_samples,
                    source_length_counts=dict(sorted(self.length_counts.items())),
                    dataset=self.config["DATASET"], dataset_kwargs=self.config.get("DATASET_KW", {}),
                    seed=self.config["SEED"], data_split=self.metadata["data_split"],
                    train_split_sha256=self.metadata["split_indices_sha256"]["train"])


class CostCatalog:
    """Reuse training profiles across learners and across window trials."""
    def __init__(self):
        self._configs = {
            (entry["RUN_ID_PREFIX"].replace("-", "_") if entry["DATASET"] == "mnist" else entry["DATASET"]):
            {**DEFAULT, **entry} for entry in DATASET_CONFIGS
        }
        self._data = {}
        self._models = {}
        self._evaluations = {}

    def profile(self, name, recorded_config=None):
        current = self._configs[name]
        config = {key: value for key, value in current.items()
                  if key not in {"TRANSFORM", "RUN_ID_PREFIX"}}
        if recorded_config is not None:
            # Use the settings associated with the measured accuracy. All
            # preprocessing and cost assumptions come from the current code.
            tested_keys = (config.keys() | {"DATASET_KW"}) - {"COST_COMPUTE_WEIGHT", "COST_ACCESS_WEIGHT"}
            config.update({key: recorded_config[key] for key in tested_keys
                           if key in recorded_config})
        config["DATA_ROOT"] = str(PROJECT_ROOT / config["DATA_ROOT"])
        data_settings = {key: config.get(key) for key in
                         ("DATASET", "DATA_ROOT", "MAX_SAMPLES", "SEED", "DATA_SPLIT", "DATASET_KW")}
        data_key = (name, json.dumps(data_settings, sort_keys=True))
        if data_key not in self._data:
            print(f"Inspecting training inputs for {name}", flush=True)
            threads = torch.get_num_threads()
            if threads > 4:
                torch.set_num_threads(4)
            try:
                self._data[data_key] = inspect_profile(name, config, deepcopy(current.get("TRANSFORM")))
            finally:
                if threads > 4:
                    torch.set_num_threads(threads)
        model_key = (name, json.dumps(config, sort_keys=True))
        if model_key not in self._models:
            meta, lengths = self._data[data_key]
            self._models[model_key] = ConfiguredCosts(config, meta, lengths)
        return self._models[model_key]

    def evaluate(self, name, method, config, length=None, hop=None):
        profile = self.profile(name, config)
        key = (id(profile), method, length, hop)
        if key in self._evaluations:
            return self._evaluations[key]
        counts = (profile.length_counts if length is None else
                  window_length_counts(profile.length_counts, length, hop))
        learner = profile.learner(method)
        cost = training_costs(
            learner, counts, original_samples=profile.original_samples,
            batch=profile.config["BATCH_SIZE"], fp_bytes=profile.fp_bytes,
            alpha=profile.config["COST_COMPUTE_WEIGHT"], beta=profile.config["COST_ACCESS_WEIGHT"],
            source_length_counts=profile.length_counts,
        )
        cost["profile_source"] = "current_training_split"
        cost["static_bytes"] = learner.get_param_memory_bytes(profile.fp_bytes)
        self._evaluations[key] = cost
        return cost

    def comparison_fields(self, name, method, config, length=None, hop=None):
        cost = self.evaluate(name, method, config, length, hop)
        memory = cost["memory"]
        return dict(
            theory_model_version=cost["model_version"], theory_work_scope=cost["work_scope"],
            windows_per_sample=cost["windows_per_sample"], fp_bytes=cost["fp_bytes"],
            static_bytes=cost["static_bytes"], training_bytes_per_batch=memory["total_bytes"],
            memory_batch_size=cost["batch_size"], memory_time_steps=cost["time_steps"],
            theory_memory_bytes=memory["total_bytes"], theory_memory_scalars=memory["total_scalars"],
            theory_compute_scalars=cost["compute"]["total_scalars"],
            theory_access_scalars=cost["access"]["total_scalars"],
            theory_time_proxy=cost["time_proxy"]["value"],
        )


def inspect_profile(name, config, pipeline):
    set_seed(config["SEED"])
    loaders, meta = get_dataloaders(
        config["DATASET"], root=config["DATA_ROOT"], batch_size=config["BATCH_SIZE"],
        max_samples=config["MAX_SAMPLES"], transform=pipeline, num_workers=0,
        pin_memory=False, seed=config["SEED"], data_split=config.get("DATA_SPLIT"),
        **config.get("DATASET_KW", {}),
    )
    train = loaders["train"].dataset
    if meta["window_length"] is not None or not meta["time_steps"]:
        raise ValueError(f"{name} requires unwindowed [T,D] training samples")
    if _has_fixed_horizon(config, pipeline):
        return meta, Counter({meta["time_steps"]: len(train)})
    lengths = Counter()
    for index in range(len(train)):
        x, _, _ = train[index]
        if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != meta["input_dim"]:
            raise ValueError(f"{name} requires nonempty [T,D] samples with a common feature dimension")
        lengths[int(x.shape[0])] += 1
    return meta, lengths


def window_length_counts(source, length, hop):
    """Apply the training window rule to each distinct recording length."""
    counts = Counter()
    for steps, occurrences in source.items():
        for start, stop in window_spans(steps, length, hop):
            counts[stop - start] += occurrences
    return counts


def report_dataset(run_id, dataset):
    """Keep the two MNIST preprocessing pipelines distinct."""
    prefix = run_id.replace("-", "_")
    for name in ("mnist_static", "mnist_rate"):
        if prefix.startswith(name + "_"):
            return name
    return dataset
