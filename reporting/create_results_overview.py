"""Unified thesis figures: memory, workload, and cycle-proxy estimates.

Run ``python -m reporting.create_results_overview`` for every dataset.
Uses the existing learner cost models and saved runs; never trains a network
or edits the thesis. Each figure is exported as PDF, SVG, and PNG.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from types import SimpleNamespace
from learners.backprop import BackpropLearner
from learners.eprop import EpropLearner
from learners.forward_forward import FFLearner
from learners.pepita import PepitaLearner

ROOT = PROJECT_ROOT / 'results'
LEADERBOARD = ROOT / 'comparisons/all_current_runs/leaderboard.csv'
MIB = 1024 ** 2
METHODS = ('bp', 'ff', 'eprop', 'pepita')
COLOR = {'bp': '#1f77b4', 'ff': '#ff7f0e', 'eprop': '#2ca02c', 'pepita': '#d62728'}
DATASETS = ('har', 'mnist_static', 'mnist_rate', 'pamap2', 'speech_commands',
            'esc50', 'large_scale_audio', 'urban8k', 'dvs_gesture', 'mitbih')


LABEL = {'bp': 'BPTT', 'ff': 'FF', 'eprop': 'E-PROP', 'pepita': 'RATE-PEPITA'}
NAME = dict(zip(DATASETS, ['HAR', 'MNIST Static', 'MNIST Rate', 'PAMAP2',
    'Speech Commands', 'ESC-50', 'Large-Scale Audio', 'UrbanSound8K',
    'DVS Gesture', 'MIT-BIH']))
METRICS = ('compute_scalars', 'access_scalars')
TITLES = ('Computation', 'Memory accesses')
Y_LABELS = ('Arithmetic work (million model units)', 'Scalar accesses (millions)')
STYLES = dict(zip(METHODS, ['-', '--', '-.', ':']))
DEFAULT_OUTPUT = PROJECT_ROOT / 'results/results_overview'
MEMORY_COPY_FAST_CYCLES = 102415
MEMORY_COPY_SLOW_CYCLES = 1103614
MEMORY_COPY_SOURCE = ('https://github.com/project-x51/esp32-s3-memorycopy/blob/'
    'c3a1a5b5bc5372e301ed0f3eb15328857a95777d/README.md')


@dataclass(frozen=True)
class HierarchyProfile:
    """Weighted cycle proxy: one proxy cycle per arithmetic unit and fast access."""
    alpha: float = 1.0
    beta_fast: float = 1.0
    beta_slow: float = MEMORY_COPY_SLOW_CYCLES/MEMORY_COPY_FAST_CYCLES
    fp_bytes: int = 4
    reference_capacity_bytes: int = 512*1024
    external_capacity_bytes: int = 8*MIB

    def __post_init__(self):
        values = (self.alpha, self.beta_fast, self.beta_slow, self.fp_bytes,
                  self.reference_capacity_bytes, self.external_capacity_bytes)
        if any(not np.isfinite(v) or v <= 0 for v in values):
            raise ValueError('Hierarchy profile values must be finite and positive')
        if self.beta_slow < self.beta_fast:
            raise ValueError('Slow-access cost must not be less than fast-access cost')

    def metadata(self):
        return dict(profile_id='esp32s3_scalar_copy_cycle_proxy_v1',
            model='proportional_access_v1', cost_unit='proxy_cycles', **self.__dict__,
            normalized_compute_weight=self.alpha/self.beta_fast,
            slow_fast_access_ratio=self.beta_slow/self.beta_fast,
            arithmetic_basis='Independent reference assumption alpha=1; not DSP benchmark timing',
            fast_service_basis='Normalized reference beta_fast=1',
            slow_service_basis='Matched 32-bit copy cycle ratio; not a universal access latency',
            benchmark=dict(internal_copy_cycles=MEMORY_COPY_FAST_CYCLES,
                external_copy_cycles=MEMORY_COPY_SLOW_CYCLES, bytes_copied=100*1024,
                scalar_reads=25600, scalar_writes=25600, cpu_hz=240e6,
                psram_hz=80e6, psram_mode='Octal DDR',
                code='32-bit scalar for loop; destination cache writeback is timed'),
            sources={
                'memory_copy': MEMORY_COPY_SOURCE,
                'datasheet': 'https://documentation.espressif.com/esp32-s3_datasheet_en.pdf',
                'board': 'https://docs.espressif.com/projects/esp-dev-kits/en/latest/esp32s3/esp32-s3-devkitc-1/user_guide_v1.0.html'},
            assumptions=[
                'Fast fraction = min(1, fast capacity / saved peak working set)',
                'Logical accesses are uniform over a fixed working set',
                'Internal SRAM budget is not hardware data-cache capacity; cache reuse is not simulated',
                'Fixed service coefficients while capacity varies parametrically',
                'Arithmetic model units and logical scalar accesses; additive service, no overlap or overhead term',
                'No additional staging-transfer costs are added to this two-tier proxy',
                'Original optimizer-state exclusions and saved workload estimates are retained',
                'Each setting uses saved mean total work and peak footprint; no per-length reweighting',
                'Full-sequence curves share a dataset BPTT all-slow normalization',
                'Windowed/full ratios compare each method to itself at the same capacity',
                'Capacity sweeps are parametric; capacities above 512 KiB are hypothetical',
                'Nominal capacity fit excludes software reservations and optimizer state',
                'The estimates are not physical cycles, seconds, measured runtime, cache hits or hardware traffic'],
            source_note='The memory ratio is workload-specific. The independent alpha=1 '
                        'assumption is not established by the memory-copy benchmark.')


REFERENCE_PROFILE = HierarchyProfile()


def hierarchy_cost(compute, accesses, memory_bytes, capacity_bytes,
                   profile=REFERENCE_PROFILE):
    """Proportional service costs; capacity can be an array, including zero."""
    for name, value in [('compute', compute), ('accesses', accesses)]:
        if not np.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if not np.isfinite(memory_bytes) or memory_bytes <= 0:
        raise ValueError('Working-set memory must be finite and positive')
    capacity = np.asarray(capacity_bytes, dtype=float)
    if np.any(~np.isfinite(capacity)) or np.any(capacity < 0):
        raise ValueError('Capacity must be finite and nonnegative')
    fraction = np.minimum(1.0, capacity/memory_bytes)
    effective_beta = (fraction*profile.beta_fast
                      + (1-fraction)*profile.beta_slow)
    return dict(fast_fraction=fraction, slow_fraction=1-fraction,
        effective_beta=effective_beta,
        compute_cost=compute*profile.alpha,
        access_cost=accesses*effective_beta,
        estimated_cost=compute*profile.alpha+accesses*effective_beta)

# Published capacity references, not measured free memory for training. Embedded
# memory capacities use binary multiples; banks and external RAM remain distinct
# hardware resources and are not added together here.
MEMORY_REFERENCES = (
    ('nrf5340', 'nRF5340 application core', .5, '512 kB RAM',
     'https://docs.nordicsemi.com/r/bundle/ps_nrf5340/page/chapters/soc_overview/doc/soc_overview.html-memory'),
    ('stm32h743', 'STM32H743', 1, 'up to 1 MB RAM',
     'https://www.st.com/en/microcontrollers-microprocessors/stm32h743ag.html'),
    ('esp32s3_n8r8', 'ESP32-S3-DevKitC-1-N8R8', 8, '8 MB external PSRAM',
     'https://documentation.espressif.com/esp-dev-kits/en/latest/esp32s3/esp32-s3-devkitc-1/user_guide_v1.0.html'),
    ('duo', 'Milk-V Duo (original)', 64, '64 MB DRAM',
     'https://milkv.io/docs/duo/overview'),
    ('duo256m', 'Milk-V Duo 256M', 256, '256 MB DRAM',
     'https://milkv.io/docs/duo/overview'),
    ('pi_zero2w', 'Raspberry Pi Zero 2 W', 512, '512 MB SDRAM',
     'https://www.raspberrypi.com/products/raspberry-pi-zero-2-w/'),
)


def memory_reference_data(fraction=1.0):
    return pd.DataFrame([dict(reference_id=key, device=device,
        published_capacity=published, capacity_bytes=int(mib*MIB),
        training_budget_fraction=fraction, budget_bytes=int(mib*MIB*fraction),
        source_url=url, specification_checked='2026-10-08')
        for key, device, mib, published, url in MEMORY_REFERENCES])


def accuracy_memory_data(baselines):
    """Keep the tested batch, architecture, accuracy and saved peak working set."""
    base, final, _, summaries = load()
    final = final[final.experiment.eq('hop') & final.dataset.isin(baselines)]
    rows = []
    for selected in final.itertuples():
        baseline = base[base.dataset.eq(selected.dataset) & base.learner.eq(selected.learner)]
        if len(baseline) != 1:
            raise ValueError(f'Missing baseline for {selected.dataset}/{selected.learner}')
        baseline = next(baseline.itertuples())
        meta = baselines[selected.dataset]
        for setting, run, accuracy in [('full_sequence', baseline, baseline.final_sample_acc),
                                       ('windowed', selected, selected.test_sample_acc)]:
            summary = summaries[run.run_id]
            checked_run_cost(run, summary, meta)
            theory = summary['memory']['theory']
            memory = theory['memory']['total_bytes']
            if not np.isfinite(memory) or memory <= 0 or not np.isfinite(accuracy) or not 0 <= accuracy <= 100:
                raise ValueError(f'Invalid accuracy or peak memory in {run.run_id}')
            # Reject stale comparison exports rather than silently mixing them
            # with observations from a different run or memory convention.
            if not np.isclose(run.theory_memory_bytes, memory, rtol=0, atol=1):
                raise ValueError(f'Exported peak memory disagrees with {run.run_id}')
            rows.append(dict(dataset=selected.dataset, learner=selected.learner,
                setting=setting, run_id=run.run_id, test_accuracy_pct=accuracy,
                memory_bytes=int(memory), memory_mib=memory/MIB,
                batch_size=meta['batch_size'], hidden_width=meta['baseline_hidden_width'],
                hidden_layers=meta['baseline_hidden_layers'],
                fp_bytes=summary['memory'].get('fp_bytes', 4),
                window_length=int(selected.win_L) if setting == 'windowed' else None,
                hop=int(selected.hop) if setting == 'windowed' else None,
                memory_source='saved_run_summary',
                window_tail_rule=summary['meta'].get('window_tail_rule', 'historical_unchanged')))
    data = pd.DataFrame(rows)
    if len(data) != len(baselines)*8 or data.duplicated(['dataset', 'learner', 'setting']).any():
        raise ValueError('Incomplete accuracy/memory comparison matrix')
    return data


def memory_budget_data(points, references):
    data = points.merge(references, how='cross')
    data['fits_modelled_memory'] = data.memory_bytes <= data.budget_bytes
    data['remaining_budget_bytes'] = data.budget_bytes - data.memory_bytes
    return data


def plot_accuracy_memory(dataset, points, references, out):
    fig, ax = plt.subplots(figsize=(10, 6.3))
    low, high = points.memory_mib.min()*.72, points.memory_mib.max()*1.45
    visible = references[(references.budget_bytes/MIB).between(low, high)]
    if visible.empty:
        # Include the nearest reference on the right when none crosses the data.
        above = references[references.budget_bytes/MIB >= points.memory_mib.max()]
        visible = above.head(1)
        if not visible.empty:
            high = max(high, visible.budget_bytes.iloc[0]/MIB*1.25)
    for ref in visible.itertuples():
        x = ref.budget_bytes/MIB
        ax.axvline(x, color='#777777', ls='--', lw=1, alpha=.65, zorder=1)
        short_name = {'nrf5340': 'nRF5340', 'stm32h743': 'STM32H743',
                      'esp32s3_n8r8': 'ESP32-S3 N8R8', 'duo': 'Milk-V Duo',
                      'duo256m': 'Milk-V Duo 256M', 'pi_zero2w': 'Pi Zero 2 W'}[ref.reference_id]
        ax.text(x, 1.012, f'{x:g} MiB\n{short_name}',
                transform=ax.get_xaxis_transform(), ha='center', va='bottom',
                fontsize=8, color='#555555')
    for method in METHODS:
        d = points[points.learner.eq(method)].set_index('setting')
        full, window = d.loc['full_sequence'], d.loc['windowed']
        ax.annotate('', xy=(window.memory_mib, window.test_accuracy_pct),
                    xytext=(full.memory_mib, full.test_accuracy_pct),
                    arrowprops=dict(arrowstyle='-|>', color=COLOR[method], lw=1.8,
                                    shrinkA=7, shrinkB=7, mutation_scale=12), zorder=2)
        ax.scatter(full.memory_mib, full.test_accuracy_pct, marker='o', s=75,
                   facecolors='white', edgecolors=COLOR[method], linewidths=1.8, zorder=4)
        ax.scatter(window.memory_mib, window.test_accuracy_pct, marker='D', s=65,
                   color=COLOR[method], edgecolors='white', linewidths=.6, zorder=5)
    ax.set_xscale('log')
    ax.set_xlim(low, high)
    ax.xaxis.set_major_locator(LogLocator(base=2, numticks=12))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    acc_min, acc_max = points.test_accuracy_pct.min(), points.test_accuracy_pct.max()
    pad = max(3, (acc_max-acc_min)*.18)
    ax.set_ylim(max(0, acc_min-pad), min(100.8, acc_max+pad))
    ax.set_xlabel('Estimated peak training memory (MiB, logarithmic scale)')
    ax.set_ylabel('Test accuracy (%)')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    fig.suptitle(NAME[dataset] + ' - Accuracy within a memory budget', fontsize=13, y=.99)
    method_handles = [Line2D([0], [0], color=COLOR[m], lw=2, label=LABEL[m]) for m in METHODS]
    fig.legend(handles=method_handles, loc='lower center', bbox_to_anchor=(.5, .083),
               ncol=4, frameon=False)
    fig.legend(handles=[Line2D([0], [0], marker='o', color='#444444', markerfacecolor='white',
               ls='', markersize=7, label='Full sequence'),
               Line2D([0], [0], marker='D', color='#444444', ls='', markersize=6,
               label='Selected window')], loc='lower center', bbox_to_anchor=(.5, .04),
               ncol=2, frameon=False)
    fig.subplots_adjust(left=.085, right=.985, bottom=.245, top=.90)
    path = out/dataset/'06_accuracy_memory_budget'
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(path.with_suffix('.'+ext), dpi=180, bbox_inches='tight')
    plt.close(fig)


def read_csv(path: Path, columns: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Saved result is missing: {path}")
    data = pd.read_csv(path)
    if columns - set(data):
        raise ValueError(f"{path} lacks columns {sorted(columns - set(data))}")
    return data

def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict]]:
    base = read_csv(ROOT / "comparisons/all_current_runs/leaderboard.csv",
                    {"dataset", "learner", "run_id", "final_sample_acc"} |
                    {f"theory_{m}" for m in METRICS})
    final = read_csv(ROOT / "comparisons/window_hop_independent/final_test_results.csv",
                     {"dataset", "learner", "run_id", "experiment", "status", "win_L",
                      "hop", "test_sample_acc", "validation_score", "selection_phase"} |
                     {f"theory_{m}" for m in METRICS})
    trials = read_csv(ROOT / "comparisons/window_hop_independent/all_trials.csv",
                      {"dataset", "learner", "run_id", "experiment", "phase", "win_L",
                       "hop", "sample_acc", "windows_per_sample"} |
                      {f"theory_{m}" for m in METRICS})
    base = base[base.dataset.isin(DATASETS) & base.learner.isin(METHODS)].copy()
    final = final[final.dataset.isin(DATASETS) & final.learner.isin(METHODS) &
                  final.status.eq("ok")].copy()
    trials = trials[trials.dataset.isin(DATASETS) & trials.learner.isin(METHODS) &
                    trials.sample_acc.notna()].copy()
    expected = len(DATASETS) * len(METHODS)
    if base.duplicated(["dataset", "learner"]).any() or len(base) != expected:
        raise ValueError("No-window results must have one complete dataset/method matrix")
    if final.duplicated(["dataset", "learner", "experiment"]).any() or any(
            final.experiment.eq(kind).sum() != expected for kind in ("window", "hop")):
        raise ValueError("Both final-test experiment matrices must be complete")
    summaries = {}
    for run_id in set(base.run_id) | set(final.run_id):
        path = ROOT / "runs" / run_id / "summary.json"
        if not path.exists():
            raise FileNotFoundError(f"Run summary is missing: {path}")
        with path.open(encoding="utf-8") as stream:
            summaries[run_id] = json.load(stream)
    return base, final, trials, summaries

def model_cost(method: str, *, batch: int = 128, steps: int = 128,
               input_dim: int = 9, classes: int = 6, width: int = 128,
               depth: int = 1) -> dict:
    widths = [width] * depth
    inputs = [input_dim + classes if method == "ff" else input_dim] + widths[:-1]
    weights = [left * right for left, right in zip(inputs, widths)]
    readout = 0 if method == "ff" else classes * widths[-1]
    f = [w + left + 2 * right for w, left, right
         in zip(weights, inputs, widths)]
    p = [w + left + right for w, left, right
         in zip(weights, inputs, widths)]
    dims = {"C": classes, "d0": input_dim, "tilde_d0": inputs[0],
            "d": widths, "tilde_inputs": inputs, "a": weights,
            "A": sum(weights), "U": sum(widths), "V": readout,
            "F": sum(f), "P": sum(p),
            "Y": readout + widths[-1] + classes if readout else 0,
            "f_l": f, "p_l": p, "L": depth}
    functions = {"bp": BackpropLearner._cost_components,
                 "eprop": EpropLearner._cost_components,
                 "ff": FFLearner._cost_components,
                 "pepita": PepitaLearner._cost_components}
    owner = SimpleNamespace(mode="accum") if method == "pepita" else None
    components = functions[method](owner, batch, steps, dims)
    return {"memory_bytes": 4 * sum(components["memory"].values()),
            "compute_scalars": sum(components["compute"].values()),
            "access_scalars": sum(components["access"].values()),
            "components": components}

def load_baselines(datasets: list[str]) -> dict[str, dict]:
    if not LEADERBOARD.exists():
        raise FileNotFoundError(f"Saved no-window results are missing: {LEADERBOARD}")
    runs = pd.read_csv(LEADERBOARD)
    required = {"dataset", "learner", "run_id", "batch", "hidden",
                "hidden_depth", "theory_memory_bytes"}
    if required - set(runs):
        raise ValueError(f"Leaderboard lacks {sorted(required - set(runs))}")
    baselines = {}
    for dataset in datasets:
        group = runs[runs.dataset.eq(dataset)].set_index("learner")
        if group.index.duplicated().any() or not set(METHODS).issubset(group.index):
            raise ValueError(f"Incomplete no-window method set for {dataset}")
        group = group.loc[list(METHODS)]
        source = group.loc["bp"]
        summary_path = ROOT / "runs" / source.run_id / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Saved run summary is missing: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        meta = summary["meta"]
        theory = summary["memory"]["theory"]
        lengths = theory["instance_length_counts"]
        batch, width, depth = int(source.batch), int(source.hidden), int(source.hidden_depth)
        # Sample-zero metadata need not describe the longest sequence. The
        # saved cost horizon corresponds to the reported peak working set.
        full_steps = int(theory.get("time_steps", max(map(int, lengths))))
        input_dim, classes = int(meta["input_dim"]), int(meta["n_classes"])
        if full_steps < 16:
            raise ValueError(f"{dataset} has only {full_steps} time steps; S=16 does not fit")
        for method, row in group.iterrows():
            if (int(row.batch), int(row.hidden), int(row.hidden_depth)) != (batch, width, depth):
                raise ValueError(f"Saved model dimensions differ across {dataset} methods")
            estimate = model_cost(method, batch=batch, steps=full_steps,
                                  input_dim=input_dim, classes=classes,
                                  width=width, depth=depth)["memory_bytes"]
            if not np.isclose(estimate, row.theory_memory_bytes, rtol=0, atol=1):
                raise ValueError(f"Cost model disagrees with saved {dataset}/{method} memory")
        baselines[dataset] = {
            "batch_size": batch, "input_features": input_dim, "classes": classes,
            "full_sequence_steps": full_steps, "baseline_hidden_width": width,
            "baseline_hidden_layers": depth, "source_run_id": source.run_id,
            "distinct_saved_lengths": len(lengths),
        }
    return baselines


def estimate(method, meta, *, steps=None, width=None, depth=None):
    return model_cost(method, batch=meta['batch_size'],
        input_dim=meta['input_features'], classes=meta['classes'],
        steps=meta['full_sequence_steps'] if steps is None else steps,
        width=meta['baseline_hidden_width'] if width is None else width,
        depth=meta['baseline_hidden_layers'] if depth is None else depth)


def theoretical_data(baselines):
    rows = []
    for dataset, meta in baselines.items():
        sequences = sorted(set(range(8, 1025, 16)) | {1024, meta['full_sequence_steps']})
        max_depth = max(8, int(meta['baseline_hidden_layers']))
        depths = range(1, max_depth + 1)
        for sweep, values in [('sequence_length', sequences), ('hidden_depth', depths)]:
            for value in values:
                steps = value if sweep == 'sequence_length' else meta['full_sequence_steps']
                depth = value if sweep == 'hidden_depth' else meta['baseline_hidden_layers']
                for method in METHODS:
                    cost = estimate(method, meta, steps=steps, depth=depth)
                    rows.append(dict(dataset=dataset, sweep=sweep, value=value,
                        learner=method, steps=steps,
                        width=meta['baseline_hidden_width'], depth=depth,
                        batch_size=meta['batch_size'], input_features=meta['input_features'],
                        classes=meta['classes'], **{m: cost[m] for m in ('memory_bytes', *METRICS)}))
    return pd.DataFrame(rows)


def nonoverlapping_window_profile(length_counts, window_length):
    """Partition every saved sequence once, keeping its shorter final window."""
    if window_length < 1 or int(window_length) != window_length:
        raise ValueError('Window length must be a positive integer')
    profile = Counter()
    for steps, count in length_counts.items():
        steps = int(steps)
        if steps < 1 or count < 1 or int(count) != count:
            raise ValueError('Sequence lengths and counts must be positive integers')
        whole, tail = divmod(steps, int(window_length))
        if whole:
            profile[int(window_length)] += whole*int(count)
        if tail:
            profile[tail] += int(count)
    if not profile:
        raise ValueError('Window sweep requires a nonempty source-length profile')
    return dict(profile)


def controlled_windowing_data(baselines):
    """Sweep all integer L at hop=L, with fixed saved dimensions and input.

    Work sums every non-overlapping window before averaging over original
    sequences, as in the saved cost reports. Memory is the largest single
    window's working set. Hypothetical settings have no accuracy estimates.
    """
    base, _, _, summaries = load()
    rows = []
    for dataset, meta in baselines.items():
        group = base[base.dataset.eq(dataset)].set_index('learner').loc[list(METHODS)]
        common = summaries[meta['source_run_id']]['memory']['theory']
        lengths = common['instance_length_counts']
        samples = common['original_samples']
        horizon = meta['full_sequence_steps']
        if sum(lengths.values()) != samples or max(map(int, lengths)) != horizon:
            raise ValueError(f'Inconsistent full-sequence length profile for {dataset}')
        source_steps = sum(int(t)*n for t, n in lengths.items())
        window_profiles = {length: nonoverlapping_window_profile(lengths, length)
                           for length in range(1, horizon+1)}
        for method, baseline in group.iterrows():
            summary = summaries[baseline.run_id]
            theory = summary['memory']['theory']
            if (theory['instance_length_counts'] != lengths
                    or theory['original_samples'] != samples):
                raise ValueError(f'Controlled sweep needs a common length profile for {dataset}')
            run = SimpleNamespace(learner=method, run_id=baseline.run_id)
            full_work = checked_run_cost(run, summary, meta)
            full_memory = theory['memory']['total_bytes']
            # Reuse pure model evaluations, never initialize or train learners.
            costs = {t: estimate(method, meta, steps=t) for t in range(1, horizon+1)}
            if not np.isclose(costs[horizon]['memory_bytes'], full_memory, rtol=0, atol=1):
                raise ValueError(f'Controlled sweep disagrees with saved {dataset}/{method} memory')
            for length, profile in window_profiles.items():
                work = {metric: sum(costs[t][metric]*n for t, n in profile.items())/samples
                        for metric in METRICS}
                peak_steps = max(profile)
                memory = costs[peak_steps]['memory_bytes']
                rows.append(dict(dataset=dataset, learner=method,
                    baseline_run_id=baseline.run_id, source_profile_run_id=meta['source_run_id'],
                    batch_size=meta['batch_size'], input_features=meta['input_features'],
                    classes=meta['classes'], hidden_width=meta['baseline_hidden_width'],
                    hidden_layers=meta['baseline_hidden_layers'], fp_bytes=4,
                    full_sequence_steps=horizon, window_length=length, hop=length,
                    relative_window_length=length/horizon, effective_hop_ratio=1.0,
                    window_tail_rule='partial_last_window_v1',
                    source_profile='saved_full_sequence_instance_length_counts',
                    original_samples=samples, distinct_source_lengths=len(lengths),
                    windows_per_sample=sum(profile.values())/samples,
                    source_steps_per_sample=source_steps/samples,
                    processed_steps_per_sample=sum(t*n for t, n in profile.items())/samples,
                    peak_window_steps=peak_steps, baseline_memory_bytes=full_memory,
                    windowed_memory_bytes=memory, memory_ratio=memory/full_memory,
                    **{f'baseline_{m}': full_work[m] for m in METRICS},
                    **{f'windowed_{m}': work[m] for m in METRICS},
                    **{f'{m}_ratio': work[m]/full_work[m] for m in METRICS}))
    return pd.DataFrame(rows)


def checked_run_cost(row, summary, meta):
    """Reproduce saved work using the observed length distribution, not max T.

    This is essential for variable-length UrbanSound8K and for short/padded
    sequences. The profile already includes the selected window and hop.
    """
    theory = summary['memory']['theory']
    config = summary['config']
    if (config['BATCH_SIZE'] != meta['batch_size'] or
        config['HIDDEN_SIZES'] != [meta['baseline_hidden_width']] * meta['baseline_hidden_layers']):
        raise ValueError(f'Inconsistent architecture/batch in {row.run_id}')
    for key, expected in [('input_dim', meta['input_features']), ('n_classes', meta['classes'])]:
        if summary['meta'][key] != expected:
            raise ValueError(f'Inconsistent {key} in {row.run_id}')
    profile = theory['instance_length_counts']
    samples = theory['original_samples']
    if samples <= 0 or not profile or any(int(t) <= 0 or n <= 0 for t, n in profile.items()):
        raise ValueError(f'Invalid training-length profile in {row.run_id}')
    total = {m: 0.0 for m in METRICS}
    for steps, count in profile.items():
        costs = estimate(row.learner, meta, steps=int(steps))
        for metric in METRICS:
            total[metric] += costs[metric] * count / samples
    for metric, category in zip(METRICS, ['compute', 'access']):
        saved = theory[category]['total_scalars']
        if not np.isfinite(saved) or saved <= 0 or not np.isclose(total[metric], saved, rtol=1e-9, atol=1):
            raise ValueError(f'Current model disagrees with saved {row.run_id}/{metric}')
        total[metric] = saved
    return total


def selected_data(baselines):
    base, final, _, summaries = load()
    final = final[final.experiment.eq('hop') & final.dataset.isin(baselines)]
    rows = []
    for selected in final.itertuples():
        baseline = base[base.dataset.eq(selected.dataset) & base.learner.eq(selected.learner)]
        if len(baseline) != 1:
            raise ValueError(f'Missing/duplicate baseline for {selected.dataset}/{selected.learner}')
        baseline = next(baseline.itertuples())
        meta = baselines[selected.dataset]
        baseline_summary = summaries[baseline.run_id]
        selected_summary = summaries[selected.run_id]
        b = checked_run_cost(baseline, baseline_summary, meta)
        w = checked_run_cost(selected, selected_summary, meta)
        rows.append(dict(dataset=selected.dataset, learner=selected.learner,
            baseline_run_id=baseline.run_id, selected_run_id=selected.run_id,
            window_length=int(selected.win_L), hop=int(selected.hop),
            effective_hop_ratio=selected.hop/selected.win_L,
            selected_validation_accuracy=selected.validation_score,
            baseline_test_accuracy=baseline.final_sample_acc,
            selected_test_accuracy=selected.test_sample_acc,
            **{f'baseline_{m}': b[m] for m in METRICS},
            **{f'selected_{m}': w[m] for m in METRICS},
            **{f'{m}_ratio': w[m]/b[m] for m in METRICS}))
    data = pd.DataFrame(rows)
    if len(data) != len(baselines)*4 or data.duplicated(['dataset','learner']).any():
        raise ValueError('Incomplete selected-run comparison matrix')
    return data


def _capacity_runs(selected, points, setting, profile):
    """Match saved memory to the exact run whose validated work is used."""
    keys = ['dataset', 'learner']
    runs = points[points.setting.eq(setting)].copy()
    if (runs.duplicated(keys).any() or selected.duplicated(keys).any()
            or set(map(tuple, runs[keys].to_numpy()))
            != set(map(tuple, selected[keys].to_numpy()))):
        raise ValueError(f'Capacity sweep requires matching unique {setting} runs')
    if runs.empty or not runs.fp_bytes.eq(profile.fp_bytes).all():
        raise ValueError('Capacity sweep requires the reference FP32 precision')
    data = selected.merge(runs[keys+['run_id', 'memory_bytes', 'batch_size']],
                          on=keys, validate='one_to_one')
    source = 'baseline_run_id' if setting == 'full_sequence' else 'selected_run_id'
    if not data.run_id.eq(data[source]).all():
        raise ValueError('Capacity-sweep memory and work must come from the same run')
    for dataset, group in data.groupby('dataset', sort=False):
        if set(group.learner) != set(METHODS) or len(group) != len(METHODS):
            raise ValueError(f'Capacity sweep needs all four methods for {dataset}')
    return data


def _capacity_grid(memories, profile):
    memories = np.asarray(memories, dtype=float)
    if np.any(~np.isfinite(memories)) or np.any(memories <= 0):
        raise ValueError('Capacity-sweep memory must be finite and positive')
    low = min(memories.min()/100, profile.reference_capacity_bytes/4)
    high = max(memories.max()*4, profile.reference_capacity_bytes*2)
    return np.unique(np.r_[np.geomspace(low, high, 321), memories,
                           profile.reference_capacity_bytes])


def windowing_tradeoff_data(selected, points):
    """Pair total-work ratios with the same saved runs' peak-memory ratio."""
    keys = ['dataset', 'learner']
    data = selected.copy()
    for setting, prefix in [('full_sequence', 'baseline'), ('windowed', 'selected')]:
        runs = _capacity_runs(selected, points, setting, REFERENCE_PROFILE)
        columns = runs[keys+['memory_bytes', 'batch_size']].rename(columns={
            'memory_bytes': prefix+'_memory_bytes', 'batch_size': prefix+'_batch_size'})
        data = data.merge(columns, on=keys, validate='one_to_one')
    if not data.baseline_batch_size.eq(data.selected_batch_size).all():
        raise ValueError('Windowing tradeoffs require the same original-sequence batch size')
    memories = data[['baseline_memory_bytes', 'selected_memory_bytes']].to_numpy()
    if np.any(~np.isfinite(memories)) or np.any(memories <= 0):
        raise ValueError('Windowing tradeoffs require finite positive peak memory')
    data['memory_ratio'] = data.selected_memory_bytes/data.baseline_memory_bytes
    return data


def capacity_sweep_data(selected, points, profile=REFERENCE_PROFILE):
    """Use saved full-sequence mean work and peak memory, without retraining.

    One common BPTT all-slow denominator per dataset preserves differences
    between methods. Each footprint and the reference capacity are exact grid
    points, so transitions and the physical reference can be reproduced.
    """
    data = _capacity_runs(selected, points, 'full_sequence', profile)
    rows = []
    for dataset, group in data.groupby('dataset', sort=False):
        capacities = _capacity_grid(group.memory_bytes, profile)
        bp = group[group.learner.eq('bp')].iloc[0]
        reference = float(hierarchy_cost(bp.baseline_compute_scalars,
            bp.baseline_access_scalars, bp.memory_bytes, 0, profile)['estimated_cost'])
        if reference <= 0:
            raise ValueError(f'BPTT normalization must be positive in {dataset}')
        for run in group.itertuples():
            values = hierarchy_cost(run.baseline_compute_scalars,
                run.baseline_access_scalars, run.memory_bytes, capacities, profile)
            for i, capacity in enumerate(capacities):
                rows.append(dict(dataset=dataset, learner=run.learner,
                    setting='full_sequence', run_id=run.run_id,
                    batch_size=run.batch_size, memory_bytes=run.memory_bytes,
                    memory_mib=run.memory_bytes/MIB, capacity_bytes=capacity,
                    capacity_mib=capacity/MIB,
                    compute_scalars=run.baseline_compute_scalars,
                    access_scalars=run.baseline_access_scalars,
                    fast_fraction=values['fast_fraction'][i],
                    slow_fraction=values['slow_fraction'][i],
                    effective_beta=values['effective_beta'][i],
                    compute_cost_units=values['compute_cost'],
                    access_cost_units=values['access_cost'][i],
                    estimated_cost_units=values['estimated_cost'][i],
                    normalized_cost=values['estimated_cost'][i]/reference,
                    test_accuracy_pct=run.baseline_test_accuracy,
                    fits_nominal_fast_memory=run.memory_bytes <= profile.reference_capacity_bytes,
                    fits_nominal_total_memory=run.memory_bytes <= profile.reference_capacity_bytes+profile.external_capacity_bytes,
                    normalization_reference_run_id=bp.run_id,
                    normalization_reference_cost_units=reference))
    return pd.DataFrame(rows)


def windowed_capacity_sweep_data(selected, points, profile=REFERENCE_PROFILE):
    """Compare total selected-window work with the same method's full run.

    Each setting uses its saved peak working set for the proportional split.
    Saved work already includes all windows, overlap and partial tails; never
    multiply it by window count again or sum per-window memory footprints.
    """
    keys = ['dataset', 'learner']
    full = _capacity_runs(selected, points, 'full_sequence', profile)
    windowed = _capacity_runs(selected, points, 'windowed', profile)
    data = full.merge(windowed[keys+['memory_bytes', 'batch_size']], on=keys,
                      suffixes=('', '_selected'), validate='one_to_one')
    if not data.batch_size.eq(data.batch_size_selected).all():
        raise ValueError('Windowed/full costs must use the same original-sequence batch size')
    frames = []
    for dataset, group in data.groupby('dataset', sort=False):
        capacities = _capacity_grid(np.r_[group.memory_bytes,
                                          group.memory_bytes_selected], profile)
        for run in group.itertuples():
            baseline = hierarchy_cost(run.baseline_compute_scalars,
                run.baseline_access_scalars, run.memory_bytes, capacities, profile)
            window = hierarchy_cost(run.selected_compute_scalars,
                run.selected_access_scalars, run.memory_bytes_selected, capacities, profile)
            if np.any(baseline['estimated_cost'] <= 0):
                raise ValueError(f'Full-sequence denominator must be positive in {run.baseline_run_id}')
            columns = dict(dataset=dataset, learner=run.learner,
                baseline_run_id=run.baseline_run_id, selected_run_id=run.selected_run_id,
                batch_size=run.batch_size, window_length=run.window_length, hop=run.hop,
                effective_hop_ratio=run.effective_hop_ratio,
                baseline_test_accuracy=run.baseline_test_accuracy,
                selected_test_accuracy=run.selected_test_accuracy,
                baseline_memory_bytes=run.memory_bytes,
                selected_memory_bytes=run.memory_bytes_selected,
                baseline_compute_scalars=run.baseline_compute_scalars,
                baseline_access_scalars=run.baseline_access_scalars,
                selected_compute_scalars=run.selected_compute_scalars,
                selected_access_scalars=run.selected_access_scalars,
                capacity_bytes=capacities, capacity_mib=capacities/MIB,
                windowed_full_cost_ratio=window['estimated_cost']/baseline['estimated_cost'])
            for prefix, costs in [('baseline', baseline), ('selected', window)]:
                for key in ('fast_fraction', 'slow_fraction', 'effective_beta'):
                    columns[f'{prefix}_{key}'] = costs[key]
                for key in ('compute_cost', 'access_cost', 'estimated_cost'):
                    columns[f'{prefix}_{key}_units'] = costs[key]
            frames.append(pd.DataFrame(columns))
    return pd.concat(frames, ignore_index=True)


def _mark_reference_capacity(ax, profile):
    capacity_mib = profile.reference_capacity_bytes/MIB
    ax.axvline(capacity_mib, color='#666666', ls='--', lw=1, zorder=2)
    ax.text(capacity_mib, .98,
            f'Reference: {profile.reference_capacity_bytes/1024:g} KiB',
            transform=ax.get_xaxis_transform(), rotation=90,
            ha='right', va='top', fontsize=8, color='#555555',
            bbox=dict(facecolor='white', edgecolor='none', alpha=.8, pad=1.5))


def plot_capacity_sweep(dataset, data, out, profile=REFERENCE_PROFILE):
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for method in METHODS:
        d = data[data.learner.eq(method)].sort_values('capacity_bytes')
        ax.plot(d.capacity_mib, d.normalized_cost,
                color=COLOR[method], linestyle=STYLES[method], lw=2)
        fits = d[d.capacity_bytes.eq(d.memory_bytes)]
        ax.scatter(fits.capacity_mib, fits.normalized_cost, s=46,
                   facecolors='white', edgecolors=COLOR[method], linewidths=1.6,
                   zorder=4)
    ax.axhline(1, color='#aaaaaa', ls='--', lw=.8, zorder=0)
    ax.set_xscale('log', base=2)
    ax.set_xlim(data.capacity_mib.min(), data.capacity_mib.max())
    ax.xaxis.set_major_locator(LogLocator(base=2, numticks=9))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:.3g}'))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_ylim(0, data.normalized_cost.max()*1.08)
    _mark_reference_capacity(ax, profile)
    ax.set_xlabel('Fast-memory capacity (MiB, logarithmic scale)')
    ax.set_ylabel('Cycle proxy\n(relative to BPTT all-slow proxy)')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=.2)
    ax.set_axisbelow(True)
    fig.suptitle(NAME[dataset] + ' - Full-sequence cycle proxy vs fast-memory capacity',
                 fontsize=12, y=.98)
    handles = [Line2D([0], [0], color=COLOR[m], linestyle=STYLES[m], lw=2,
                       label=LABEL[m]) for m in METHODS]
    handles.append(Line2D([0], [0], marker='o', color='#666666',
        markerfacecolor='white', ls='', markersize=6, label='Working set fits'))
    fig.legend(handles=handles, ncol=5, loc='lower center',
               bbox_to_anchor=(.5, .047), frameon=False, fontsize=9)
    fig.text(.5, .015, f'Cycle proxy: alpha = beta_fast = 1; beta_slow = {profile.beta_slow:.2f}. '
             'Proportional placement; not measured runtime.',
             ha='center', fontsize=8, color='#666666')
    fig.tight_layout(rect=(0, .115, 1, .97))
    path = out/dataset/'05_fast_memory_capacity_sweep'
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(path.with_suffix('.'+ext), dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_windowed_capacity_sweep(dataset, data, out, profile=REFERENCE_PROFILE):
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for method in METHODS:
        d = data[data.learner.eq(method)].sort_values('capacity_bytes')
        ax.plot(d.capacity_mib, d.windowed_full_cost_ratio,
                color=COLOR[method], linestyle=STYLES[method], lw=2)
    ax.axhline(1, color='#444444', ls='--', lw=1)
    ax.set_xscale('log', base=2)
    ax.set_xlim(data.capacity_mib.min(), data.capacity_mib.max())
    ax.xaxis.set_major_locator(LogLocator(base=2, numticks=9))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:.3g}'))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_ylim(0, max(1.1, data.windowed_full_cost_ratio.max()*1.08))
    _mark_reference_capacity(ax, profile)
    ax.set_xlabel('Fast-memory capacity (MiB, logarithmic scale)')
    ax.set_ylabel('Cycle-proxy ratio\n(selected windows / own full sequence)')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=.2)
    ax.set_axisbelow(True)
    fig.suptitle(NAME[dataset] + ' - Windowed/full cycle proxy vs fast-memory capacity',
                 fontsize=12, y=.98)
    fig.legend(handles=[Line2D([0], [0], color=COLOR[m], linestyle=STYLES[m],
                   lw=2, label=LABEL[m]) for m in METHODS],
               ncol=4, loc='lower center', bbox_to_anchor=(.5, .047), frameon=False)
    fig.text(.5, .015, f'Below 1: fewer proxy cycles with windows. '
             f'alpha = beta_fast = 1; beta_slow = {profile.beta_slow:.2f}. Not measured runtime.',
             ha='center', fontsize=8, color='#666666')
    fig.tight_layout(rect=(0, .115, 1, .97))
    path = out/dataset/'07_windowed_cost_capacity_sweep'
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(path.with_suffix('.'+ext), dpi=180, bbox_inches='tight')
    plt.close(fig)


def write_hierarchy_notes(output, profile=REFERENCE_PROFILE):
    metadata = profile.metadata()
    (output/'memory_hierarchy_profile.json').write_text(
        json.dumps(metadata, indent=2), encoding='utf-8')
    pd.DataFrame([
        dict(coefficient='alpha', normalized_value=profile.alpha,
             unit='proxy cycles/arithmetic model unit',
             derivation='Independent simplifying reference assumption', source_url=''),
        dict(coefficient='beta_fast', normalized_value=profile.beta_fast,
             unit='proxy cycles/logical scalar access',
             derivation='Fast-access cost defines the reference unit', source_url=''),
        dict(coefficient='beta_slow', normalized_value=profile.beta_slow,
             unit='proxy cycles/logical scalar access',
             derivation=f'beta_fast * ({MEMORY_COPY_SLOW_CYCLES} / {MEMORY_COPY_FAST_CYCLES})',
             source_url=MEMORY_COPY_SOURCE),
    ]).to_csv(output/'memory_hierarchy_coefficients.csv', index=False)
    (output/'memory_hierarchy_README.md').write_text(f"""# Memory-hierarchy cycle proxy

Cycle proxies assign one proxy cycle to an arithmetic model unit or fast access.
They are weighted counts, distinct from measured processor cycles.
The reference is alpha={profile.alpha:g}, beta_fast={profile.beta_fast:g}, and
beta_slow={profile.beta_slow:.12g}. Plot labels round the slow/fast ratio to 10.78;
calculations retain the exact ratio {MEMORY_COPY_SLOW_CYCLES}/{MEMORY_COPY_FAST_CYCLES}.
Alpha=1 is an independent model assumption, not inferred from memory benchmarks.
No matrix-multiplication timing is added as a supposedly pure arithmetic cost.

## Provenance and scope

[The pinned benchmark results]({MEMORY_COPY_SOURCE}) report the same 100 KiB
32-bit scalar copy workload at 102,415 cycles internally and 1,103,614 cycles
in PSRAM. Each copy has 25,600 scalar reads and 25,600 scalar writes, so the
access-count denominator cancels in their ratio. The source uses 240 MHz CPU,
80 MHz Octal PSRAM, and times destination-cache writeback. This is an observed
whole-copy slowdown, including loop and cache behavior, not a universal load
latency or a measured SNN training cost. Other kernels can yield different ratios.

Nominal fast capacity is {profile.reference_capacity_bytes/1024:g} KiB on-chip SRAM;
external capacity is {profile.external_capacity_bytes/MIB:g} MiB PSRAM. These are
optimistic model budgets, without software reservations. SRAM capacity is not
cache capacity. Source URLs and source observations are in the profile JSON.

## Model

    f = min(1, K/W)
    m = 1-f
    beta_effective = beta_fast + m*(beta_slow-beta_fast)
    estimated_cost = alpha*compute + beta_effective*accesses

Beta_slow is the total slow-service cost; the additional penalty is
beta_slow-beta_fast. Uniform access frequency and fixed proportional placement
are assumptions, not cache-miss predictions. Locality, vectorization, separate
read/write costs and overlap are not simulated. No extra overhead or staging
term is added. Optimizer-state exclusions remain those of the original model.

Saved total work already includes every selected window, overlap, gap and tail.
Work is averaged over original sequences at the configured batch size, never
multiplied again by window count. Each setting uses its own peak working set;
window memories are not summed. Peak memory weights the saved mean work under
the whole-working-set approximation, without per-length beta reweighting.

## Figures and normalization

- 05_fast_memory_capacity_sweep: full-sequence costs versus log capacity, divided
  by one constant per dataset: full BPTT all-slow cost at the reference ratio.
  Open circles mark each footprint. The grid includes all footprints and 512 KiB.
  A labeled vertical line marks the nominal SRAM reference in both capacity charts.
  Values beyond nominal SRAM are
  hypothetical capacity scenarios, not adjustable SRAM on an actual ESP32-S3.
- 07_windowed_cost_capacity_sweep: selected-window cost divided by that method's
  own full-sequence cost at the same capacity. Below 1 means lower cycle proxy.
  Different methods have different denominators; these ratios do not rank their
  absolute costs. All eight footprints occur exactly on the grid.

Corresponding CSVs retain raw C/M counts, cycle proxies in reference units, footprint,
source run IDs and assumptions. Reference CSVs evaluate exactly 512 KiB.
Nominal fit columns are capacity arithmetic only, not deployment validation;
full BPTT/FF may exceed the actual board's total nominal RAM. Curves remain
parametric and are not filtered by board feasibility. Saved test accuracies are
context; selected settings are not accuracy-matched. No training is rerun.

The former Vega ns/ms columns and time-named exports are removed on regeneration.
""", encoding='utf-8')



def style(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.set_yscale('log', base=10)
    ax.grid(alpha=.2, which='both')
    ax.set_axisbelow(True)


def finish(fig, title, path, *, legend=True, bar_legend=False):
    fig.suptitle(title, fontsize=13, y=.99)
    if legend:
        handles = ([Patch(facecolor=COLOR[m], label=LABEL[m]) for m in METHODS]
                   if bar_legend else
                   [Line2D([0],[0], color=COLOR[m], linestyle=STYLES[m],
                           linewidth=2, label=LABEL[m]) for m in METHODS])
        fig.legend(handles=handles, ncol=4, loc='lower center', frameon=False)
    fig.tight_layout(rect=(0, .07 if legend else .02, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ['pdf','svg','png']:
        fig.savefig(path.with_suffix('.'+ext), dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_theory(dataset, meta, data, out):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4))
    panels = [(sweep, metric, xlabel, prefix + ' - ' + title)
              for sweep, xlabel, prefix in [('sequence_length', 'Time steps', 'Sequence length'),
                                            ('hidden_depth', 'Hidden layers', 'Network depth')]
              for metric, title in zip(METRICS, TITLES)]
    for ax, (sweep, metric, xlabel, title) in zip(axes.flat, panels):
        for method in METHODS:
            d = data[data.sweep.eq(sweep) & data.learner.eq(method)]
            ax.plot(d.value, d[metric] / 1e6, color=COLOR[method],
                    linestyle=STYLES[method], lw=2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(Y_LABELS[METRICS.index(metric)], fontsize=9)
        if sweep == 'hidden_depth':
            ax.set_xticks(sorted(data[data.sweep.eq('hidden_depth')].value.unique()))
        style(ax)
    finish(fig, NAME[dataset] + ' - Computation and memory accesses', out/dataset/'01_computation_and_access_scaling')


def plot_windowing_sweep(dataset, data, out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.9), sharex=True)
    work_span = max(.02, float(data[[m+'_ratio' for m in METRICS]].max().max())-1)
    for ax, column, title in zip(axes,
            ['memory_ratio', 'compute_scalars_ratio', 'access_scalars_ratio'],
            ['Peak memory', 'Total computation', 'Total memory accesses']):
        for method in METHODS:
            d = data[data.learner.eq(method)].sort_values('window_length')
            ax.plot(d.relative_window_length, d[column], color=COLOR[method],
                    linestyle=STYLES[method], lw=2, zorder=3)
        ax.axhline(1, color='#555555', ls='--', lw=1, zorder=1)
        ax.set_xlim(0, 1)
        ax.set_xticks(np.linspace(0, 1, 6))
        if column == 'memory_ratio':
            ax.set_ylim(0, 1.05)
        else:
            # Keep small per-window cost changes visible; do not force zero.
            ax.set_ylim(1-work_span*.12, 1+work_span*1.15)
        ax.set_xlabel('Relative window length L / T')
        ax.set_ylabel('Windowed / full-sequence ratio')
        ax.set_title(title, fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=.2)
        ax.set_axisbelow(True)
    handles = [Line2D([0], [0], color=COLOR[m], linestyle=STYLES[m],
                      lw=2, label=LABEL[m]) for m in METHODS]
    fig.legend(handles=handles, ncol=4, loc='lower center',
               bbox_to_anchor=(.5, .055), frameon=False)
    fig.text(.5, .018, 'Hop = L; all time steps retained, including partial tails. '
             'T = maximum saved full-sequence length. Theoretical costs; accuracy not estimated.',
             ha='center', fontsize=8.5)
    fig.suptitle(NAME[dataset] + ' - Controlled window-length sweep', fontsize=13, y=.99)
    fig.tight_layout(rect=(0, .13, 1, .96))
    path = out/dataset/'02_controlled_windowing_sweep'
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(path.with_suffix('.'+ext), dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_memory(dataset, theory, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for ax, sweep, title, xlabel in zip(axes, ['sequence_length','hidden_depth'],
            ['Sequence length', 'Network depth'], ['Time steps', 'Hidden layers']):
        for method in METHODS:
            d = theory[theory.sweep.eq(sweep) & theory.learner.eq(method)]
            ax.plot(d.value, d.memory_bytes/MIB, color=COLOR[method], linestyle=STYLES[method], lw=2)
        ax.set_title(title, fontsize=10); ax.set_xlabel(xlabel)
        ax.set_ylabel('Peak training memory (MiB)')
        if sweep == 'hidden_depth':
            ax.set_xticks(sorted(theory[theory.sweep.eq(sweep)].value.unique()))
        style(ax)
    finish(fig, NAME[dataset] + ' - Training-memory scaling', out/dataset/'03_training_memory_scaling')


def memory_case_study_data(baselines):
    rows = []
    for dataset, meta in baselines.items():
        if meta['baseline_hidden_layers'] != 1:
            raise ValueError(f'{dataset} baseline is not a one-layer model')
        t, width = meta['full_sequence_steps'], meta['baseline_hidden_width']
        configs = [('baseline', 'Baseline', t, width, 1),
                   ('windowed_s16', 'Windowed', 16, width, 1),
                   ('deep_2_layer', 'Deep', t, width, 2),
                   ('wide_deep_2_layer', 'Wide deep', t, 1024, 2)]
        for key, label, steps, hidden, depth in configs:
            for method in METHODS:
                cost = estimate(method, meta, steps=steps, width=hidden, depth=depth)
                rows.append(dict(dataset=dataset, configuration=key,
                    configuration_label=label, learner=method,
                    time_steps=steps, hidden_width=hidden, hidden_layers=depth,
                    batch_size=meta['batch_size'], input_features=meta['input_features'],
                    classes=meta['classes'], memory_bytes=cost['memory_bytes']))
    return pd.DataFrame(rows)


def plot_memory_case_study(dataset, data, out):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True)
    for ax, (key, group) in zip(axes, data.groupby('configuration', sort=False)):
        d = group.set_index('learner').loc[list(METHODS)]
        ax.bar(np.arange(4), d.memory_bytes/MIB,
               color=[COLOR[m] for m in METHODS], width=.72)
        ax.set_xticks(np.arange(4), [LABEL[m] for m in METHODS], rotation=20, ha='right')
        ax.set_title(d.configuration_label.iloc[0], fontsize=10)
        style(ax)
    axes[0].set_ylabel('Peak training memory (MiB)')
    finish(fig, NAME[dataset] + ' - Memory case study',
           out/dataset/'04_memory_case_study', bar_legend=True)


def gallery(datasets, output):
    html=['<!doctype html><meta charset="utf-8"><title>Results overview</title>',
      '<style>body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:0 20px;color:#222} '
      'img{max-width:100%;height:auto}article{border-top:1px solid #ccc;margin-top:35px} '
      'nav{position:sticky;top:0;background:white;padding:12px}a{margin-right:12px}</style>',
      '<h1>Memory, workload, and cycle-proxy figures</h1>',
      '<p>Run comparisons use the costs and window-length profiles saved with each experiment. '
      'New runs include truncated final windows; historical runs retain their recorded preprocessing.</p>',
      '<p>The controlled window-length sweep applies hop = L to the same saved '
      'full-sequence input profiles for all methods. Its hypothetical settings '
      'have theoretical costs only; accuracy is not estimated.</p>',
      '<p><a href="accuracy_memory_README.md">Accuracy/memory chart notes and hardware sources</a>'
      '<a href="accuracy_memory.csv">Accuracy/memory values</a>'
      '<a href="memory_budget_comparison.csv">Memory-budget comparisons</a>'
      + ('<a href="memory_hierarchy_README.md">Capacity model and coefficients</a>'
         '<a href="fast_memory_capacity_sweep.csv">Capacity-sweep values</a>'
         if (output/'memory_hierarchy_README.md').exists() else '')
      + ('<a href="controlled_windowing_sweep.csv">Controlled window-length values</a>'
         if (output/'controlled_windowing_sweep.csv').exists() else '')
      + ('<a href="windowed_cost_capacity_sweep.csv">Windowed/full cost values</a>'
         if (output/'windowed_cost_capacity_sweep.csv').exists() else '')
      + '</p>',
      '<nav>'+''.join(f'<a href="#{d}">{escape(NAME[d])}</a>' for d in datasets)+'</nav>']
    for dataset in datasets:
        html.append(f'<article id="{dataset}"><h2>{escape(NAME[dataset])}</h2>')
        names = ['01_computation_and_access_scaling', '02_controlled_windowing_sweep',
                 '03_training_memory_scaling', '04_memory_case_study', '05_fast_memory_capacity_sweep',
                 '06_accuracy_memory_budget', '07_windowed_cost_capacity_sweep']
        for name in names:
            if not (output/dataset/(name+'.png')).exists():
                continue
            path=f'{dataset}/{name}'
            html.append(f'<p><a href="{path}.pdf">PDF</a><a href="{path}.svg">SVG</a></p>'
                        f'<img loading="lazy" src="{path}.png" alt="{escape(NAME[dataset]+": "+name)}">')
        html.append('</article>')
    (output/'index.html').write_text('\n'.join(html),encoding='utf-8')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets',nargs='+',choices=DATASETS,default=list(DATASETS))
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--accuracy-memory-only', action='store_true',
                        help='Generate the accuracy/memory charts without replacing other cost charts')
    parser.add_argument('--memory-budget-fraction', type=float, default=1.0,
                        help='Fraction of each published memory capacity available to training (default: 1)')
    args=parser.parse_args(argv)
    if len(set(args.datasets)) != len(args.datasets):
        parser.error('Duplicate dataset IDs')
    if not 0 < args.memory_budget_fraction <= 1:
        parser.error('--memory-budget-fraction must be greater than 0 and at most 1')
    output=args.output_dir.resolve()
    output.mkdir(parents=True,exist_ok=True)
    meta=load_baselines(args.datasets)
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    points = accuracy_memory_data(meta)
    references = memory_reference_data(args.memory_budget_fraction)
    points.to_csv(output/'accuracy_memory.csv', index=False)
    references.to_csv(output/'memory_budget_references.csv', index=False)
    memory_budget_data(points, references).to_csv(output/'memory_budget_comparison.csv', index=False)
    (output/'accuracy_memory_README.md').write_text(
        '# Accuracy within a memory budget\n\n'
        'Each dataset has 06_accuracy_memory_budget.pdf, .svg and .png. '
        'Colors identify learning methods. Open circles show full-sequence runs; '
        'filled diamonds show the validation-selected window/hop runs evaluated on the test set. '
        'Arrows connect those two observed results. The x axis is logarithmic.\n\n'
        'Accuracy, configured batch size, architecture and numerical precision are retained '
        'from saved experiments; no hypothetical networks or smaller batches are substituted. '
        'Peak memory comes from each run summary and is checked against the comparison export. '
        'The plotted data match the thesis main-table quantities. Historical accuracy is not rerun.\n\n'
        'Dashed lines use published hardware memory capacities as reference budgets. '
        'Only nearby lines are shown to keep each chart readable. Device names, original capacity '
        'wording and source URLs are in memory_budget_references.csv. '
        'Capacities are expressed in binary bytes for comparison with MiB. The STM32 reference '
        'is its advertised nominal upper capacity; its memory is banked. ESP32 uses external '
        'PSRAM only; its internal SRAM is not added. The original 64 MB Milk-V Duo is a '
        'historical, discontinued reference.\n\n'
        f'Training budget fraction: {args.memory_budget_fraction:g}. At 1, lines are optimistic '
        'capacity ceilings, with no software/OS overhead reserved. Use '
        '`--memory-budget-fraction 0.75`, for example, to allocate 75% to the modelled working set. '
        'This fraction is an assumption, not a measurement of available memory. '
        'Analytical fit does not establish successful device deployment or measured runtime. '
        'The cost-model scope remains unchanged, including exclusion of optimizer-state memory.\n\n'
        'accuracy_memory.csv contains all plotted points and source run IDs. '
        'memory_budget_comparison.csv checks every point against every reference, '
        'including references outside the displayed range. `fits_modelled_memory` means '
        'memory_bytes <= budget_bytes. Negative remaining_budget_bytes denotes an overrun.\n\n'
        'Run `python -m reporting.create_results_overview` for all seven figure types, or add '
        '`--accuracy-memory-only` to preserve the other existing chart exports.\n\n'
        '## Published specifications\n\n'+''.join(
            f'- [{r.device}]({r.source_url}): {r.published_capacity}.\n'
            for r in references.itertuples()), encoding='utf-8')
    for dataset in args.datasets:
        plot_accuracy_memory(dataset, points[points.dataset.eq(dataset)], references, output)
    if args.accuracy_memory_only:
        gallery(args.datasets, output)
        print(f'Saved accuracy/memory figures for {len(args.datasets)} datasets; '
              f'validated {len(points)} observations; gallery: {output / "index.html"}')
        return
    theory=theoretical_data(meta)
    controlled = controlled_windowing_data(meta)
    controlled.to_csv(output/'controlled_windowing_sweep.csv', index=False)
    selected=windowing_tradeoff_data(selected_data(meta), points)
    sweep = capacity_sweep_data(selected, points)
    sweep.to_csv(output/'fast_memory_capacity_sweep.csv', index=False)
    sweep[sweep.capacity_bytes.eq(REFERENCE_PROFILE.reference_capacity_bytes)].to_csv(
        output/'fast_memory_reference_costs.csv', index=False)
    window_sweep = windowed_capacity_sweep_data(selected, points)
    window_sweep.to_csv(output/'windowed_cost_capacity_sweep.csv', index=False)
    window_sweep[window_sweep.capacity_bytes.eq(REFERENCE_PROFILE.reference_capacity_bytes)].to_csv(
        output/'windowed_cost_reference_costs.csv', index=False)
    from reporting.esp32_cycle_proxy import esp32_cycle_proxy_data
    esp32_cycle_proxy_data(window_sweep[window_sweep.capacity_bytes.eq(REFERENCE_PROFILE.reference_capacity_bytes)],
        REFERENCE_PROFILE.reference_capacity_bytes, REFERENCE_PROFILE.external_capacity_bytes).to_csv(
        output/'esp32_cycle_proxy.csv', index=False)
    write_hierarchy_notes(output)
    memory_cases = memory_case_study_data(meta)
    memory_cases.to_csv(output/'memory_case_study.csv', index=False)
    theory.to_csv(output/'theoretical_scaling.csv',index=False)
    selected.to_csv(output/'selected_vs_full.csv',index=False)
    (output/'configurations.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    for dataset in args.datasets:
        plot_theory(dataset,meta[dataset],theory[theory.dataset.eq(dataset)],output)
        plot_windowing_sweep(dataset, controlled[controlled.dataset.eq(dataset)], output)
        plot_memory(dataset, theory[theory.dataset.eq(dataset)], output)
        plot_memory_case_study(dataset, memory_cases[memory_cases.dataset.eq(dataset)], output)
        plot_capacity_sweep(dataset, sweep[sweep.dataset.eq(dataset)], output)
        plot_windowed_capacity_sweep(dataset,
            window_sweep[window_sweep.dataset.eq(dataset)], output)
        # Remove obsolete report artifacts when regenerating the gallery.
        for ext in ('pdf', 'svg', 'png'):
            (output/dataset/f'05_time_coefficient_sweep.{ext}').unlink(missing_ok=True)
            (output/dataset/f'02_windowing_cost_comparison.{ext}').unlink(missing_ok=True)
            (output/dataset/f'07_windowed_time_capacity_sweep.{ext}').unlink(missing_ok=True)
            (output/dataset/f'08_slow_memory_cost_sensitivity.{ext}').unlink(missing_ok=True)
        print(f'Saved 7 figures for {dataset}',flush=True)
    (output/'time_coefficient_sweep.csv').unlink(missing_ok=True)
    for name in ('windowed_time_capacity_sweep.csv', 'windowed_time_reference_costs.csv',
                 'slow_memory_cost_sensitivity.csv', 'slow_memory_reference_costs.csv'):
        (output/name).unlink(missing_ok=True)
    gallery(args.datasets,output)
    (output/'README.md').write_text("""# Results overview

Open index.html to browse all datasets. Figures are exported as PDF, SVG and PNG.
Run `python -m reporting.create_results_overview`; optional arguments are
`--datasets har pamap2` and `--output-dir PATH`.

1. **Computation and memory-access scaling**: sequence length and network depth.
   Work counts are shown separately; there is no equal-weight time panel.
2. **Controlled window-length sweep**: three panels show peak memory, total
   computation and total memory accesses versus L/T. Every method uses every
   integer window length L from 1 to T, with non-overlapping windows (hop=L).
   T is the maximum saved full-sequence length; variable-length datasets retain
   their complete saved length distribution. All methods must share that input
   distribution. The same absolute L is applied to each sequence within a dataset.
   Short final windows are truncated; no time steps are repeated, omitted or added.
   Work is summed over all windows and averaged over original sequences, using
   the saved batch size. Memory is the largest window's working set, not a sum.
   Each method is normalized by its own validated full-sequence costs, reaching
   1 at L/T=1. Per-window terms already present in the analytical formulas repeat
   for each window; no extra overhead coefficient is added. Work panels use
   linear y axes with a common range for the two work panels. These hypothetical configurations have no
   accuracy estimates. Values, source run IDs, dimensions, length-profile source,
   window counts and time-step coverage are in controlled_windowing_sweep.csv.
   The previous selected-configuration cost bars/scatter are replaced. Actual
   validation-selected comparisons remain in selected_vs_full.csv, chart 06
   (accuracy/memory), and chart 07 (cycle proxy versus capacity).
3. **Training-memory scaling**: peak memory versus sequence length and depth.
4. **Memory case study**: baseline, 16-step window, two baseline-width hidden
   layers, and two 1024-neuron hidden layers. Peak memory counts one window's
   working set, not the sum over all windows. Values and dimensions are in
   memory_case_study.csv.

5. **Fast-memory capacity sweep**: one curve for each learning method, using its
   saved full-sequence work and peak memory. A normalized copy-ratio reference is
   held fixed while fast-memory capacity varies. All methods share one BPTT
   all-slow normalization reference. Markers indicate each working set's capacity.
   The proportional-access assumption and coefficient derivations are documented
   in memory_hierarchy_README.md and memory_hierarchy_profile.json. Values are in
   fast_memory_capacity_sweep.csv, with reference-capacity values in
   fast_memory_reference_costs.csv and coefficients in memory_hierarchy_coefficients.csv.

6. **Accuracy within a memory budget**: observed full-sequence and selected-window
   accuracy versus estimated peak training memory. Arrows connect each method's
   runs; dashed lines mark published hardware memory capacities. Values, feasibility
   comparisons and hardware source links are in accuracy_memory.csv,
   memory_budget_comparison.csv and memory_budget_references.csv. See
   accuracy_memory_README.md for assumptions. Use --accuracy-memory-only to add
   these figures without replacing the other existing cost charts, or
   --memory-budget-fraction 0.75 to reserve 25% of each capacity for other uses.

7. **Windowed/full cycle proxy versus capacity**: selected-window total cost
   divided by the same method's full-sequence cost at each capacity. Below 1 means
   windowing reduces cycle proxy. The denominator is each method's own baseline,
   not chart 05's common BPTT all-slow reference. Each setting uses its own saved
   peak working set and total work over all windows, including overlap and tails.
   Values and both source run IDs are in windowed_cost_capacity_sweep.csv;
   reference-capacity comparisons are in windowed_cost_reference_costs.csv.

The previous common-coefficient sweep and equal-weight time panels are removed.
The capacity model assumes the fast-access fraction equals min(1, capacity /
peak working set). Its coefficients are common to all methods; uniform access
frequency and fixed placement are assumptions. It is an analytical service proxy,
not physical cycles or measured execution time. Alpha=beta_fast=1 is a reference
assumption. Beta_slow=1103614/102415 is a matched scalar-copy slowdown, about 10.78,
not a universal PSRAM penalty. No coefficient-combination grid or overhead term is used.

Fixed dimensions are recorded in configurations.json. Pure theory uses the saved
maximum horizon for variable-length data; run comparisons use observed lengths.
Costs are analytical, not measured runtime or hardware traffic. Run comparisons
use the length profiles and costs recorded with those experiments. New windowed
runs record actual lengths, including shorter final windows. Historical runs retain
their original recorded costs and preprocessing. No training is run by this report.
The plotted values are saved
in theoretical_scaling.csv, controlled_windowing_sweep.csv and selected_vs_full.csv.
""", encoding='utf-8')
    print(f'Validated {len(selected)*2} run profiles; gallery: {output / "index.html"}')


if __name__ == '__main__':
    main()
