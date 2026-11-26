"""Unified thesis figures: memory, workload, and cycle-proxy estimates.

Run this script to generate cost figures from the current experiment
configuration and dataset. It adds accuracy and validation-selected
comparisons wherever the matching experiment records are available.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
import sys
from types import SimpleNamespace

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

from experiment_config import RUNS
from learners.backprop import BackpropLearner
from learners.base import cost_dimensions
from learners.eprop import EpropLearner
from learners.forward_forward import FFLearner
from learners.pepita import PepitaLearner
from reporting.configured_costs import CostCatalog, report_dataset, window_length_counts
from timeseries.transforms import TAIL_RULE

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
MEMORY_BUDGET_FRACTION = 1.0
OBSERVED_FIGURES = ('06_accuracy_memory_budget', '07_windowed_cost_capacity_sweep')
OBSERVED_EXPORTS = ('accuracy_memory.csv', 'accuracy_memory_README.md', 'memory_budget_references.csv',
                    'memory_budget_comparison.csv', 'selected_vs_full.csv', 'windowed_cost_capacity_sweep.csv',
                    'windowed_cost_reference_costs.csv', 'esp32_cycle_proxy.csv')
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
                'Fast fraction = min(1, fast capacity / peak working set)',
                'Logical accesses are uniform over a fixed working set',
                'Internal SRAM budget is not hardware data-cache capacity; cache reuse is not simulated',
                'Fixed service coefficients while capacity varies parametrically',
                'Arithmetic model units and logical scalar accesses; additive service, no overlap or overhead term',
                'No additional staging-transfer costs are added to this two-tier proxy',
                'The training cost formulas exclude optimizer-state memory',
                'Each setting uses mean total work and peak footprint; no per-length reweighting',
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


def accuracy_memory_data(datasets, catalog, observations):
    """Pair measured accuracy with a fresh cost estimate at the tested settings."""
    base, final, _, summaries = observations
    final = final[final.experiment.eq('hop') & final.dataset.isin(datasets)]
    rows = []
    for selected in final.itertuples():
        baseline = base[base.dataset.eq(selected.dataset) & base.learner.eq(selected.learner)]
        if len(baseline) != 1:
            raise ValueError(f'Missing baseline for {selected.dataset}/{selected.learner}')
        baseline = next(baseline.itertuples())
        for setting, run, accuracy in [('full_sequence', baseline, baseline.final_sample_acc),
                                       ('windowed', selected, selected.test_sample_acc)]:
            summary = summaries[run.run_id]
            config = summary['config']
            length = int(selected.win_L) if setting == 'windowed' else None
            hop = int(selected.hop) if length is not None else None
            source = catalog.profile(selected.dataset, config)
            theory = catalog.evaluate(selected.dataset, selected.learner, config, length, hop)
            memory = theory['memory']['total_bytes']
            if not np.isfinite(memory) or memory <= 0 or not np.isfinite(accuracy) or not 0 <= accuracy <= 100:
                raise ValueError(f'Invalid accuracy or peak memory in {run.run_id}')
            rows.append(dict(dataset=selected.dataset, learner=selected.learner,
                setting=setting, run_id=run.run_id, test_accuracy_pct=accuracy,
                memory_bytes=int(memory), memory_mib=memory/MIB,
                batch_size=config['BATCH_SIZE'], hidden_width=config['HIDDEN_SIZES'][0],
                hidden_layers=len(config['HIDDEN_SIZES']), hidden_sizes=config['HIDDEN_SIZES'],
                fp_bytes=theory['fp_bytes'], window_length=length, hop=hop,
                compute_scalars=theory['compute']['total_scalars'],
                access_scalars=theory['access']['total_scalars'],
                memory_source=theory['profile_source'],
                input_features=source.metadata['input_dim'],
                observed_input_features=summary['meta'].get('input_dim'),
                train_split_sha256=source.metadata['split_indices_sha256']['train'],
                observed_train_split_sha256=(summary['meta'].get('split_indices_sha256') or {}).get('train'),
                window_tail_rule=TAIL_RULE if length is not None else None,
                observed_window_tail_rule=summary['meta'].get('window_tail_rule')))
    data = pd.DataFrame(rows)
    if len(data) != len(datasets)*8 or data.duplicated(['dataset', 'learner', 'setting']).any():
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
    windows = points[points.setting.eq('windowed')]
    if windows.observed_window_tail_rule.ne(windows.window_tail_rule).any():
        fig.text(.5, .015, 'Some accuracy observations used an earlier window policy. '
                 'Memory uses the current policy.', ha='center', fontsize=8)
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

def load(datasets=DATASETS, *, include_trials=True):
    base = read_csv(ROOT / "comparisons/all_current_runs/leaderboard.csv",
                    {"dataset", "learner", "run_id", "final_sample_acc"})
    final = read_csv(ROOT / "comparisons/window_hop_independent/final_test_results.csv",
                     {"dataset", "learner", "run_id", "experiment", "status", "win_L",
                      "hop", "test_sample_acc", "validation_score"})
    base = base[base.dataset.isin(datasets) & base.learner.isin(METHODS)].copy()
    final = final[final.dataset.isin(datasets) & final.learner.isin(METHODS) &
                  final.status.eq("ok")].copy()
    trials = None
    if include_trials:
        trials = read_csv(ROOT / "comparisons/window_hop_independent/all_trials.csv",
                          {"dataset", "learner", "sample_acc"})
        trials = trials[trials.dataset.isin(datasets) & trials.learner.isin(METHODS) &
                        trials.sample_acc.notna()].copy()
    else:
        final = final[final.experiment.eq("hop")]
    expected = len(datasets) * len(METHODS)
    if base.duplicated(["dataset", "learner"]).any() or len(base) != expected:
        raise ValueError("No-window results must have one complete dataset/method matrix")
    if final.duplicated(["dataset", "learner", "experiment"]).any() or any(
            final.experiment.eq(kind).sum() != expected
            for kind in (("window", "hop") if include_trials else ("hop",))):
        raise ValueError("Requested final-test experiment matrices must be complete")
    summaries = {}
    for run_id in set(base.run_id) | set(final.run_id):
        path = ROOT / "runs" / run_id / "summary.json"
        if not path.exists():
            raise FileNotFoundError(f"Run summary is missing: {path}")
        with path.open(encoding="utf-8") as stream:
            summaries[run_id] = json.load(stream)
    return base, final, trials, summaries


def load_available_observations(datasets):
    """Read measured accuracy and the selected schedule directly from the run records."""
    baseline_ids = {
        (report_dataset(config['RUN_ID'], config['DATASET']), config['LEARNER']): config['RUN_ID']
        for config in RUNS
    }
    selections = ROOT / 'optuna/final_tests/independent'
    baselines, selected, summaries = [], [], {}
    for dataset in datasets:
        dataset_baselines, dataset_selected, dataset_summaries = [], [], {}
        for method in METHODS:
            baseline_id = baseline_ids.get((dataset, method))
            if baseline_id is None:
                break
            try:
                manifest = json.loads((selections/f'{dataset}-{method}-hop.json').read_text(encoding='utf-8'))
                selected_id = manifest['run_id']
                baseline, windowed = (
                    json.loads((ROOT/'runs'/run_id/'summary.json').read_text(encoding='utf-8'))
                    for run_id in (baseline_id, selected_id)
                )
            except FileNotFoundError:
                break
            baseline_accuracy = baseline.get('final', {}).get('sample_acc')
            test_accuracy = windowed.get('final', {}).get('sample_acc')
            if (baseline.get('status') != 'ok' or windowed.get('status') != 'ok'
                    or baseline_accuracy is None or test_accuracy is None):
                break
            schedule = manifest['window']
            dataset_baselines.append(dict(dataset=dataset, learner=method, run_id=baseline_id,
                                          final_sample_acc=baseline_accuracy))
            dataset_selected.append(dict(dataset=dataset, learner=method, run_id=selected_id,
                                         experiment='hop', win_L=schedule['L'], hop=schedule['hop'],
                                         test_sample_acc=test_accuracy,
                                         validation_score=manifest['validation_score']))
            dataset_summaries.update({baseline_id: baseline, selected_id: windowed})
        if len(dataset_baselines) == len(METHODS):
            baselines.extend(dataset_baselines)
            selected.extend(dataset_selected)
            summaries.update(dataset_summaries)
    if baselines:
        return pd.DataFrame(baselines), pd.DataFrame(selected), None, summaries
    return None


def model_cost(method: str, *, batch: int = 128, steps: int = 128,
               input_dim: int = 9, classes: int = 6, width: int = 128,
               depth: int = 1) -> dict:
    effective_inputs = input_dim + classes if method == "ff" else input_dim
    dims = cost_dimensions(input_dim, classes, [width] * depth, effective_inputs, method != "ff")
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
    """Read archived settings for the local historical-report migration."""
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


def theoretical_data(profiles):
    rows = []
    for dataset, profile in profiles.items():
        sequences = sorted(set(range(8, 1025, 16)) | {1024, profile.full_sequence_steps})
        sizes = profile.hidden_sizes
        max_depth = max(8, len(sizes))
        depths = range(1, max_depth + 1)
        for sweep, values in [('sequence_length', sequences), ('hidden_depth', depths)]:
            for value in values:
                steps = value if sweep == 'sequence_length' else profile.full_sequence_steps
                hidden = sizes if sweep == 'sequence_length' else resize_depth(sizes, value)
                for method in METHODS:
                    rows.append(dict(dataset=dataset, sweep=sweep, value=value,
                        learner=method, steps=steps, width=hidden[0], depth=len(hidden),
                        **profile.dimensions(hidden), **profile.estimate(method, steps, hidden)))
    return pd.DataFrame(rows)


def resize_depth(sizes, depth):
    """Keep configured widths and repeat the last width when adding layers."""
    return sizes[:depth] + (sizes[-1],) * max(0, depth - len(sizes))


def nonoverlapping_window_profile(length_counts, window_length):
    """Partition every sequence once, keeping its shorter final window."""
    if window_length < 1 or int(window_length) != window_length:
        raise ValueError('Window length must be a positive integer')
    lengths = Counter()
    for steps, count in length_counts.items():
        steps = int(steps)
        if steps < 1 or count < 1 or int(count) != count:
            raise ValueError('Sequence lengths and counts must be positive integers')
        lengths[steps] += int(count)
    if not lengths:
        raise ValueError('Window sweep requires a nonempty source-length profile')
    return window_length_counts(lengths, window_length, window_length)


def summarize_profile(costs, counts, samples):
    return dict(memory_bytes=max(costs[t]['memory_bytes'] for t in counts),
                **{metric: sum(costs[t][metric]*count for t, count in counts.items())/samples
                   for metric in METRICS})


def controlled_windowing_data(profiles):
    """Sweep every integer window length with hop equal to length."""
    rows = []
    for dataset, source in profiles.items():
        lengths = source.length_counts
        samples = source.original_samples
        horizon = source.full_sequence_steps
        source_steps = sum(t*n for t, n in lengths.items())
        window_profiles = {length: nonoverlapping_window_profile(lengths, length)
                           for length in range(1, horizon+1)}
        for method in METHODS:
            costs = {t: source.estimate(method, t) for t in range(1, horizon+1)}
            full = summarize_profile(costs, lengths, samples)
            for length, profile in window_profiles.items():
                windowed = summarize_profile(costs, profile, samples)
                rows.append(dict(dataset=dataset, learner=method,
                    **source.dimensions(), hidden_width=source.hidden_sizes[0],
                    hidden_layers=len(source.hidden_sizes),
                    full_sequence_steps=horizon, window_length=length, hop=length,
                    relative_window_length=length/horizon, effective_hop_ratio=1.0,
                    window_tail_rule=TAIL_RULE, source_profile='current_training_split',
                    original_samples=samples, distinct_source_lengths=len(lengths),
                    windows_per_sample=sum(profile.values())/samples,
                    source_steps_per_sample=source_steps/samples,
                    processed_steps_per_sample=sum(t*n for t, n in profile.items())/samples,
                    peak_window_steps=max(profile, key=lambda t: (costs[t]['memory_bytes'], t)),
                    memory_ratio=windowed['memory_bytes']/full['memory_bytes'],
                    **{f'baseline_{key}': value for key, value in full.items()},
                    **{f'windowed_{key}': value for key, value in windowed.items()},
                    **{f'{m}_ratio': windowed[m]/full[m] for m in METRICS}))
    return pd.DataFrame(rows)


def checked_run_cost(row, summary, meta):
    """Validate a saved snapshot for the local historical-report migration."""
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


def selected_data(points, final):
    keys = ['dataset', 'learner']
    columns = ['run_id', 'test_accuracy_pct', *METRICS]
    baseline = points[points.setting.eq('full_sequence')][keys + columns]
    windowed = points[points.setting.eq('windowed')][keys + columns + ['window_length', 'hop']]
    data = baseline.rename(columns={key: f'baseline_{key}' for key in columns}).merge(
        windowed.rename(columns={key: f'selected_{key}' for key in columns}),
        on=keys, validate='one_to_one')
    data = data.rename(columns={f'{prefix}_test_accuracy_pct': f'{prefix}_test_accuracy'
                                for prefix in ('baseline', 'selected')})
    scores = final[final.experiment.eq('hop')].set_index('run_id').validation_score
    data['selected_validation_accuracy'] = data.selected_run_id.map(scores)
    data['effective_hop_ratio'] = data.hop/data.window_length
    for metric in METRICS:
        data[f'{metric}_ratio'] = data[f'selected_{metric}']/data[f'baseline_{metric}']
    return data


def _capacity_runs(selected, points, setting):
    """Match each observation to the costs evaluated for its configuration."""
    keys = ['dataset', 'learner']
    runs = points[points.setting.eq(setting)].copy()
    if (runs.duplicated(keys).any() or selected.duplicated(keys).any()
            or set(map(tuple, runs[keys].to_numpy()))
            != set(map(tuple, selected[keys].to_numpy()))):
        raise ValueError(f'Capacity sweep requires matching unique {setting} runs')
    data = selected.merge(runs[keys+['run_id', 'memory_bytes', 'batch_size', 'fp_bytes']],
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
    """Pair total-work and peak-memory ratios for the same tested settings."""
    keys = ['dataset', 'learner']
    data = selected.copy()
    for setting, prefix in [('full_sequence', 'baseline'), ('windowed', 'selected')]:
        runs = _capacity_runs(selected, points, setting)
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


def configured_capacity_inputs(profiles, catalog):
    rows = []
    for dataset in profiles:
        for method in METHODS:
            full = catalog.evaluate(dataset, method, None)
            rows.append(dict(dataset=dataset, learner=method, batch_size=full['batch_size'],
                fp_bytes=full['fp_bytes'], memory_bytes=full['memory']['total_bytes'],
                baseline_compute_scalars=full['compute']['total_scalars'],
                baseline_access_scalars=full['access']['total_scalars']))
    return pd.DataFrame(rows)


def capacity_sweep_data(selected, points=None, profile=REFERENCE_PROFILE):
    """Use full-sequence mean work and peak memory.

    One common BPTT all-slow denominator per dataset preserves differences
    between methods. Each footprint and the reference capacity are exact grid
    points, so transitions and the physical reference can be reproduced.
    """
    data = selected if points is None else _capacity_runs(selected, points, 'full_sequence')
    rows = []
    for dataset, group in data.groupby('dataset', sort=False):
        capacities = _capacity_grid(group.memory_bytes, profile)
        bp = group[group.learner.eq('bp')].iloc[0]
        reference = float(hierarchy_cost(bp.baseline_compute_scalars,
            bp.baseline_access_scalars, bp.memory_bytes, 0, profile)['estimated_cost'])
        if reference <= 0:
            raise ValueError(f'BPTT normalization must be positive in {dataset}')
        for run in group.itertuples():
            provenance = {} if points is None else dict(
                run_id=run.run_id, test_accuracy_pct=run.baseline_test_accuracy,
                normalization_reference_run_id=bp.run_id)
            values = hierarchy_cost(run.baseline_compute_scalars,
                run.baseline_access_scalars, run.memory_bytes, capacities, profile)
            for i, capacity in enumerate(capacities):
                rows.append(dict(dataset=dataset, learner=run.learner,
                    setting='full_sequence', batch_size=run.batch_size, fp_bytes=run.fp_bytes,
                    memory_bytes=run.memory_bytes,
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
                    fits_nominal_fast_memory=run.memory_bytes <= profile.reference_capacity_bytes,
                    fits_nominal_total_memory=run.memory_bytes <= profile.reference_capacity_bytes+profile.external_capacity_bytes,
                    normalization_reference_cost_units=reference, **provenance))
    return pd.DataFrame(rows)


def windowed_capacity_sweep_data(selected, points, profile=REFERENCE_PROFILE):
    """Compare validation-selected windows with the same method's full sequence.

    Each setting uses its peak working set for the proportional split.
    """
    keys = ['dataset', 'learner']
    full = _capacity_runs(selected, points, 'full_sequence')
    windowed = _capacity_runs(selected, points, 'windowed')
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
                raise ValueError(f'Full-sequence denominator must be positive for {dataset}/{run.learner}')
            columns = dict(dataset=dataset, learner=run.learner,
                baseline_run_id=run.baseline_run_id, selected_run_id=run.selected_run_id,
                baseline_test_accuracy=run.baseline_test_accuracy, selected_test_accuracy=run.selected_test_accuracy,
                batch_size=run.batch_size, fp_bytes=run.fp_bytes, window_length=run.window_length, hop=run.hop,
                effective_hop_ratio=run.effective_hop_ratio,
                window_source='validation_selected',
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


def esp32_cycle_proxy_data(window_reference, fast_capacity=512*1024,
                           slow_capacity=8*1024**2):
    """Export reference cycle proxies and nominal memory-fit flags."""
    rows=[]
    for r in window_reference.itertuples():
        for setting,prefix in [('full_sequence','baseline'),('windowed','selected')]:
            memory=getattr(r,prefix+'_memory_bytes')
            rows.append(dict(dataset=r.dataset,learner=r.learner,setting=setting,
                run_id=getattr(r,prefix+'_run_id'),batch_size=r.batch_size,
                fp_bytes=r.fp_bytes, window_source=r.window_source,
                window_length=r.window_length if setting == 'windowed' else None,
                hop=r.hop if setting == 'windowed' else None,
                compute_scalars=getattr(r,prefix+'_compute_scalars'),
                access_scalars=getattr(r,prefix+'_access_scalars'),
                memory_bytes=memory,fast_capacity_bytes=fast_capacity,
                total_capacity_bytes=fast_capacity+slow_capacity,
                fast_fraction=getattr(r,prefix+'_fast_fraction'),
                effective_beta=getattr(r,prefix+'_effective_beta'),
                cycle_proxy=getattr(r,prefix+'_estimated_cost_units'),
                cycle_proxy_millions=getattr(r,prefix+'_estimated_cost_units')/1e6,
                fits_fast_memory=memory<=fast_capacity,
                fits_total_memory=memory<=fast_capacity+slow_capacity))
    return pd.DataFrame(rows)


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
    ax.set_ylabel('Cycle-proxy ratio\n(windows / own full sequence)')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=.2)
    ax.set_axisbelow(True)
    fig.suptitle(NAME[dataset] + ' - Windowed/full cycle proxy vs fast-memory capacity',
                 fontsize=12, y=.98)
    fig.legend(handles=[Line2D([0], [0], color=COLOR[m], linestyle=STYLES[m],
                   lw=2, label=LABEL[m]) for m in METHODS],
               ncol=4, loc='lower center', bbox_to_anchor=(.5, .047), frameon=False)
    fig.text(.5, .015, 'Window and hop were selected by validation accuracy. Below 1 indicates fewer proxy cycles. '
             f'alpha = beta_fast = 1, beta_slow = {profile.beta_slow:.2f}.',
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

Total work includes every window produced by the configured schedule, including
overlap and partial tails.
Work is averaged over original sequences at the configured batch size, never
multiplied again by window count. Each setting uses its own peak working set;
window memories are not summed. Peak memory weights the mean work under
the whole-working-set approximation, without per-length beta reweighting.

## Figures and normalization

- 05_fast_memory_capacity_sweep: full-sequence costs versus log capacity, divided
  by one constant per dataset: full BPTT all-slow cost at the reference ratio.
  Open circles mark each footprint. The grid includes all footprints and 512 KiB.
  A labeled vertical line marks the nominal SRAM reference in both capacity charts.
  Values beyond nominal SRAM are
  hypothetical capacity scenarios, not adjustable SRAM on an actual ESP32-S3.
- 07_windowed_cost_capacity_sweep: windowed cost divided by that method's
  own full-sequence cost at the same capacity. Below 1 means lower cycle proxy.
  Different methods have different denominators; these ratios do not rank their
  absolute costs. All eight footprints occur exactly on the grid.

Corresponding CSVs retain raw C/M counts, cycle proxies in reference units and
footprints. Run IDs identify validation-selected settings when used.
Reference CSVs evaluate exactly 512 KiB. The memory-copy coefficients are
FP32-derived reference weights per scalar access, including when a hypothetical
configuration uses a different precision.
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


def plot_theory(dataset, data, out):
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
    work = data[[m+'_ratio' for m in METRICS]]
    work_range = max(.02, float(work.max().max()) - 1, 1 - float(work.min().min()))
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
            ax.set_ylim(0, max(1.05, data.memory_ratio.max()*1.05))
        else:
            # Keep small per-window cost changes visible; do not force zero.
            ax.set_ylim(min(1-work_range*.12, work.min().min()-work_range*.05),
                        max(1+work_range*.12, work.max().max()+work_range*.15))
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
             'T is the maximum preprocessed training sequence length.',
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


def memory_case_study_data(profiles):
    rows = []
    for dataset, source in profiles.items():
        t, sizes = source.full_sequence_steps, source.hidden_sizes
        deep = resize_depth(sizes, max(2, len(sizes) + 1))
        case_steps = min(16, t)
        configs = [('baseline', 'Baseline', t, sizes),
                   ('windowed_s16', f'{case_steps}-step window', case_steps, sizes),
                   (f'deep_{len(deep)}_layer', 'Deep', t, deep),
                   (f'wide_deep_{len(deep)}_layer', 'Wide deep', t, (1024,) * len(deep))]
        for key, label, steps, hidden in configs:
            for method in METHODS:
                cost = source.estimate(method, steps, hidden)
                rows.append(dict(dataset=dataset, configuration=key,
                    configuration_label=label, learner=method,
                    time_steps=steps, hidden_width=hidden[0], hidden_layers=len(hidden),
                    **source.dimensions(hidden), memory_bytes=cost['memory_bytes']))
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


def gallery(datasets, output, *, observed_datasets=()):
    html=['<!doctype html><meta charset="utf-8"><title>Results overview</title>',
      '<style>body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:0 20px;color:#222} '
      'img{max-width:100%;height:auto}article{border-top:1px solid #ccc;margin-top:35px} '
      'nav{position:sticky;top:0;background:white;padding:12px}a{margin-right:12px}</style>',
      '<h1>Memory, workload, and cycle-proxy figures</h1>',
      '<p>Costs are recalculated from the current learner formulas and training data.</p>',
      '<p>' + ('<a href="accuracy_memory_README.md">Accuracy/memory chart notes and hardware sources</a>'
      '<a href="accuracy_memory.csv">Accuracy/memory values</a>'
      '<a href="memory_budget_comparison.csv">Memory-budget comparisons</a>' if observed_datasets else '')
      + ('<a href="memory_hierarchy_README.md">Capacity model and coefficients</a>'
         '<a href="fast_memory_capacity_sweep.csv">Capacity-sweep values</a>'
         if (output/'memory_hierarchy_README.md').exists() else '')
      + ('<a href="controlled_windowing_sweep.csv">Controlled window-length values</a>'
         if (output/'controlled_windowing_sweep.csv').exists() else '')
      + ('<a href="windowed_cost_capacity_sweep.csv">Windowed/full cost values</a>'
         if observed_datasets else '')
      + '</p>',
      '<nav>'+''.join(f'<a href="#{d}">{escape(NAME[d])}</a>' for d in datasets)+'</nav>']
    for dataset in datasets:
        html.append(f'<article id="{dataset}"><h2>{escape(NAME[dataset])}</h2>')
        names = ['01_computation_and_access_scaling', '02_controlled_windowing_sweep',
                 '03_training_memory_scaling', '04_memory_case_study', '05_fast_memory_capacity_sweep']
        if dataset in observed_datasets:
            names.extend(OBSERVED_FIGURES)
        for name in names:
            if not (output/dataset/(name+'.png')).exists():
                continue
            path=f'{dataset}/{name}'
            html.append(f'<p><a href="{path}.pdf">PDF</a><a href="{path}.svg">SVG</a></p>'
                        f'<img loading="lazy" src="{path}.png" alt="{escape(NAME[dataset]+": "+name)}">')
        html.append('</article>')
    (output/'index.html').write_text('\n'.join(html),encoding='utf-8')


def write_accuracy_report(datasets, points, output, fraction):
    references = memory_reference_data(fraction)
    points.to_csv(output/'accuracy_memory.csv', index=False)
    references.to_csv(output/'memory_budget_references.csv', index=False)
    memory_budget_data(points, references).to_csv(output/'memory_budget_comparison.csv', index=False)
    (output/'accuracy_memory_README.md').write_text(
        '# Accuracy within a memory budget\n\n'
        'Each dataset has 06_accuracy_memory_budget.pdf, .svg and .png. '
        'Colors identify learning methods. Open circles show full-sequence runs; '
        'filled diamonds show the validation-selected window/hop runs evaluated on the test set. '
        'Arrows connect those two observed results. The x axis is logarithmic.\n\n'
        'Accuracy, batch size, hidden dimensions and numerical precision come from the original experiments. '
        'Peak memory is recalculated from the current learner formulas and training split. '
        'CSV fields record the window rules used for the costs and for the measured accuracy.\n\n'
        'Dashed lines use published hardware memory capacities as reference budgets. '
        'Only nearby lines are shown to keep each chart readable. Device names, original capacity '
        'wording and source URLs are in memory_budget_references.csv. '
        'Capacities are expressed in binary bytes for comparison with MiB. The STM32 reference '
        'is its advertised nominal upper capacity; its memory is banked. ESP32 uses external '
        'PSRAM only; its internal SRAM is not added. The original 64 MB Milk-V Duo is a '
        'historical, discontinued reference.\n\n'
        f'Training budget fraction: {fraction:g}. At 1, lines are optimistic '
        'capacity ceilings, with no software/OS overhead reserved. Set '
        '`MEMORY_BUDGET_FRACTION` in the report script to choose another fraction. '
        'This fraction is an assumption, not a measurement of available memory. '
        'Analytical fit does not establish successful device deployment or measured runtime. '
        'The cost-model scope remains unchanged, including exclusion of optimizer-state memory.\n\n'
        'accuracy_memory.csv contains all plotted points and source run IDs. '
        'memory_budget_comparison.csv checks every point against every reference, '
        'including references outside the displayed range. `fits_modelled_memory` means '
        'memory_bytes <= budget_bytes. Negative remaining_budget_bytes denotes an overrun.\n\n'
        'Run `python reporting/create_results_overview.py` to regenerate the report.\n\n'
        '## Published specifications\n\n'+''.join(
            f'- [{r.device}]({r.source_url}): {r.published_capacity}.\n'
            for r in references.itertuples()), encoding='utf-8')
    for dataset in datasets:
        plot_accuracy_memory(dataset, points[points.dataset.eq(dataset)], references, output)


def write_cost_report(profiles, catalog, output):
    theory = theoretical_data(profiles)
    controlled = controlled_windowing_data(profiles)
    memory_cases = memory_case_study_data(profiles)
    sweep = capacity_sweep_data(configured_capacity_inputs(profiles, catalog))
    capacity = REFERENCE_PROFILE.reference_capacity_bytes

    theory.to_csv(output/'theoretical_scaling.csv', index=False)
    controlled.to_csv(output/'controlled_windowing_sweep.csv', index=False)
    memory_cases.to_csv(output/'memory_case_study.csv', index=False)
    sweep.to_csv(output/'fast_memory_capacity_sweep.csv', index=False)
    sweep[sweep.capacity_bytes.eq(capacity)].to_csv(output/'fast_memory_reference_costs.csv', index=False)
    configs = {dataset: p.export_metadata() for dataset, p in profiles.items()}
    (output/'configurations.json').write_text(json.dumps(configs, indent=2), encoding='utf-8')
    write_hierarchy_notes(output)

    for dataset in profiles:
        plot_theory(dataset, theory[theory.dataset.eq(dataset)], output)
        plot_windowing_sweep(dataset, controlled[controlled.dataset.eq(dataset)], output)
        plot_memory(dataset, theory[theory.dataset.eq(dataset)], output)
        plot_memory_case_study(dataset, memory_cases[memory_cases.dataset.eq(dataset)], output)
        plot_capacity_sweep(dataset, sweep[sweep.dataset.eq(dataset)], output)
        print(f'Saved cost figures for {dataset}', flush=True)

    (output/'README.md').write_text("""# Results overview

Open index.html to browse the figures. Each chart is available as PDF, SVG and PNG.

Figures 01–05 use the current settings from experiment_config.py and the current
learner cost formulas. configurations.json records dimensions, split hashes and
the full distribution of preprocessed training sequence lengths. Depth sweeps
truncate the configured layer widths or repeat the last width for added layers.

The controlled window-length sweep evaluates every integer length L from 1 to
the longest training sequence, using hop=L. Work is summed across all actual
window lengths and averaged over original recordings. Peak memory is the largest
per-window working set. A setting with one full window per recording has a ratio
of 1 for every cost.

The memory case study includes the configured model, up to 16 steps per
window, an additional hidden layer, and a model of the same increased depth
with 1024 units per layer.

Figures 06 and 07 use the tested baseline and validation-selected window length
and hop for each learner. These figures are included where the corresponding
experiments are available. Costs at those settings use the current formulas.
Figures 05 and 07 use the cycle proxy documented in memory_hierarchy_README.md.
The CSV files contain the plotted values and reference-capacity estimates.
""", encoding='utf-8')


def write_observed_report(datasets, catalog, observations, output):
    points = accuracy_memory_data(datasets, catalog, observations)
    selected = windowing_tradeoff_data(selected_data(points, observations[1]), points)
    sweep = windowed_capacity_sweep_data(selected, points)
    capacity = REFERENCE_PROFILE.reference_capacity_bytes
    reference = sweep[sweep.capacity_bytes.eq(capacity)]
    write_accuracy_report(datasets, points, output, MEMORY_BUDGET_FRACTION)
    selected.to_csv(output/'selected_vs_full.csv', index=False)
    sweep.to_csv(output/'windowed_cost_capacity_sweep.csv', index=False)
    reference.to_csv(output/'windowed_cost_reference_costs.csv', index=False)
    esp32_cycle_proxy_data(reference, capacity, REFERENCE_PROFILE.external_capacity_bytes).to_csv(
        output/'esp32_cycle_proxy.csv', index=False)
    for dataset in datasets:
        plot_windowed_capacity_sweep(dataset, sweep[sweep.dataset.eq(dataset)], output)


def remove_unavailable_observed_exports(output, datasets, observed):
    """Clear charts and tables generated by an earlier overview."""
    for dataset in set(datasets) - set(observed):
        for figure in OBSERVED_FIGURES:
            for extension in ('pdf', 'svg', 'png'):
                (output/dataset/f'{figure}.{extension}').unlink(missing_ok=True)
    if not observed:
        for filename in OBSERVED_EXPORTS:
            (output/filename).unlink(missing_ok=True)


def main():
    output = DEFAULT_OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})

    catalog = CostCatalog()
    profiles = {dataset: catalog.profile(dataset) for dataset in DATASETS}
    write_cost_report(profiles, catalog, output)

    observed = []
    try:
        observations = load_available_observations(DATASETS)
        if observations is not None:
            available = [dataset for dataset in DATASETS if dataset in set(observations[0].dataset)]
            write_observed_report(available, catalog, observations, output)
            observed = available
        remove_unavailable_observed_exports(output, DATASETS, observed)
    finally:
        gallery(DATASETS, output, observed_datasets=observed)
    missing = [dataset for dataset in DATASETS if dataset not in observed]
    if missing:
        print('Skipped accuracy and selected-window capacity figures for '
              + ', '.join(missing) + ' because matching saved experiments are incomplete.')
    print(f'Gallery: {output / "index.html"}')


if __name__ == '__main__':
    main()
