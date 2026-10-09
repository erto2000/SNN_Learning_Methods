"""Unified thesis figures: memory, computation, and memory accesses.

Run ``python -m reporting.create_results_overview`` for every dataset.
Uses the existing learner cost models and saved runs; never trains a network
or edits the thesis. Each figure is exported as PDF, SVG, and PNG.
"""
from __future__ import annotations

import argparse
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
PLOT_METRICS = (*METRICS, 'time_proxy')
TITLES = ('Computation', 'Memory accesses', 'Time (cost proxy)')
Y_LABELS = ('Arithmetic work (million model units)', 'Scalar accesses (millions)', 'Time cost (million units)')
STYLES = dict(zip(METHODS, ['-', '--', '-.', ':']))
DEFAULT_OUTPUT = PROJECT_ROOT / 'results/results_overview'

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
        lengths = summary["memory"]["theory"]["instance_length_counts"]
        batch, width, depth = int(source.batch), int(source.hidden), int(source.hidden_depth)
        full_steps = int(meta["time_steps"])
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


def add_time_proxy(theory, selected):
    theory, selected = theory.copy(), selected.copy()
    theory['time_proxy'] = theory.compute_scalars + theory.access_scalars
    for setting in ['baseline', 'selected']:
        selected[setting+'_time_proxy'] = (selected[setting+'_compute_scalars']
                                           + selected[setting+'_access_scalars'])
    selected['time_proxy_ratio'] = selected.selected_time_proxy / selected.baseline_time_proxy
    return theory, selected


def coefficient_sweep_data(selected):
    """Fixed coefficient sum of two; the midpoint matches weights (1, 1)."""
    rows = []
    for r in selected.itertuples():
        for alpha in np.linspace(0, 2, 101):
            beta = 2-alpha
            rows.append(dict(dataset=r.dataset, learner=r.learner,
                compute_weight=alpha, access_weight=beta,
                baseline_time_proxy=alpha*r.baseline_compute_scalars + beta*r.baseline_access_scalars))
    return pd.DataFrame(rows)


def plot_coefficient_sweep(dataset, data, out):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for method in METHODS:
        d = data[data.learner.eq(method)]
        ax.plot(d.compute_weight, d.baseline_time_proxy/1e6,
                color=COLOR[method], linestyle=STYLES[method], lw=2)
    ax.axvline(1, color='#777777', linestyle='--', lw=1)
    ax.set_xlabel('Computation weight (memory-access weight = 2 - x)')
    ax.set_ylabel('Full-sequence time cost (million units)')
    ax.set_xticks([0,.5,1,1.5,2]); ax.set_xlim(0,2)
    ax.spines[['top','right']].set_visible(False); ax.grid(alpha=.2)
    ax.set_ylim(bottom=0)
    finish(fig, NAME[dataset] + ' - Full-sequence time coefficient sweep',
           out/dataset/'05_time_coefficient_sweep')


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
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.4))
    panels = [(sweep, metric, xlabel, prefix + ' - ' + title)
              for sweep, xlabel, prefix in [('sequence_length', 'Time steps', 'Sequence length'),
                                            ('hidden_depth', 'Hidden layers', 'Network depth')]
              for metric, title in zip(PLOT_METRICS, TITLES)]
    for ax, (sweep, metric, xlabel, title) in zip(axes.flat, panels):
        for method in METHODS:
            d = data[data.sweep.eq(sweep) & data.learner.eq(method)]
            ax.plot(d.value, d[metric] / 1e6, color=COLOR[method],
                    linestyle=STYLES[method], lw=2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(Y_LABELS[PLOT_METRICS.index(metric)], fontsize=9)
        if sweep == 'hidden_depth':
            ax.set_xticks(sorted(data[data.sweep.eq('hidden_depth')].value.unique()))
        style(ax)
    finish(fig, NAME[dataset] + ' - Computation, memory accesses and time', out/dataset/'01_computation_and_access_scaling')


def plot_selected(dataset, data, limit, out):
    d=data.set_index('learner').loc[list(METHODS)]
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),sharex=True,sharey=True)
    y=np.arange(4)
    for ax,metric,title in zip(axes,PLOT_METRICS,TITLES):
        vals=d[metric+'_ratio'].to_numpy()
        ax.barh(y,vals,color=[COLOR[m] for m in METHODS],height=.6)
        ax.axvline(1,color='#333333',ls='--',lw=1)
        for yy,v in zip(y,vals):
            ax.text(v+limit*.015,yy,f'{v:.3f}x',va='center',fontsize=9, zorder=4,
                    bbox=dict(facecolor='white', edgecolor='none', pad=.4))
        ax.set_xlim(0,limit)
        ax.set_xlabel('Selected windowed / full-sequence cost')
        ax.set_title(title,fontsize=10)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='x',alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y,[LABEL[m] for m in METHODS])
    axes[0].invert_yaxis()
    finish(fig, NAME[dataset] + ' - Windowed vs full-sequence training cost', out/dataset/'02_windowing_cost_comparison', bar_legend=True)


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
      '<h1>Memory, computation, and memory-access figures</h1>',
      '<p>Run comparisons use the costs and window-length profiles saved with each experiment. '
      'New runs include truncated final windows; historical runs retain their recorded preprocessing.</p>',
      '<p><a href="accuracy_memory_README.md">Accuracy/memory chart notes and hardware sources</a>'
      '<a href="accuracy_memory.csv">Accuracy/memory values</a>'
      '<a href="memory_budget_comparison.csv">Memory-budget comparisons</a></p>',
      '<nav>'+''.join(f'<a href="#{d}">{escape(NAME[d])}</a>' for d in datasets)+'</nav>']
    for dataset in datasets:
        html.append(f'<article id="{dataset}"><h2>{escape(NAME[dataset])}</h2>')
        names = ['01_computation_and_access_scaling', '02_windowing_cost_comparison',
                 '03_training_memory_scaling', '04_memory_case_study', '05_time_coefficient_sweep',
                 '06_accuracy_memory_budget']
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
        'Run `python -m reporting.create_results_overview` for all six figures, or add '
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
    selected=selected_data(meta)
    theory, selected = add_time_proxy(theory, selected)
    sweep = coefficient_sweep_data(selected)
    sweep.to_csv(output/'time_coefficient_sweep.csv', index=False)
    memory_cases = memory_case_study_data(meta)
    memory_cases.to_csv(output/'memory_case_study.csv', index=False)
    theory.to_csv(output/'theoretical_scaling.csv',index=False)
    selected.to_csv(output/'selected_vs_full.csv',index=False)
    (output/'configurations.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    for dataset in args.datasets:
        plot_theory(dataset,meta[dataset],theory[theory.dataset.eq(dataset)],output)
        dataset_selected = selected[selected.dataset.eq(dataset)]
        limit = max(1.25, float(dataset_selected[[m+'_ratio' for m in PLOT_METRICS]].max().max())*1.2)
        plot_selected(dataset,dataset_selected,limit,output)
        plot_memory(dataset, theory[theory.dataset.eq(dataset)], output)
        plot_memory_case_study(dataset, memory_cases[memory_cases.dataset.eq(dataset)], output)
        plot_coefficient_sweep(dataset, sweep[sweep.dataset.eq(dataset)], output)
        print(f'Saved 6 figures for {dataset}',flush=True)
    gallery(args.datasets,output)
    (output/'README.md').write_text("""# Results overview

Open index.html to browse all datasets. Figures are exported as PDF, SVG and PNG.
Run `python -m reporting.create_results_overview`; optional arguments are
`--datasets har pamap2` and `--output-dir PATH`.

1. **Computation, memory-access and time scaling**: sequence length and network depth.
2. **Windowed vs full-sequence training cost**: selected-window/hop cost ratios,
   using each run's recorded window-length profile. The baseline is 1.
3. **Training-memory scaling**: peak memory versus sequence length and depth.
4. **Memory case study**: baseline, 16-step window, two baseline-width hidden
   layers, and two 1024-neuron hidden layers. Peak memory counts one window's
   working set, not the sum over all windows. Values and dimensions are in
   memory_case_study.csv.

5. **Time coefficient sweep**: full-sequence cost only, as computation weight
   increases from 0 to 2 and memory-access weight decreases from 2 to 0.
   The total budget is 2. The midpoint (1, 1) exactly matches the equal-weight
   time costs in the other figures. Values are in time_coefficient_sweep.csv.

6. **Accuracy within a memory budget**: observed full-sequence and selected-window
   accuracy versus estimated peak training memory. Arrows connect each method's
   runs; dashed lines mark published hardware memory capacities. Values, feasibility
   comparisons and hardware source links are in accuracy_memory.csv,
   memory_budget_comparison.csv and memory_budget_references.csv. See
   accuracy_memory_README.md for assumptions. Use --accuracy-memory-only to add
   these figures without replacing the other existing cost charts, or
   --memory-budget-fraction 0.75 to reserve 25% of each capacity for other uses.

Time panels use time_proxy = computation + memory accesses (both weights 1).
Windowed/full time ratios divide these sums, not the individual cost ratios.
Time is an analytical cost proxy, not seconds or measured execution time.

Fixed dimensions are recorded in configurations.json. Pure theory uses the saved
maximum horizon for variable-length data; run comparisons use observed lengths.
Costs are analytical, not measured runtime or hardware traffic. Run comparisons
use the length profiles and costs recorded with those experiments. New windowed
runs record actual lengths, including shorter final windows. Historical runs retain
their original recorded costs and preprocessing. No training is run by this report.
The plotted values are saved
in theoretical_scaling.csv and selected_vs_full.csv.
""", encoding='utf-8')
    print(f'Validated {len(selected)*2} run profiles; gallery: {output / "index.html"}')


if __name__ == '__main__':
    main()
