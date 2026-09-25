"""Build figures and tables comparing completed standard and window-search runs."""
from pathlib import Path
import shutil
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT = Path("results/experiment_report")
COMBINED = OUT / "combined"
DATASET_OUT = OUT / "datasets"
RAW = OUT / "raw_results"
METHODS = ["bp", "ff", "eprop", "pepita"]
METHOD_LABEL = {
    "bp": "BPTT", "ff": "Forward-Forward",
    "eprop": "E-PROP", "pepita": "RATE-PEPITA",
}
COLORS = {
    "bp": "#377eb8", "ff": "#ff7f00",
    "eprop": "#4daf4a", "pepita": "#e41a1c",
}
DATASETS = [
    "har", "mnist_static", "mnist_rate", "pamap2", "speech_commands", "esc50",
    "large_scale_audio", "urban8k", "dvs_gesture", "mitbih",
]
DATASET_LABEL = {
    "har": "HAR", "mnist_static": "MNIST Static", "mnist_rate": "MNIST Rate",
    "pamap2": "PAMAP2",
    "speech_commands": "Speech\nCommands", "esc50": "ESC-50",
    "large_scale_audio": "Large-Scale\nAudio", "urban8k": "UrbanSound8K",
    "dvs_gesture": "DVS Gesture", "mitbih": "MIT-BIH",
}
DATASET_SHORT = {
    "har": "HAR", "mnist_static": "MNIST-S", "mnist_rate": "MNIST-R",
    "pamap2": "PAMAP2", "speech_commands": "SC",
    "esc50": "ESC-50", "large_scale_audio": "LSA", "urban8k": "US8K",
    "dvs_gesture": "DVS", "mitbih": "MIT-BIH",
}


def style(ax, grid="y"):
    ax.grid(True, axis=grid, color="#d8d8d8", linewidth=0.65, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#777777")


def save(fig, directory, stem):
    directory.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(directory / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_data():
    standard = pd.read_csv(
        "results/comparisons/all_current_runs/leaderboard.csv"
    )
    standard = standard[standard.dataset.isin(DATASETS)][
        ["dataset", "learner", "final_sample_acc", "theory_memory_bytes", "theory_compute_scalars",
         "theory_access_scalars", "theory_time_proxy"]
    ].rename(columns={"final_sample_acc": "no_window_acc",
        **{f"theory_{name}": f"standard_cost_{name}" for name in
           ("memory_bytes", "compute_scalars", "access_scalars", "time_proxy")}})
    final = pd.read_csv(
        "results/comparisons/window_hop_independent/final_test_results.csv"
    )
    final = final[final.experiment.eq("hop")][
        ["dataset", "learner", "test_sample_acc", "validation_score", "win_L", "hop", "theory_memory_bytes",
         "theory_compute_scalars", "theory_access_scalars", "theory_time_proxy"]
    ].rename(columns={"test_sample_acc": "window_acc",
        "validation_score": "selected_validation_acc", "win_L": "window_L",
        **{f"theory_{name}": f"windowed_cost_{name}" for name in
           ("memory_bytes", "compute_scalars", "access_scalars", "time_proxy")}})
    merged = standard.merge(final, on=["dataset", "learner"], validate="one_to_one")
    merged["window_gain"] = merged.window_acc - merged.no_window_acc
    merged["memory_reduction_percent"] = 100 * (
        1 - merged.windowed_cost_memory_bytes / merged.standard_cost_memory_bytes
    )
    merged["actual_hop_ratio"] = merged.hop / merged.window_L
    return merged


def accuracy_panels(d):
    columns = 4
    rows = math.ceil(len(DATASETS) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(14, 3.5 * rows), sharey=True)
    x = np.arange(len(METHODS))
    width = 0.36
    for ax, dataset in zip(axes.flat, DATASETS):
        g = d[d.dataset.eq(dataset)].set_index("learner").loc[METHODS]
        ax.bar(x - width / 2, g.no_window_acc, width, color="#aeb6bf",
               label="No window")
        ax.bar(x + width / 2, g.window_acc, width,
               color=[COLORS[m] for m in METHODS], label="Optimized window")
        ax.set_title(DATASET_LABEL[dataset].replace("\n", " "), fontsize=10, weight="bold")
        ax.set_xticks(x, [METHOD_LABEL[m] for m in METHODS], rotation=28, ha="right", fontsize=8)
        ax.set_ylim(0, 103)
        style(ax)
    for ax in axes.flat[len(DATASETS):]:
        ax.set_visible(False)
    for row in range(rows):
        axes[row, 0].set_ylabel("Test accuracy (%)")
    handles = [plt.Rectangle((0, 0), 1, 1, color="#aeb6bf"),
               plt.Rectangle((0, 0), 1, 1, color="#4b86b4")]
    fig.legend(handles, ["No window", "Optimized window"], loc="upper center",
               ncols=2, frameon=False, bbox_to_anchor=(0.5, 1.015))
    fig.suptitle("Test accuracy before and after temporal-window optimization",
                 fontsize=14, weight="bold", y=1.06)
    save(fig, COMBINED, "01_accuracy_by_dataset")


def window_gain(d):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    y = np.arange(len(DATASETS))
    for ax, method in zip(axes.flat, METHODS):
        g = d[d.learner.eq(method)].set_index("dataset").loc[DATASETS]
        vals = g.window_gain.to_numpy()
        ax.barh(y, vals, color=np.where(vals >= 0, COLORS[method], "#b64b4b"))
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_yticks(y, [DATASET_LABEL[x].replace("\n", " ") for x in DATASETS], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(METHOD_LABEL[method], weight="bold")
        for yi, value in zip(y, vals):
            ax.text(value + (0.35 if value >= 0 else -0.35), yi, f"{value:+.1f}",
                    va="center", ha="left" if value >= 0 else "right", fontsize=8)
        style(ax, grid="x")
    axes[1, 0].set_xlabel("Change in test accuracy (percentage points)")
    axes[1, 1].set_xlabel("Change in test accuracy (percentage points)")
    fig.suptitle("Accuracy change produced by optimized temporal windowing",
                 fontsize=14, weight="bold")
    save(fig, COMBINED, "02_windowing_accuracy_gain")


def parameter_heatmaps(d):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    for ax, column, title, fmt in [
        (axes[0], "window_L", "Selected window length", ".0f"),
        (axes[1], "actual_hop_ratio", "Selected hop-to-window ratio", ".2f"),
    ]:
        matrix = d.pivot(index="learner", columns="dataset", values=column).loc[METHODS, DATASETS]
        image = ax.imshow(matrix, aspect="auto", cmap="YlGnBu")
        ax.set_xticks(np.arange(len(DATASETS)),
                      [DATASET_SHORT[x] for x in DATASETS], fontsize=8,
                      rotation=30, ha="right")
        ax.set_yticks(np.arange(len(METHODS)), [METHOD_LABEL[x] for x in METHODS], fontsize=8)
        ax.set_title(title, weight="bold")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix.iloc[i, j]
                ax.text(j, i, format(value, fmt), ha="center", va="center", fontsize=8,
                        color="white" if value > np.nanmedian(matrix.to_numpy()) else "#1f1f1f")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Temporal parameters selected using validation data",
                 fontsize=14, weight="bold")
    save(fig, COMBINED, "03_selected_window_parameters")


def accuracy_memory(d):
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    for method in METHODS:
        g = d[d.learner.eq(method)]
        ax.scatter(g.memory_reduction_percent, g.window_gain, s=70,
                   color=COLORS[method], edgecolor="white", linewidth=0.6,
                   label=METHOD_LABEL[method])
        for _, row in g.iterrows():
            ax.annotate(DATASET_SHORT[row.dataset],
                        (row.memory_reduction_percent, row.window_gain),
                        xytext=(4, 3), textcoords="offset points", fontsize=6.5, alpha=0.8)
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_xlabel("Peak estimated-memory reduction (%)")
    ax.set_ylabel("Test-accuracy change (percentage points)")
    ax.set_title("Accuracy–memory effect of optimized temporal windowing", weight="bold")
    ax.legend(frameon=False, ncols=2)
    style(ax, grid="both")
    save(fig, COMBINED, "04_accuracy_memory_tradeoff")


def validation_test(d):
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    for method in METHODS:
        g = d[d.learner.eq(method)]
        ax.scatter(g.selected_validation_acc, g.window_acc, s=65,
                   color=COLORS[method], edgecolor="white", linewidth=0.6,
                   label=METHOD_LABEL[method])
    ax.plot([55, 102], [55, 102], linestyle="--", color="#555555", linewidth=1)
    ax.set_xlim(55, 102)
    ax.set_ylim(55, 102)
    ax.set_xlabel("Validation accuracy used for selection (%)")
    ax.set_ylabel("Held-out test accuracy (%)")
    ax.set_title("Validation-to-test behavior of selected configurations", weight="bold")
    ax.legend(frameon=False, ncols=2)
    style(ax, grid="both")
    save(fig, COMBINED, "05_validation_vs_test")


def no_window_method_comparisons(d):
    """Compare learning methods directly using only full-sequence runs."""
    no_window_columns = [
        "dataset", "learner", "no_window_acc",
        "standard_cost_memory_bytes", "standard_cost_compute_scalars",
        "standard_cost_access_scalars", "standard_cost_time_proxy",
    ]
    no_window = d[no_window_columns].copy()
    no_window.to_csv(RAW / "no_window_method_results.csv", index=False,
                     float_format="%.4f")

    # Accuracy heatmap: direct absolute comparison.
    accuracy = d.pivot(index="learner", columns="dataset",
                       values="no_window_acc").loc[METHODS, DATASETS]
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    im = ax.imshow(accuracy, aspect="auto", cmap="YlGnBu", vmin=45, vmax=100)
    ax.set_xticks(range(len(DATASETS)), [DATASET_SHORT[x] for x in DATASETS],
                  rotation=30, ha="right")
    ax.set_yticks(range(4), [METHOD_LABEL[x] for x in METHODS])
    ax.set_title("No-window test accuracy by learning method", weight="bold")
    for i in range(4):
        for j in range(len(DATASETS)):
            value = accuracy.iloc[i, j]
            best = value == accuracy.iloc[:, j].max()
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=9,
                    weight="bold" if best else "normal",
                    color="white" if value > 84 else "#222222")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cb.set_label("Held-out test accuracy (%)")
    save(fig, COMBINED, "08_no_window_accuracy")

    # Relative costs make datasets with very different input sizes comparable.
    cost_specs = [
        ("standard_cost_memory_bytes", "Peak learner memory"),
        ("standard_cost_compute_scalars", "Arithmetic work"),
        ("standard_cost_access_scalars", "Memory access"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (column, title) in zip(axes, cost_specs):
        matrix = d.pivot(index="learner", columns="dataset", values=column).loc[METHODS, DATASETS]
        relative = matrix / matrix.min(axis=0)
        im = ax.imshow(relative, aspect="auto", cmap="YlOrRd", vmin=1,
                       vmax=max(3, relative.to_numpy().max()))
        ax.set_xticks(range(len(DATASETS)), [DATASET_SHORT[x] for x in DATASETS],
                      rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(4), [METHOD_LABEL[x] for x in METHODS], fontsize=8)
        ax.set_title(title, weight="bold")
        for i in range(4):
            for j in range(len(DATASETS)):
                ax.text(j, i, f"{relative.iloc[i, j]:.1f}×", ha="center",
                        va="center", fontsize=7.5)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("No-window estimated cost relative to the lowest-cost method",
                 fontsize=14, weight="bold")
    save(fig, COMBINED, "09_no_window_relative_costs")

    # Cross-dataset method profile: mean accuracy and geometric-mean relative costs.
    profile_rows = []
    for method in METHODS:
        group = d[d.learner.eq(method)]
        row = {"learner": method, "mean_accuracy": group.no_window_acc.mean()}
        for column, short in [
            ("standard_cost_memory_bytes", "memory"),
            ("standard_cost_compute_scalars", "compute"),
            ("standard_cost_access_scalars", "access"),
        ]:
            relative = []
            for dataset in DATASETS:
                dg = d[d.dataset.eq(dataset)]
                relative.append(float(group[group.dataset.eq(dataset)][column].iloc[0]) /
                                float(dg[column].min()))
            row[f"geomean_relative_{short}"] = float(np.exp(np.mean(np.log(relative))))
        profile_rows.append(row)
    profile = pd.DataFrame(profile_rows)
    profile.to_csv(RAW / "no_window_method_summary.csv", index=False,
                   float_format="%.4f")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    x = np.arange(4)
    axes[0].bar(x, profile.mean_accuracy, color=[COLORS[m] for m in METHODS])
    axes[0].set_xticks(x, [METHOD_LABEL[m] for m in METHODS], rotation=20, ha="right")
    axes[0].set_ylabel("Mean held-out test accuracy (%)")
    axes[0].set_title("Accuracy averaged across datasets", weight="bold")
    for i, value in enumerate(profile.mean_accuracy):
        axes[0].text(i, value + 0.5, f"{value:.1f}", ha="center", fontsize=9)
    style(axes[0])
    width = 0.23
    for i, (column, label, hatch) in enumerate([
        ("geomean_relative_memory", "Memory", ""),
        ("geomean_relative_compute", "Compute", "//"),
        ("geomean_relative_access", "Access", "xx"),
    ]):
        axes[1].bar(x + (i - 1) * width, profile[column], width,
                    color=[COLORS[m] for m in METHODS], alpha=0.85,
                    hatch=hatch, edgecolor="white", label=label)
    axes[1].axhline(1, color="#555555", linestyle="--", linewidth=0.9)
    axes[1].set_xticks(x, [METHOD_LABEL[m] for m in METHODS], rotation=20, ha="right")
    axes[1].set_ylabel("Geometric-mean cost relative to dataset minimum")
    axes[1].set_title("Algorithmic cost averaged across datasets", weight="bold")
    axes[1].legend(frameon=False, ncols=3)
    style(axes[1])
    fig.suptitle("No-window method comparison", fontsize=14, weight="bold")
    save(fig, COMBINED, "10_no_window_method_summary")

    # Dataset-specific absolute comparisons and accuracy-cost tradeoffs.
    for dataset in DATASETS:
        g = d[d.dataset.eq(dataset)].set_index("learner").loc[METHODS]
        out = DATASET_OUT / dataset
        out.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 8))
        panels = [
            ("no_window_acc", "Test accuracy (%)", False),
            ("standard_cost_memory_bytes", "Peak memory (MB)", True),
            ("standard_cost_compute_scalars", "Arithmetic work (million units)", True),
            ("standard_cost_access_scalars", "Memory access (million scalars)", True),
        ]
        divisors = [1, 1024 ** 2, 1e6, 1e6]
        for ax, (column, ylabel, log), divisor in zip(axes.flat, panels, divisors):
            values = g[column] / divisor
            bars = ax.bar(np.arange(4), values, color=[COLORS[m] for m in METHODS])
            ax.set_xticks(np.arange(4), [METHOD_LABEL[m] for m in METHODS],
                          rotation=22, ha="right", fontsize=8)
            ax.set_ylabel(ylabel)
            if log:
                ax.set_yscale("log")
            for bar, value in zip(bars, values):
                label = f"{value:.1f}" if value >= 1 else f"{value:.2f}"
                ax.text(bar.get_x() + bar.get_width() / 2, value * (1.08 if log else 1.01),
                        label, ha="center", va="bottom", fontsize=8)
            style(ax)
        fig.suptitle(f"{DATASET_LABEL[dataset].replace(chr(10), ' ')}: no-window method comparison",
                     fontsize=14, weight="bold")
        save(fig, out, "05_no_window_method_comparison")

        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), sharey=True)
        for ax, (column, xlabel, divisor) in zip(axes, [
            ("standard_cost_memory_bytes", "Peak memory (MB)", 1024 ** 2),
            ("standard_cost_compute_scalars", "Arithmetic work (million units)", 1e6),
            ("standard_cost_access_scalars", "Memory access (million scalars)", 1e6),
        ]):
            for method in METHODS:
                row = g.loc[method]
                ax.scatter(row[column] / divisor, row.no_window_acc, s=85,
                           color=COLORS[method], edgecolor="white", linewidth=0.7,
                           label=METHOD_LABEL[method])
                ax.annotate(METHOD_LABEL[method],
                            (row[column] / divisor, row.no_window_acc),
                            xytext=(4, 3), textcoords="offset points", fontsize=7)
            ax.set_xscale("log")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Test accuracy (%)")
            style(ax, grid="both")
        fig.suptitle(f"{DATASET_LABEL[dataset].replace(chr(10), ' ')}: no-window accuracy–cost tradeoffs",
                     fontsize=14, weight="bold")
        save(fig, out, "06_no_window_accuracy_cost_tradeoffs")


def export_tables(d):
    d.to_csv(RAW / "complete_results.csv", index=False, float_format="%.4f")
    compact = d[[
        "dataset", "learner", "no_window_acc", "window_acc",
        "window_gain", "window_L", "hop", "actual_hop_ratio",
        "selected_validation_acc", "memory_reduction_percent",
    ]].copy()
    compact["learner"] = compact.learner.map(METHOD_LABEL)
    compact.to_csv(RAW / "results_compact.csv", index=False, float_format="%.3f")
    method_summary = d.groupby("learner").agg(
        no_window_accuracy=("no_window_acc", "mean"),
        optimized_accuracy=("window_acc", "mean"),
        windowing_gain=("window_gain", "mean"),
        memory_reduction=("memory_reduction_percent", "mean"),
    ).reindex(METHODS)
    method_summary.index = method_summary.index.map(METHOD_LABEL)
    method_summary.to_csv(RAW / "method_summary.csv", float_format="%.3f")
    dataset_summary = d.groupby("dataset").agg(
        no_window_accuracy=("no_window_acc", "mean"),
        optimized_accuracy=("window_acc", "mean"),
        windowing_gain=("window_gain", "mean"),
        memory_reduction=("memory_reduction_percent", "mean"),
    ).reindex(DATASETS)
    dataset_summary.to_csv(RAW / "dataset_summary.csv", float_format="%.3f")


def estimated_cost_outputs(d):
    metrics = [
        ("cost_memory_bytes", "Peak learner memory (MB)", 1024 ** 2),
        ("cost_compute_scalars", "Arithmetic work (million units)", 1e6),
        ("cost_access_scalars", "Memory access (million scalars)", 1e6),
    ]
    long_rows = []
    for _, row in d.iterrows():
        for setting, prefix in (("No window", "standard"), ("Optimized window", "windowed")):
            long_rows.append({
                "dataset": row.dataset, "learner": row.learner, "setting": setting,
                "memory_bytes": row[f"{prefix}_cost_memory_bytes"],
                "compute_scalars": row[f"{prefix}_cost_compute_scalars"],
                "access_scalars": row[f"{prefix}_cost_access_scalars"],
                "time_proxy_alpha1_beta1": row[f"{prefix}_cost_time_proxy"],
            })
    costs = pd.DataFrame(long_rows)
    costs.to_csv(RAW / "estimated_costs_long.csv", index=False, float_format="%.4f")

    # One large heatmap per component stays legible in a report.
    ratio_titles = {
        "cost_memory_bytes": "Peak learner memory",
        "cost_compute_scalars": "Arithmetic work",
        "cost_access_scalars": "Memory access",
    }
    for number, (metric, title, _) in enumerate(metrics, start=1):
        fig, ax = plt.subplots(figsize=(10.5, 4.8))
        ratio = d.pivot(index="learner", columns="dataset",
                        values=f"windowed_{metric}").loc[METHODS, DATASETS]
        base = d.pivot(index="learner", columns="dataset",
                       values=f"standard_{metric}").loc[METHODS, DATASETS]
        matrix = 100 * ratio / base
        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=160)
        ax.set_xticks(range(len(DATASETS)), [DATASET_SHORT[x] for x in DATASETS],
                      rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(METHODS)), [METHOD_LABEL[x] for x in METHODS], fontsize=8)
        ax.set_title(f"Optimized-window {ratio_titles[metric]} relative to no-window cost",
                     weight="bold")
        for i in range(4):
            for j in range(len(DATASETS)):
                ax.text(j, i, f"{matrix.iloc[i, j]:.0f}%", ha="center", va="center",
                        fontsize=7, color="white" if matrix.iloc[i, j] > 110 else "#222222")
        colorbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
        colorbar.set_label("Windowed / no-window cost (%)")
        save(fig, COMBINED, f"06{chr(96 + number)}_{metric}_ratio")

    # Dataset-specific raw cost panels.
    for dataset in DATASETS:
        g = d[d.dataset.eq(dataset)].set_index("learner").loc[METHODS]
        fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
        y = np.arange(4)
        for ax, (metric, ylabel, divisor) in zip(axes, metrics):
            standard = (g[f"standard_{metric}"] / divisor).to_numpy()
            windowed = (g[f"windowed_{metric}"] / divisor).to_numpy()
            for yi, method, before, after in zip(y, METHODS, standard, windowed):
                ax.plot([before, after], [yi, yi], color=COLORS[method], linewidth=2.2)
                ax.scatter(before, yi, s=58, facecolor="white", edgecolor=COLORS[method],
                           linewidth=1.8, zorder=3)
                ax.scatter(after, yi, s=58, color=COLORS[method], edgecolor="white",
                           linewidth=0.6, zorder=4)
                ax.annotate(f"{100 * after / before:.0f}%", (max(before, after), yi),
                            xytext=(7, 0), textcoords="offset points", va="center", fontsize=8)
            ax.set_xscale("log")
            ax.set_xlabel(ylabel)
            ax.set_yticks(y, [METHOD_LABEL[m] for m in METHODS], fontsize=8)
            ax.invert_yaxis()
            style(ax, grid="x")
        handles = [
            plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="white",
                       markeredgecolor="#555555", label="No window"),
            plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="#555555",
                       markeredgecolor="white", label="Optimized window"),
        ]
        fig.legend(handles=handles, loc="upper center", ncols=2, frameon=False,
                   bbox_to_anchor=(0.5, 1.02))
        fig.suptitle(f"{DATASET_LABEL[dataset].replace(chr(10), ' ')}: estimated costs",
                     fontsize=14, weight="bold", y=1.08)
        save(fig, DATASET_OUT / dataset, "02_estimated_costs")


def assumption_sensitivity(d):
    assumptions = [
        ("0.10", 0.10, 1.0), ("0.25", 0.25, 1.0),
        ("1.00", 1.00, 1.0), ("4.00", 4.00, 1.0),
        ("10.00", 10.00, 1.0),
    ]
    rows = []
    for _, row in d.iterrows():
        for setting, prefix in (("No window", "standard"), ("Optimized window", "windowed")):
            for name, alpha, beta in assumptions:
                compute = row[f"{prefix}_cost_compute_scalars"]
                access = row[f"{prefix}_cost_access_scalars"]
                rows.append({
                    "dataset": row.dataset, "learner": row.learner, "setting": setting,
                    "assumption": name, "alpha": alpha, "beta": beta,
                    "compute_scalars": compute, "access_scalars": access,
                    "weighted_cost": alpha * compute + beta * access,
                })
    sensitivity = pd.DataFrame(rows)
    sensitivity["relative_to_best"] = sensitivity.groupby(
        ["dataset", "setting", "assumption"]
    ).weighted_cost.transform(lambda x: x / x.min())
    sensitivity.to_csv(RAW / "cost_assumption_sensitivity.csv", index=False, float_format="%.5f")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for ax, setting in zip(axes, ("No window", "Optimized window")):
        matrix = sensitivity[sensitivity.setting.eq(setting)].groupby(
            ["learner", "assumption"]
        ).relative_to_best.mean().unstack().loc[METHODS, [x[0] for x in assumptions]]
        ratios = np.array([float(x[0]) for x in assumptions])
        for method in METHODS:
            ax.plot(ratios, matrix.loc[method], marker="o", linewidth=2.2,
                    color=COLORS[method], label=METHOD_LABEL[method])
        ax.set_xscale("log")
        ax.set_xticks(ratios, [x[0] for x in assumptions])
        ax.set_title(setting, weight="bold")
        ax.set_xlabel("Relative compute/access weight (α/β)")
        ax.set_ylabel("Mean cost relative to best method")
        style(ax, grid="both")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=4, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Cost sensitivity to the relative price of compute and memory access",
                 fontsize=14, weight="bold")
    save(fig, COMBINED, "07_cost_assumption_sensitivity")

    for dataset in DATASETS:
        subset = sensitivity[sensitivity.dataset.eq(dataset)]
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=True)
        ratios = np.array([float(a[0]) for a in assumptions])
        for ax, setting in zip(axes, ("No window", "Optimized window")):
            part = subset[subset.setting.eq(setting)]
            for method in METHODS:
                values = part[part.learner.eq(method)].set_index("assumption").loc[
                    [a[0] for a in assumptions], "relative_to_best"
                ]
                ax.plot(ratios, values, marker="o", linewidth=2.2,
                        color=COLORS[method], label=METHOD_LABEL[method])
            ax.set_xscale("log")
            ax.set_xticks(ratios, [a[0] for a in assumptions])
            ax.set_title(setting, weight="bold")
            ax.set_ylabel("Cost relative to best method")
            ax.set_xlabel("Relative compute/access weight (α/β)")
            style(ax, grid="both")
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncols=4, frameon=False,
                   bbox_to_anchor=(0.5, 1.02))
        fig.suptitle(f"{DATASET_LABEL[dataset].replace(chr(10), ' ')}: cost-assumption sensitivity",
                     fontsize=14, weight="bold", y=1.08)
        save(fig, DATASET_OUT / dataset, "03_cost_assumption_sensitivity")


def dataset_accuracy_and_search(d):
    trials = pd.read_csv("results/comparisons/window_hop_independent/all_trials.csv")
    trials["dataset"] = pd.Categorical(trials.dataset, DATASETS, ordered=True)
    trials["learner"] = pd.Categorical(trials.learner, METHODS, ordered=True)
    trials = trials.sort_values(["dataset", "learner", "experiment", "phase", "trial"])
    trials.to_csv(RAW / "optuna_all_validation_trials.csv", index=False)
    standard_raw = pd.read_csv("results/comparisons/all_current_runs/leaderboard.csv")
    standard_raw = standard_raw[standard_raw.dataset.isin(DATASETS)]
    standard_raw["dataset"] = pd.Categorical(standard_raw.dataset, DATASETS, ordered=True)
    standard_raw["learner"] = pd.Categorical(standard_raw.learner, METHODS, ordered=True)
    standard_raw.sort_values(["dataset", "learner"]).to_csv(
        RAW / "standard_run_results.csv", index=False)
    final_raw = pd.read_csv("results/comparisons/window_hop_independent/final_test_results.csv")
    final_raw["dataset"] = pd.Categorical(final_raw.dataset, DATASETS, ordered=True)
    final_raw["learner"] = pd.Categorical(final_raw.learner, METHODS, ordered=True)
    final_raw.sort_values(["dataset", "learner", "experiment"]).to_csv(
        RAW / "optuna_final_test_results.csv", index=False)
    for dataset in DATASETS:
        g = d[d.dataset.eq(dataset)].set_index("learner").loc[METHODS]
        out = DATASET_OUT / dataset
        out.mkdir(parents=True, exist_ok=True)
        g.reset_index().to_csv(out / "results.csv", index=False, float_format="%.4f")
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        x = np.arange(4)
        width = 0.36
        ax.bar(x - width / 2, g.no_window_acc, width, color="#b8b8b8", label="No window")
        ax.bar(x + width / 2, g.window_acc, width,
               color=[COLORS[m] for m in METHODS], label="Optimized window")
        ax.set_xticks(x, [METHOD_LABEL[m] for m in METHODS])
        ax.set_ylabel("Held-out test accuracy (%)")
        ax.set_title(f"{DATASET_LABEL[dataset].replace(chr(10), ' ')}: test accuracy", weight="bold")
        ax.legend(frameon=False, ncols=2)
        style(ax)
        save(fig, out, "01_test_accuracy")

        td = trials[trials.dataset.eq(dataset)]
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
        for method in METHODS:
            w = td[(td.learner.eq(method)) & (td.experiment.eq("window"))]
            h = td[(td.learner.eq(method)) & (td.experiment.eq("hop"))]
            axes[0].scatter(w.win_L, w.sample_acc, s=20, alpha=0.65,
                            color=COLORS[method], label=METHOD_LABEL[method])
            axes[1].scatter(h.hop_ratio, h.sample_acc, s=20, alpha=0.65,
                            color=COLORS[method], label=METHOD_LABEL[method])
        axes[0].set_xlabel("Window length")
        axes[1].set_xlabel("Hop-to-window ratio")
        for ax in axes:
            ax.set_ylabel("Validation accuracy (%)")
            style(ax, grid="both")
        axes[0].set_title("Window-length trials", weight="bold")
        axes[1].set_title("Hop-ratio trials", weight="bold")
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncols=4, frameon=False,
                   bbox_to_anchor=(0.5, 1.02))
        fig.suptitle(f"{DATASET_LABEL[dataset].replace(chr(10), ' ')}: Optuna search",
                     fontsize=14, weight="bold", y=1.08)
        save(fig, out, "04_optuna_search")


def write_notes(d):
    gains = int((d.window_gain > 0).sum())
    losses = int((d.window_gain < 0).sum())
    text = f"""# Experiment results report

This directory summarizes the completed standard and independent two-stage
Optuna experiments. All optimized-window accuracies are held-out test results;
window length and hop were selected using validation accuracy.

## Main observations

- Mean no-window accuracy: **{d.no_window_acc.mean():.2f}%**.
- Mean optimized-window accuracy: **{d.window_acc.mean():.2f}%**.
- Mean windowing change: **{d.window_gain.mean():+.2f} percentage points**.
- Windowing improved {gains} of {len(d)} dataset-method combinations and reduced {losses}.

## Combined figure guide

1. `01_accuracy_by_dataset`: direct method comparison with and without windowing.
2. `02_windowing_accuracy_gain`: where windowing helps or hurts each method.
3. `03_selected_window_parameters`: final temporal parameters selected on validation.
4. `04_accuracy_memory_tradeoff`: joint accuracy and peak learner-memory effect.
5. `05_validation_vs_test`: checks whether selected validation performance transfers to test.
6. `06a`–`06c`: large, separate memory, computation, and access-ratio heatmaps.
7. `07_cost_assumption_sensitivity`: method costs across α/β from 0.1 to 10.
8. `08_no_window_accuracy`: absolute full-sequence accuracy by method and dataset.
9. `09_no_window_relative_costs`: full-sequence memory, compute, and access comparisons.
10. `10_no_window_method_summary`: cross-dataset accuracy and cost profile by method.

The `datasets` directory contains figures and a result table for each
dataset. The `raw_results` directory contains standard runs, validation trials,
final Optuna tests, estimated components, and every cost-assumption scenario.
MNIST Static and MNIST Rate are separate datasets in all summaries.
PNG files are convenient for review and SVG files suit resizable reports.

Estimated memory is algorithmic learner working memory under sequential
window processing. It is not measured process or GPU memory. Results are from
one seed, so small differences should not be presented as statistical evidence.
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main():
    if OUT.exists():
        output = OUT.resolve()
        if output.parent != Path("results").resolve():
            raise ValueError(f"Unexpected report output path: {output}")
        shutil.rmtree(output)
    COMBINED.mkdir(parents=True)
    DATASET_OUT.mkdir(parents=True)
    RAW.mkdir(parents=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlepad": 8, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white",
    })
    data = load_data()
    if len(data) != len(DATASETS) * len(METHODS):
        raise ValueError(f"Expected {len(DATASETS) * len(METHODS)} dataset-method results, found {len(data)}")
    export_tables(data)
    accuracy_panels(data)
    window_gain(data)
    parameter_heatmaps(data)
    accuracy_memory(data)
    validation_test(data)
    no_window_method_comparisons(data)
    dataset_accuracy_and_search(data)
    estimated_cost_outputs(data)
    assumption_sensitivity(data)
    write_notes(data)
    print(f"Saved experiment report to {OUT.resolve()}")


if __name__ == "__main__":
    main()
