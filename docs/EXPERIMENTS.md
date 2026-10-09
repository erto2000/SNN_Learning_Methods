# Experiments and reporting

The shared matrix in `experiment_config.py` covers ten dataset variants
and four learners, ordered BPTT, Forward-Forward, E-prop, and RATE-PEPITA.
MNIST Static and MNIST Rate use the same source images with different temporal
encodings. Each has a 10,000-sample cap. Their PEPITA runs use Adam with a
`1e-4` learning rate and the default `0.1` feedback modulation ratio.
MNIST Rate draws reproducible spikes per sample using the run's `SEED`.

HAR and PAMAP2 expose channel names and sampling rates for sensor plots.
PAMAP2 selects the 16g accelerometer, gyroscope, and magnetometer axes from
each sensor location. The corrected gyroscope and magnetometer column offsets
apply to new data loading; historical run summaries are preserved.

## Data protocol

`DATA_SPLIT = {"train": 80, "validation": 10, "test": 10}` defines percentages
of the available sample pool. Dataset loading applies class filters and sample
limits, combines the raw loader's original train and test portions, and makes
one seeded stratified partition. The split is made before preprocessing or
windowing. Preprocessing statistics are fitted on training samples only.

The normal and Optuna paths use the same partition and global `SEED` for a
given dataset configuration. Normal runs report test accuracy. Search trials
rank configurations by validation accuracy, then retrain the selected window
and hop settings at the normal epoch budget and report held-out test accuracy.
No separate full-sequence run is needed for Optuna final tests.

This is a new sample-level partition, not a dataset's official train/test
boundary. Samples from one subject or recording are not necessarily kept in
one partition. Report this protocol when presenting the results;
do not describe the test values as official or subject-independent benchmarks.
The saved run summaries include split counts, actual percentages, and hashes
of the selected sample indices.

## Run and compare

Run these entry points from the project root; they need no configuration flags:

```powershell
.\venv\Scripts\python.exe run_no_window.py
.\venv\Scripts\python.exe optuna_window_independent_two_phase.py
.\venv\Scripts\python.exe -m reporting.compare_no_window_runs
.\venv\Scripts\python.exe -m reporting.compare_independent_window_search
.\venv\Scripts\python.exe -m reporting.create_results_overview
```

`run_no_window.py` runs every configured dataset and learner. It writes a fresh
result for each run ID, replacing any result already at that path. The direct
entry point of the independent Optuna script currently selects the two MNIST
variants and uses `results/optuna/window_hop_independent_mnist_10k_v1.db`.
Its `datasets` setting can be edited for another campaign. All result files,
Optuna databases, comparison tables, and figures live under ignored `results/`.

The standard comparison writes a separate folder for each dataset variant and
an `all_current_runs` table. The independent Optuna comparison writes trial,
best-validation, and final-test tables. It keeps MNIST Static and Rate distinct.
`create_results_overview.py` is the single chart generator for all ten variants.
It writes figures to `results/results_overview/`; open `index.html` to browse
them. Each dataset has computation and memory-access scaling, windowed versus
full-sequence training costs, training-memory scaling, and the four-configuration
memory case study, a full-sequence time-coefficient sweep, and an accuracy versus
memory-budget chart using the actual full-sequence and selected-window runs. PDF, SVG, PNG,
and underlying CSV values are exported.

Use `--datasets har pamap2` or `--output-dir PATH` to select datasets or output.
The generator uses saved metadata and learner cost equations without training.

The accuracy/memory figure uses open circles for full sequences and filled
diamonds for selected windows, connected by an arrow for each method. Published
device memory capacities provide reference lines, with source URLs exported to
`memory_budget_references.csv`. These are capacity ceilings, not measured free
memory or deployment claims. `--memory-budget-fraction 0.75` allocates 75% of
each reference capacity to the modelled training working set. All configurations
retain their tested batch size, architecture and accuracy. Values and comparisons
are exported to `accuracy_memory.csv` and `memory_budget_comparison.csv`.
Use `--accuracy-memory-only` to add these figures while preserving the other
existing cost-chart exports, including locally recalculated historical charts.

The former `create_compute_access_charts.py`, `create_figure2_by_dataset.py`,
and `analyze_costs.py` entry points have been consolidated into this script.
The old broad report's search plots, budget plots, and weighted-proxy plots
are no longer generated. Existing output directories are historical snapshots.

## Interpreting estimated costs

Memory is an algorithmic peak working set for sequentially processed windows.
Compute and memory accesses describe mean work per configured batch of
original sequences, including all windows. They are estimates, not measured
GPU allocation, energy, or runtime. Forward-Forward costs cover a complete
layerwise sweep. Optimizer-state memory is excluded. RATE-PEPITA retains two
spike-rate buffers, one for each forward phase.

The weighted time proxy is `COST_COMPUTE_WEIGHT * compute + COST_ACCESS_WEIGHT * accesses`.
The two weights are set in `experiment_config.py`; plots can explore other weights.
The overview adds time-cost panels with computation and access weights both
set to 1. A coefficient-sweep chart holds their sum at 2, moving from access-only
to computation-only cost. Its midpoint is (1, 1), matching the other time panels. It shows only
full-sequence cost; these are model cost units, not seconds. Run comparisons use
the costs and window-length profiles recorded by each experiment. Historical runs
retain their original preprocessing and recorded estimates.
All current results use one seed, so small accuracy differences should not be
treated as statistically established method rankings.


## Partial final windows

The window schedule is defined by `window_spans` in `timeseries/transforms.py`.
It is recorded as `partial_last_window_v1` in run metadata.
Window starts stay on the hop grid, and iteration stops at the first window
that reaches the sequence end. The final window contains only the remaining
samples. For T=101, S=q=50, the lengths are 50, 50, 1. For overlapping
T=101, S=50, q=25, the lengths are 50, 50, 50, 26; no redundant one-step
suffix is appended. Hops larger than S keep their intended gaps. If the next
hop lies beyond the sequence end while a terminal remainder is uncovered,
append that remainder after the last covered window. For an exceptionally large
gap, its start is bounded below by T-S to keep the terminal window at most S
steps. For example, T=128, S=32, q=45 uses lengths 32,32,32,6.

Unequal windows use zero padding only for rectangular data-loader storage.
`window_lengths` metadata produces valid time masks; the learner iterator groups
windows by their actual lengths and removes storage padding before execution.
Evaluation combines votes from all length groups once per original sample.
A partial window has one vote, matching the existing majority-vote protocol.

Analytical work is the sum of C(T_i) and M(T_i) for all actual window lengths,
not window_count times C(S) or M(S). Per-window update overhead still repeats.
Peak memory is the maximum per-window working set.

New runs record actual processed lengths and source_length_counts before
windowing. Their computation, access, and peak-memory estimates are calculated
from those lengths during the run. Comparison tables, final-test exports, and
overview figures read the saved estimates without reconstructing old experiments.

The earlier analytical recalculation of historical runs is a separate local
migration. Its generated charts and data remain local artifacts. Running the
standard reporting commands again uses each experiment's recorded costs;
it does not apply the historical migration or change saved observations.
