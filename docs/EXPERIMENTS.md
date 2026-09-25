# Experiments and reporting

The shared matrix in `experiment_config.py` covers ten dataset variants
and four learners, ordered BPTT, Forward-Forward, E-prop, and RATE-PEPITA.
MNIST Static and MNIST Rate use the same source images with different temporal
encodings. Each has a 10,000-sample cap. Their PEPITA runs use Adam with a
`1e-4` learning rate and the default `0.1` feedback modulation ratio.
MNIST Rate draws reproducible spikes per sample using the run's `SEED`.

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
`create_results_overview.py` uses these current tables to build combined
figures, per-dataset figures, and raw tables for all ten variants.

## Interpreting estimated costs

Memory is an algorithmic peak working set for sequentially processed windows.
Compute and memory accesses describe mean work per configured batch of
original sequences, including all windows. They are estimates, not measured
GPU allocation, energy, or runtime. Forward-Forward costs cover a complete
layerwise sweep. Optimizer-state memory is excluded. RATE-PEPITA retains two
spike-rate buffers, one for each forward phase.

The weighted time proxy is `COST_COMPUTE_WEIGHT * compute + COST_ACCESS_WEIGHT * accesses`.
The two weights are set in `experiment_config.py`; plots can explore other weights.
All current results use one seed, so small accuracy differences should not be
treated as statistically established method rankings.
