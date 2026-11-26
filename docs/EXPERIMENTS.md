# Experiments and reporting

The shared matrix in `experiment_config.py` covers ten dataset variants
and four learners, ordered BPTT, Forward-Forward, E-prop, and RATE-PEPITA.
MNIST Static and MNIST Rate use the same source images with different temporal
encodings, both with 30 time steps. Each has a 10,000-sample cap. Their PEPITA runs use Adam with a
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
result for each run ID, replacing any result already at that path. The independent
Optuna script also runs all ten variants and uses
`results/optuna/window_hop_independent_recording_batch_v3.db`.
Its `datasets` argument can select a smaller campaign. All result files,
Optuna databases, comparison tables, and figures live under ignored `results/`.

Use the independent script for the thesis protocol: select window length first,
then select hop at the chosen length. The joint-grid script explores a different
protocol. The independent broad window search covers every integer from the
rounded 10% horizon (at least one step) to the inspected training maximum.
Full length is a normal candidate; it is not enqueued as a special trial.
Refinement covers the rounded 75%--125% neighborhood, bounded by one step and
the inspected maximum. Hop search samples integer hops directly within its ratio
bounds. Both refinement samplers use three startup trials so TPE can adapt within
the ten-trial budget.

Budgets are up to 20 newly trained networks at five epochs and ten at ten epochs
per stage. Duplicate proposals do not consume that budget. Observations for the
same effective schedule and epoch count are reused across stages. Effective
schedule equivalence is checked only with a guaranteed fixed horizon or a
complete training-length profile; otherwise deduplication conservatively uses
the exact integer length and hop. A finite exhausted space stops early without
moving unused budget to another stage. The selected ten-epoch non-overlap setting
remains a validation-based fallback if hop tuning is worse. Full-sequence runs
are separate baselines, and are not injected into the search.

Before a fresh campaign, archive the existing runs, Optuna databases, final-test
manifests, and comparison/report folders together. Resume checks reject studies
or selected final tests from a different training protocol. Regenerate comparisons
and charts after both scripts finish; update thesis numbers from those fresh exports.
The current thesis figures and accuracy values describe the archived campaign.

The standard comparison writes a separate folder for each dataset variant and
an `all_current_runs` table. The independent Optuna comparison writes trial,
best-validation, and final-test tables. It keeps MNIST Static and Rate distinct.

```powershell
python reporting/create_results_overview.py
```

The script regenerates theoretical figures and CSVs from `experiment_config.py`,
the datasets, and the current learner formulas. It writes to
`results/results_overview/`. Open `index.html` to browse the results.
Accuracy and validation-selected capacity figures are added for datasets with a
complete set of baseline runs and validation-selected hop records.

The script covers all ten variants. It produces computation and memory-access
scaling, a controlled window-length sweep, training-memory scaling, a memory
case study, and a full-sequence fast-memory capacity sweep. The memory case
study includes the original illustrative 16-step window. When run records are
available, the script adds the accuracy/memory chart and windowed/full
cycle-proxy capacity sweep. Figures are exported as PDF, SVG and PNG, with
underlying values in CSV files.

The overview reads measured accuracy and tested settings directly from run
summaries and Optuna selection manifests. Reporting and comparison commands
calculate theoretical costs with the current implementation and preprocessing. For historical runs,
the current windowing rule can differ from the one used to measure accuracy.
The accuracy/memory CSV identifies both rules and training split hashes.

`02_controlled_windowing_sweep` replaces the selected-configuration cost
bars/scatter. Three panels show peak memory, total computation and total memory
accesses versus relative window length L/T. T is the maximum preprocessed training sequence
length. Every method uses every integer L from 1 to T with hop=L, so the policy
is non-overlapping throughout. Within each dataset the preprocessed input-length
distribution, batch size and model dimensions are common to all methods.
Variable-length datasets use the full training split's length distribution, with
the same L applied to every sequence. Short final windows count at their actual
length, and every source timestep counts once.
Total work sums all windows and is averaged over original sequences, while peak
memory uses the largest working set of any window. Ratios use each method's own
calculated full-sequence costs and are 1 at L=T. The per-window overhead is
defined by the learner formulas.
The axes are linear, with a common y range for both work panels.
`controlled_windowing_sweep.csv` exports the curves, profile source, dimensions,
window counts and time-step coverage. `configurations.json` records the source
length histogram. Actual validation-selected settings remain
in `selected_vs_full.csv` and the separate accuracy/memory and capacity figures.

The accuracy/memory figure uses open circles for full sequences and filled
diamonds for selected windows, connected by an arrow for each method. Published
device memory capacities provide reference lines, with source URLs exported to
`memory_budget_references.csv`. These are capacity ceilings, not measured free
memory or deployment claims. To allocate 75% of each reference capacity to the
modelled training working set, set `MEMORY_BUDGET_FRACTION` to `0.75` in the
report script. All configurations
retain their tested batch size, architecture and accuracy. Values and comparisons
are exported to `accuracy_memory.csv` and `memory_budget_comparison.csv`.

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

Saved experiments retain their original weighted time proxy,
`COST_COMPUTE_WEIGHT * compute + COST_ACCESS_WEIGHT * accesses`. The overview
now shows computation and memory-access counts separately, without equal-weight
time panels, and replaces the former coefficient sweep with
`05_fast_memory_capacity_sweep` for every dataset and all four learners.

The hierarchy figures now use weighted cycle proxies:
`alpha=1`, `beta_fast=1`, `beta_slow=1103614/102415` (about 10.78).
The memory ratio comes from matched 32-bit scalar copies of 100 KiB in
internal memory versus PSRAM in the pinned source:
https://github.com/project-x51/esp32-s3-memorycopy/blob/c3a1a5b5bc5372e301ed0f3eb15328857a95777d/README.md
The measured totals are 102,415 and 1,103,614 cycles; the shared 51,200-access
denominator cancels. The loop and cache-writeback behavior belong to this
benchmark scenario. This is not a universal PSRAM access penalty. Alpha=1 is
an independent simplifying assumption, not a DSP timing measurement.

With peak working set W and fast capacity K, `f=min(1,K/W)` and
`cost=alpha*C + (f*beta_fast+(1-f)*beta_slow)*M`. The slow coefficient is a
total cost, not an additional penalty; that penalty is beta_slow-beta_fast.
Uniform access frequency and proportional fixed placement are assumptions.
The model does not simulate caches, vectorization, or separate read/write
costs. No additional overhead or staging-transfer term is introduced.
The numerical values are proxy cycles: one per arithmetic model unit or fast access.
These weighted counts are distinct from measured processor cycles.

`05_fast_memory_capacity_sweep` varies log capacity with one fixed denominator
per dataset: full BPTT all-slow cost at the reference ratio. Open circles mark
each footprint. The grid includes all footprints and the 512 KiB nominal SRAM
reference, marked by a labeled vertical line in both capacity charts. Larger capacities are
hypothetical scenarios. Reference rows are exported to fast_memory_reference_costs.csv.

`07_windowed_cost_capacity_sweep` compares windowed cost with the same method's
full-sequence cost at each capacity. Each method uses its validation-selected
window length and hop. A ratio below
1 means fewer proxy cycles for that method. Each
setting uses its own calculated peak footprint and mean total work, including all
windows, overlap, gaps and actual tails. Work is never multiplied by window
count again; peak window memory is not summed. Reference rows are in
windowed_cost_reference_costs.csv, and curves in windowed_cost_capacity_sweep.csv.

memory_hierarchy_profile.json, memory_hierarchy_coefficients.csv and
memory_hierarchy_README.md record the exact counts, units, assumptions and
source URLs. Regenerating reports takes no training or evaluation passes.

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
`window_lengths` metadata produces valid time masks. Full sequences keep their
original minibatch, including recordings of different lengths. Independent
windows are flattened in sample order into minibatches of at most the configured
batch size; short final windows do not create separate length groups or extra
optimizer updates. A single window covering each whole sequence therefore uses
the same inputs, masks, and batching as its full-sequence baseline.

All four learners freeze ended recordings and exclude padding from readouts,
learning, and activity statistics. RATE-PEPITA divides rates by each recording's
valid length and calibrates feedback using only valid inputs from the whole
batch. Real silence within a recording remains valid. BPTT, E-prop, FF, and
RATE-PEPITA each update once per original recording minibatch. The optional original
timestep-wise PEPITA mode retains its separate update schedule.

Evaluation combines all window votes once per original sample. Partial-time
evaluation uses the requested fraction of each window's own valid length.
Each window's vote is weighted by its valid length. Partial-time votes use the
number of steps actually evaluated. A one-step tail therefore cannot tie a
100-step window merely by predicting a different class.

During learning, each window has weight equal to its valid length divided by
the sum of valid window lengths for its original recording. These weights sum
to one per recording, including overlapping schedules. They multiply each
window's learning contribution before batch averaging, and are not normalized
again within a minibatch. An unwindowed sequence or a single full-length window
has weight one. Run metadata records both weighting rules.

Window contributions are accumulated with model weights held fixed. Each
chunk contributes its weighted gradient sum divided by the original recording
batch size, including smaller final chunks. Adam takes one step after all windows
from that recording batch have been processed. This matches the full-sequence
update count without retaining temporal graphs across windows. The optional
original timestep-wise PEPITA mode retains its legacy schedule; thesis runs use
RATE-PEPITA accumulation mode. A single full-length window retains the original
unwindowed path exactly. The core analytical cost formulas remain unchanged;
Adam internals and gradient-accumulation bookkeeping are outside that model.

Analytical work is the sum of C(T_i) and M(T_i) for all actual window lengths,
not window_count times C(S) or M(S). Per-window update overhead still repeats.
Peak memory is the maximum per-window working set.
These estimates count valid algorithmic steps; masked rectangular execution can
still perform arithmetic for padded positions on the training device.

New runs record actual processed lengths and source_length_counts before
windowing. Their saved estimates document the calculations made during training.
Reporting derives the length distribution by inspecting the training inputs
before windowing. It applies the same window spans as the current transform.
For variable-length audio, the maximum comes from the full training split.

ESP32 cycle-proxy tables use `esp32_cycle_proxy.csv`. A configuration fits
nominal total memory when its peak footprint is at most 512 KiB + 8 MiB;
this is separate from fitting fast SRAM.
