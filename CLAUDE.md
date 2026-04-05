# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project comparing biologically-inspired learning algorithms for Spiking Neural Networks (SNNs) on time-series classification tasks. Implements four learning methods: Backpropagation (BPTT), Forward-Forward, E-Prop, and PEPITA — evaluated across 9 datasets (HAR, MNIST, Speech Commands, ESC-50, UrbanSound8K, PAMAP2, MIT-BIH, DVS Gesture, Large-Scale Audio).

## Commands

```bash
# Setup
pip install -r requirements.txt   # PyTorch with CUDA 12.1, snntorch, optuna, librosa, etc.

# Training — configure experiments in RUNS list, then run
python run_training.py

# Hyperparameter optimization (window length/hop via Optuna, two-phase)
python optuna_window_grid_two_phase.py

# Results analysis
python compare_results.py

# Dataset visualization
python visualize_datasets.py
```

No test suite exists. Validation is done by running training and checking results in `results/runs/{RUN_ID}/summary.json`.

## Architecture

### Training Flow

`run_training.py` defines experiment configs (dicts spreading `**DEFAULT`) → `utils/runner.py:run_one()` orchestrates: load dataset with transforms → build `NetConfig` → instantiate learner → training loop (`utils/training.py`) → save results (`visualization/training_results.py`).

### Learners (`learners/`)

All inherit from `BaseLearner` (`base.py`) which defines `forward()`, `train_step()`, `predict_batch()`, and memory estimation methods. Registered in `learners/registry.py` as `{"bp", "ff", "eprop", "pepita"}`.

- **BackpropLearner** — standard BPTT with configurable time aggregation (mean/sum/last)
- **FFLearner** — greedy layer-wise training with goodness scores
- **EpropLearner** — online learning with manual gradient updates (no autograd)
- **PepitaLearner** — two-pass update with feedback matrix and relative step control

### Network (`networks/`)

**SNNCore** (`snn_core.py`): LIF neuron layers with configurable surrogate gradients (fast_sigmoid, atan, triangular, etc.), optional recurrence, optional normalization (LayerNorm/BatchNorm/RMSNorm). Configured via `NetConfig` dataclass in `specs.py`.

### Dataset Pipeline (`timeseries/`)

`timeseries/registry.py` maps dataset names to loaders via `get_dataloaders()`. Each dataset module lives in `timeseries/datasets/`. Preprocessing uses composable transforms from `timeseries/transforms.py` (ZScore, ToLogMel, SlidingWindow, DeterministicSpikes, EventToVoxel, etc.) — transforms have a `fit()`/`__call__()` pattern.

### Results

Saved to `results/runs/{RUN_ID}/` with `summary.json` (full results) and `metrics.json` (flat metrics). Visualization tools in `visualization/`.

## Key Conventions

- **Experiment config**: each run is a dict spreading `**DEFAULT` and overriding specific fields (DATASET, LEARNER, EPOCHS, TRANSFORM, HIDDEN_SIZES, etc.). Dataset-specific kwargs go in `DATASET_KW`.
- **Transform pipelines**: defined per-dataset at the top of `run_training.py` (e.g., `HAR_PIPELINE`, `SC_PIPELINE`). Audio datasets use `ToLogMel`; neuromorphic uses `EventToVoxel`.
- **Energy estimation**: computed as synops × energy_per_synop (0.9 pJ for fp32, 0.4 pJ for fp16). Supports fp16/bf16 eval and int8 weight quantization.
- **Learner hyperparameters** are prefixed: `BP_` (backprop), `FF_` (forward-forward), `EP_` (e-prop), `PEP_` (PEPITA).
- `data/` and `results/` are gitignored. Datasets auto-download on first use.
- `experimental/` contains legacy/prototype algorithm variants — not part of the main pipeline.
