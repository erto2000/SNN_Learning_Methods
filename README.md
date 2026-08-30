# Memory-Efficient Spiking Neural Network Training with Backpropagation-Free Learning Rules

Code for the paper by Ertuğrul Keskin and Arda Yurdakul (Boğaziçi University), presented at the [29th Euromicro Conference on Digital System Design (DSD 2026)](https://dsd-seaa.com/), Kraków, Poland, September 2–4, 2026.

Code written and maintained by Ertuğrul Keskin. Research supervised by Arda Yurdakul.

📄 Paper: [Not Published Yet](https://ieeexplore.ieee.org/)

## What this is

Training an SNN with backpropagation through time means storing every neuron's state at every time step, so training memory grows with sequence length. This repository implements four learning rules under one shared shallow SNN and one evaluation protocol, and compares their accuracy against their training-memory footprint:

| Learner | Key | Stored during training |
|---|---|---|
| BPTT | `bp` | Full neuron state at every time step |
| Forward-Forward | `ff` | Only the layer currently being trained |
| E-PROP | `eprop` | One eligibility trace per synapse |
| **RATE-PEPITA** (proposed) | `pepita` | Accumulated spike counts per neuron |

**RATE-PEPITA** is this work's contribution: an SNN adaptation of PEPITA that replaces the continuous ANN activations of the original with time-averaged spike rates. Two forward passes, no backward pass, no temporal history, no per-synapse traces — memory scales only with the number of neurons.

The repository also implements **input-level temporal windowing**, where each sequence is split into shorter segments before training. Window length and hop ratio are selected per dataset by a two-phase TPE search with Optuna.

## Results

Averaged over eight datasets, relative to BPTT without windowing:

**Without windowing**

| Method | Mean accuracy Δ (pp) | Mean memory saving (%) |
|---|---|---|
| BPTT | — | — |
| FF | −3.6 | −1.5 |
| E-PROP | −8.4 | +60.6 |
| RATE-PEPITA | −22.7 | +76.7 |

**With temporal windowing**

| Method | Mean accuracy Δ (pp) | Mean memory saving (%) |
|---|---|---|
| BPTT | +4.8 | +58.9 |
| FF | +5.0 | +65.9 |
| E-PROP | +1.1 | +79.2 |
| RATE-PEPITA | −6.4 | **+94.0** |

Windowing improves accuracy for every learning rule and increases the memory savings. Per-dataset numbers are in the paper. Memory figures are analytical estimates from the hardware-agnostic model in Section III-C, not hardware measurements.

## Setup

```bash
git clone https://github.com/erto2000/SNN_Learning_Methods.git
cd SNN_Learning_Methods
pip install -r requirements.txt
```

`requirements.txt` pins CUDA 12.1 builds of `torch`, `torchaudio` and `torchvision`. For a CPU-only machine, install those three from the default PyPI index instead.

## Usage

Experiments are configured in code, not through command-line flags.

**Training.** Add or uncomment a run in the `RUNS` list of `run_training.py`, then:

```bash
python run_training.py
```

Each run entry sets `DATASET`, `LEARNER`, the transform pipeline and any overrides on top of `DEFAULT`:

```python
{
    **DEFAULT,
    "RUN_ID":    "har-pepita",
    "DATASET":   "har",
    "LEARNER":   "pepita",
    "TRANSFORM": HAR_PIPELINE,
}
```

Datasets: `har`, `pamap2`, `mitbih`, `dvs_gesture`, `esc50`, `urban8k`, `speech_commands`, `large_scale_audio`, `mnist`.

**Windowing search.** The two-phase TPE search — 20 trials × 5 epochs broad, then 10 trials × 10 epochs refined around the best point — reads its baseline config from `run_training.RUNS`, so the dataset must have an active entry there:

```bash
python optuna_window_independent_two_phase.py   # window length, then hop ratio
python optuna_window_grid_two_phase.py          # joint grid variant
```

**Memory model and figures.**

```bash
python calculate_memory_requirements.py   # analytical footprints + scaling plots
python compare_results.py                 # aggregate results across runs
python visualize_datasets.py              # dataset inspection plots
```

Results are written to `results/` and `optuna_results/`, both git-ignored.

## Layout

```
run_training.py          entry point; run configs live in RUNS
learners/                bp, ff, eprop, pepita update rules
networks/                LIF network, layer specs, int8 eval
timeseries/              dataset builders, transforms, windowing
utils/                   runner, training loop, segment handling
visualization/           plots and result tables
experimental/            earlier prototypes, not part of the paper
```

## Setup used in the paper

One hidden layer with 128 LIF neurons, leak 0.9, threshold 1.0; direct encoding; 10 epochs, batch size 128, FP32, seed 123; fast-sigmoid surrogate gradient with slope 25 for BPTT; learner-specific learning rates; class-balanced splits, with a fixed class subset on some datasets so one architecture fits all.

## Citation

```bibtex
@inproceedings{keskin2026ratepepita,
  title     = {Memory-Efficient Spiking Neural Network Training with Backpropagation-Free Learning Rules},
  author    = {Keskin, Ertu{\u{g}}rul and Yurdakul, Arda},
  booktitle = {Proceedings of the 29th Euromicro Conference on Digital System Design (DSD)},
  address   = {Krak{\'o}w, Poland},
  year      = {2026}
}
```

## License

Code released under the [MIT License](LICENSE). The paper is © IEEE; see the links above for the published version.
