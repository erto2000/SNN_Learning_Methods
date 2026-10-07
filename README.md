# SNN Learning Methods

The project workflow, experiment protocol, and dataset settings are documented in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md). Keep detailed project notes there so the repository root stays tidy.

Edit shared settings in `experiment_config.py`, then run `run_no_window.py` for standard experiments or an Optuna script for window search. Training entry points use configuration in the source files. Comparison and reporting modules are in `reporting/`, and dataset visualization helpers are in `visualization/`. The chart and dataset-visualization entry points accept command-line arguments; use `--help` to see their options. Full training and Optuna searches can be expensive, so check configuration before running a full campaign.
