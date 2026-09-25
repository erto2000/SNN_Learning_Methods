# SNN Learning Methods

The project workflow, experiment protocol, and dataset settings are documented in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md). Keep detailed project notes there so the repository root stays tidy.

Edit shared settings in `experiment_config.py`, then run `run_no_window.py` for standard experiments or an Optuna script for window search. Comparison and reporting modules are in `reporting/`, and dataset visualization is in `visualization/`. The entry points use configuration in the source files rather than command-line arguments. Full training and Optuna searches can be expensive, so check configuration before running a full campaign.
