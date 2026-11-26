# SNN Learning Methods

The project workflow, experiment protocol, and dataset settings are documented in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md). Keep detailed project notes there so the repository root stays tidy.

Edit shared settings in `experiment_config.py`, then run `run_no_window.py` for standard experiments or an Optuna script for window search. Training entry points use configuration in the source files. Comparison and reporting modules are in `reporting/`, and dataset visualization helpers are in `visualization/`. Full training and Optuna searches can be expensive, so check configuration before running a full campaign.

Run `python reporting/create_results_overview.py` to regenerate the theoretical charts. The report includes accuracy and validation-selected window comparisons when the experiment records are available. Open `results/results_overview/index.html` for the figures and CSV exports.
