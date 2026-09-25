"""Run every configured learner on full, unwindowed sequences."""

from experiment_config import RUNS
from utils.runner import run_one, summarize
from visualization.training_results import save_results


RESULTS_DIR = "results"


def main():
    results = []
    for run in RUNS:
        result = run_one(run)
        save_results([result], base_dir=RESULTS_DIR)
        results.append(result)
    summarize(results)


if __name__ == "__main__":
    main()
