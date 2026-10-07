"""Inspect the datasets and preprocessing used by the experiment runs.

Run ``python -m reporting.visualize_datasets`` from the project root, or run
this file directly from an IDE. No model is trained.
Writes individual raw/processed comparisons, a combined figure for all selected
datasets, and a combined HAR/audio/events/ECG/rate-coded-image figure (PNG/SVG).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment_config import DATASET_CONFIGS, DEFAULT
from visualization.dataset_inspector import build_dataset_viz
from visualization.dataset_visualization import save_combined_examples


CONFIGS = {config["RUN_ID_PREFIX"]: config for config in DATASET_CONFIGS}
TYPE_EXAMPLES = ["har", "sc", "dvs", "mitbih", "mnist-rate"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=CONFIGS,
                        default=list(CONFIGS), metavar="DATASET",
                        help="IDs to inspect (default: all): " + ", ".join(CONFIGS))
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Optional cap for a quicker inspection run")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "results",
                        help="Base output directory (default: project results folder)")
    parser.add_argument("--tag", default="all_datasets_overview",
                        help="Subdirectory name under dataset_visualization")
    parser.add_argument("--type-datasets", nargs="+", choices=CONFIGS,
                        default=TYPE_EXAMPLES, metavar="DATASET",
                        help="Order for the dataset-type combined figure (uses inspected IDs only; "
                             "default: har sc dvs mitbih mnist-rate)")
    args = parser.parse_args(argv)
    if args.max_samples is not None and args.max_samples < 3:
        parser.error("--max-samples must be at least 3")
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("Each dataset ID should appear only once")
    if len(set(args.type_datasets)) != len(args.type_datasets):
        parser.error("Each dataset-type ID should appear only once")

    output = args.output_dir.resolve()
    data_root = (PROJECT_ROOT / DEFAULT["DATA_ROOT"]).resolve()
    examples = {}
    overview = ["# Dataset example figures", "",
                "Each figure pairs two illustrative training records with their",
                "processed versions. The same record is shown in both columns.", ""]
    for dataset_id in args.datasets:
        config = CONFIGS[dataset_id]
        configured_cap = config.get("MAX_SAMPLES", DEFAULT["MAX_SAMPLES"])
        cap = (min(configured_cap, args.max_samples)
               if configured_cap is not None and args.max_samples is not None
               else configured_cap if args.max_samples is None else args.max_samples)
        print(f"Inspecting {dataset_id} (max_samples={cap})...", flush=True)
        artifacts = build_dataset_viz(
            ID=dataset_id,
            DATASET=config["DATASET"],
            SPLITS=["train", "validation", "test"],
            DATA_ROOT=str(data_root),
            MAX_SAMPLES=cap,
            TRANSFORM=config["TRANSFORM"],
            NOTES="Pipeline and selection from experiment_config.py",
            SEED=DEFAULT["SEED"],
            DATA_SPLIT=dict(DEFAULT["DATA_SPLIT"]),
            DATASET_KWARGS=config.get("DATASET_KW"),
            base_dir=str(output),
            tag=args.tag,
            INCLUDE_COMBINED_EXAMPLE=True,
        )
        examples[dataset_id] = artifacts["combined_example"]
        print(f"Saved {dataset_id}: {artifacts['out_dir']}", flush=True)
        caption = (Path(artifacts["out_dir"]) / "thesis" / "caption.txt").read_text(
            encoding="utf-8").strip()
        overview.extend([f"## {dataset_id}", "",
                         f"[PNG]({dataset_id}/thesis/representative_examples.png) · "
                         f"[SVG]({dataset_id}/thesis/representative_examples.svg)",
                         "", caption, ""])
    report_root = output / "dataset_visualization" / args.tag
    combined = [("combined_all_datasets", list(examples), "One example from each dataset"),
                ("combined_dataset_types", [key for key in args.type_datasets if key in examples],
                 "Examples across dataset types")]
    links = ["## Combined figures", "",
             "One illustrative training record per row; raw input and processed "
             "representation use the same record. All HAR and PAMAP2 channels are shown.", ""]
    provenance = {}
    for stem, dataset_ids, title in combined:
        if not dataset_ids:
            continue
        path = report_root / f"{stem}.png"
        save_combined_examples([examples[key] for key in dataset_ids], str(path), title)
        links.extend([f"{title}: [PNG]({stem}.png) · [SVG]({stem}.svg)", ""])
        provenance[stem] = [
            {"dataset_id": key, "train_index": examples[key]["train_index"],
             "class": examples[key]["label"]} for key in dataset_ids]
        print(f"Saved combined figure: {path}", flush=True)
    (report_root / "combined_selection.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8")
    overview[5:5] = links
    (report_root / "THESIS_FIGURES.md").write_text(
        "\n".join(overview), encoding="utf-8")


if __name__ == "__main__":
    main()
