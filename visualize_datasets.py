# visualize_datasets.py
"""
Standalone dataset visualization (no training).
- Edit VIS list below (like RUNS in main.py).
- Run:  python scripts/visualize_datasets.py
Artifacts go to: results/_dataset_visualization/<TAG>/<ID>/
"""

from visualization.dataset_inspector import build_dataset_viz
import os

# ──────────────────────────────────────────────────────────────────────────────
# DEFAULTS (edit as needed)
BASE_DIR = "results"
TAG = "har_exploration"
SEED = 123

# Pipelines: reuse the same ones you have in main.py if you want post-pipeline views
import timeseries.transforms as transforms
L, H = 64, 64
HAR_PIPELINE = transforms.Compose([
    transforms.ToFloat32(),
    transforms.Ensure2D(),
    transforms.SlidingWindow(length=L, hop=H),
])
SC_PIPELINE = transforms.Compose([
    transforms.ToFloat32(),
    transforms.Ensure2D(),
    transforms.ToLogMel(sample_rate=16000, n_mels=64, win_len_ms=25, hop_ms=10),
])
MNIST_STATIC_PIPELINE = transforms.Compose([
    transforms.ToFloat32(),
    transforms.RepeatStatic(T=50),
])
MNIST_RATE_PIPELINE = transforms.Compose([
    transforms.ToFloat32(),
    transforms.DeterministicSpikes(gain=0.7, T=20, base_seed=0),
])

# VIS “jobs”: each is an inspection you want to run (raw and/or post)
VIS = [
    dict(
        ID="har-raw",
        DATASET="har",
        SPLITS=["train", "test"],
        DATA_ROOT="./data",
        MAX_SAMPLES=2000,
        TRANSFORM=None,
        NOTES="HAR raw signals",
        SEED=SEED,
    ),
    dict(
        ID="har-post-L128H64",
        DATASET="har",
        SPLITS=["train", "test"],
        DATA_ROOT="./data",
        MAX_SAMPLES=2000,
        TRANSFORM=HAR_PIPELINE,
        NOTES="HAR after SlidingWindow(L=128, H=64)",
        SEED=SEED,
    ),
    dict(ID="sc-post-mels64", DATASET="speech_commands", SPLITS=["train","test"], DATA_ROOT="./data",
         MAX_SAMPLES=4000, TRANSFORM=SC_PIPELINE, NOTES="SC log-mel", SEED=SEED),
    dict(ID="mnist-static", DATASET="mnist", SPLITS=["train","test"], DATA_ROOT="./data",
         MAX_SAMPLES=2000, TRANSFORM=MNIST_STATIC_PIPELINE, NOTES="MNIST repeated static", SEED=SEED),
    dict(ID="mnist-rate", DATASET="mnist", SPLITS=["train","test"], DATA_ROOT="./data",
         MAX_SAMPLES=2000, TRANSFORM=MNIST_RATE_PIPELINE, NOTES="MNIST rate-coded spikes", SEED=SEED),
]
# ──────────────────────────────────────────────────────────────────────────────

def main():
    for job in VIS:
        artifacts = build_dataset_viz(base_dir=BASE_DIR, tag=TAG, **job)
        print("\n===== Dataset Visualization =====")
        print(f"ID: {job['ID']}  |  out: {artifacts['out_dir']}")

        # Figures created
        for k, v in artifacts["figs"].items():
            if v: print(f"- {k}: {v}")

        # Quick table locations
        corpus = os.path.join(artifacts["out_dir"], "corpus")
        print(f"- per-split counts (csv/json) in: {corpus}")
        print(f"- overall counts: {os.path.join(corpus,'class_counts_overall.csv')}")

        # Summary
        print(f"- summary: {artifacts['summary_path']}")

if __name__ == "__main__":
    main()
