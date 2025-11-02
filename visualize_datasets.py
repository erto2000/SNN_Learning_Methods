import os
from visualization.dataset_inspector import build_dataset_viz
import timeseries.transforms as transforms

# ──────────────────────────────────────────────────────────────────────────────
# DEFAULTS (edit as needed)
BASE_DIR = "results"
TAG = "all_datasets_overview"
SEED = 123

# ──────────────────────────────────────────────────────────────────────────────
# PIPELINES — match exactly what we use in run_training.py

# HAR: fixed windows
L = 128
H = 64
HAR_PIPELINE = transforms.Compose([
    transforms.ToFloat32(),
    transforms.Ensure2D(),
    transforms.SlidingWindow(length=L, hop=H),   # [T,D] -> [S,L,D]
])

# MNIST: static image to temporal sequence
MNIST_STATIC_PIPELINE = transforms.Compose([
    transforms.ToFloat32(),
    transforms.RepeatStatic(T=50),
])

# MNIST: rate-coded spikes
MNIST_RATE_PIPELINE = transforms.Compose([
    transforms.ToFloat32(),
    transforms.DeterministicSpikes(gain=0.7, T=20, base_seed=0),  # [1,784] -> [20,784]
])

# Speech Commands: waveform → log-mel → z-score
SC_PIPELINE = transforms.Compose([
    transforms.ToFloat32(),
    transforms.Ensure2D(),                                 # waveform -> [T,1]
    transforms.ToLogMel(sample_rate=16000, n_mels=64,
                        win_len_ms=25, hop_ms=10),         # -> [F,M]
    transforms.ZScore(),
])

# Audio (ESC-50 / UrbanSound8K): 4s @16k → log-mel(64)
AUDIO_SR = 16000
AUDIO_T  = AUDIO_SR * 4  # 4 seconds
ESC50_PIPELINE = transforms.Compose([
    transforms.ToFloat32(),
    transforms.Ensure2D(),
    transforms.RandomTimeCrop(AUDIO_T),
    transforms.ToLogMel(sample_rate=AUDIO_SR, n_mels=64, win_len_ms=25, hop_ms=10),
    transforms.ZScore(),   # auto-fit on train
])
URBAN8K_PIPELINE = ESC50_PIPELINE  # identical defaults

# PAMAP2 (IMU): shorter windows to keep VRAM low
PAMAP2_PIPELINE = transforms.Compose([
    transforms.ToFloat32(),
    transforms.ZScore(),
    transforms.SlidingWindow(length=128, hop=64),   # lighter than 256/128
])

# MIT-BIH (ECG): 1s@360Hz windows with overlap
MITBIH_PIPELINE = transforms.Compose([
    transforms.ToFloat32(),
    transforms.ZScore(),
    transforms.SlidingWindow(length=360, hop=180),
])

# DVS128 Gesture (neuromorphic events): events → voxel → short windows
DVS_GESTURE_PIPELINE = transforms.Compose([
    transforms.EventToVoxel(H=128, W=128, bins=200, polarity=True),  # -> [T, H*W*2]
    transforms.ZScore(),
    transforms.SlidingWindow(length=50, hop=25),
])

# ──────────────────────────────────────────────────────────────────────────────
# VIS “jobs”: each is an inspection you want to run (raw and/or post)
# Keep caps modest so runs are quick; bump MAX_SAMPLES if you want fuller views.

VIS = [
    # # ── HAR (Inertial)
    # dict(ID="har-raw",
    #      DATASET="har", SPLITS=["train", "test"], DATA_ROOT="./data",
    #      MAX_SAMPLES=2000, TRANSFORM=None, NOTES="HAR raw signals", SEED=SEED),
    # dict(ID="har-post-L128H64",
    #      DATASET="har", SPLITS=["train", "test"], DATA_ROOT="./data",
    #      MAX_SAMPLES=2000, TRANSFORM=HAR_PIPELINE, NOTES="HAR after SlidingWindow(L=128, H=64)", SEED=SEED),
    #
    # # ── MNIST (two views)
    # dict(ID="mnist-static",
    #      DATASET="mnist", SPLITS=["train","test"], DATA_ROOT="./data",
    #      MAX_SAMPLES=2000, TRANSFORM=MNIST_STATIC_PIPELINE, NOTES="MNIST repeated static", SEED=SEED),
    # dict(ID="mnist-rate",
    #      DATASET="mnist", SPLITS=["train","test"], DATA_ROOT="./data",
    #      MAX_SAMPLES=2000, TRANSFORM=MNIST_RATE_PIPELINE, NOTES="MNIST rate-coded spikes", SEED=SEED),
    #
    # # ── Speech Commands
    # dict(ID="sc-post-mels64",
    #      DATASET="speech_commands", SPLITS=["train","test"], DATA_ROOT="./data",
    #      MAX_SAMPLES=4000, TRANSFORM=SC_PIPELINE, NOTES="Speech Commands log-mel + ZScore", SEED=SEED),
    #
    # # ── ESC-50 (environmental audio)
    # dict(ID="esc50-post-mels64",
    #      DATASET="esc50", SPLITS=["train","test"], DATA_ROOT="./data",
    #      MAX_SAMPLES=2000, TRANSFORM=ESC50_PIPELINE, NOTES="ESC-50 4s crops → log-mel(64) + ZScore", SEED=SEED),
    #
    # # ── UrbanSound8K (urban audio)
    # dict(ID="urban8k-post-mels64",
    #      DATASET="urban8k", SPLITS=["train","test"], DATA_ROOT="./data",
    #      MAX_SAMPLES=2000, TRANSFORM=URBAN8K_PIPELINE, NOTES="Urban8K 4s crops → log-mel(64) + ZScore", SEED=SEED),
    #
    # # ── PAMAP2 (IMU)
    # dict(ID="pamap2-post",
    #      DATASET="pamap2", SPLITS=["train","test"], DATA_ROOT="./data",
    #      MAX_SAMPLES=None, TRANSFORM=PAMAP2_PIPELINE,
    #      NOTES="PAMAP2 ZScore + SlidingWindow(128,64)",
    #      SEED=SEED),
    #
    # # ── MIT-BIH (ECG)
    # dict(ID="mitbih-post",
    #      DATASET="mitbih", SPLITS=["train","test"], DATA_ROOT="./data",
    #      MAX_SAMPLES=2000, TRANSFORM=MITBIH_PIPELINE,
    #      NOTES="MIT-BIH ZScore + 1s windows (360) hop 180",
    #      SEED=SEED),
    #
    # # ── DVS128 Gesture (neuromorphic)
    # dict(ID="dvs-post",
    #      DATASET="dvs_gesture", SPLITS=["train","test"], DATA_ROOT="./data",
    #      MAX_SAMPLES=400, TRANSFORM=DVS_GESTURE_PIPELINE,
    #      NOTES="DVS events → voxel (128×128×bins) → SlidingWindow(50,25) + ZScore",
    #      SEED=SEED),
]

# If you need to tweak dataset-specific kwargs (like you do in run_training.py),
# you can add a DATASET_KW dict to any job above, e.g.:
# dict(..., DATASET_KW={"train_subjects":[101,...], "test_subjects":[109], "min_len":100})

# ──────────────────────────────────────────────────────────────────────────────

def main():
    for job in VIS:
        # Split out optional DATASET_KW (build_dataset_viz ignores unknown kwargs)
        dataset_kw = job.pop("DATASET_KW", None)
        if dataset_kw:
            artifacts = build_dataset_viz(base_dir=BASE_DIR, tag=TAG, **job, **dataset_kw)
        else:
            artifacts = build_dataset_viz(base_dir=BASE_DIR, tag=TAG, **job)

        print("\n===== Dataset Visualization =====")
        print(f"ID: {job['ID']}  |  out: {artifacts['out_dir']}")

        # Figures created
        for k, v in artifacts["figs"].items():
            if v:
                print(f"- {k}: {v}")

        # Quick table locations
        corpus = os.path.join(artifacts["out_dir"], "corpus")
        print(f"- per-split counts (json) in: {corpus}")
        print(f"- overall counts json: {os.path.join(corpus,'class_counts_overall.json')}")

        # Summary
        print(f"- summary: {artifacts['summary_path']}")

if __name__ == "__main__":
    main()