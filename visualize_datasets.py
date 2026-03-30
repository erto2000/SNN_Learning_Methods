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

# HAR
HAR_PIPELINE = transforms.Compose([
    transforms.ToFloat32(),
    transforms.Ensure2D(),
    transforms.ZScore(),
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

# Speech Commands
SC_PIPELINE = transforms.Compose([
    transforms.ToFloat32(),
    transforms.Ensure2D(),                                 # waveform -> [T,1]
    transforms.ToLogMel(sample_rate=16000, n_mels=64, win_len_ms=25, hop_ms=10),         # -> [F,M]
    transforms.ZScore(),
])

# ESC-50: 5s @44100Hz → log-mel(64)
ESC50_PIPELINE = transforms.Compose([
    transforms.ToFloat32(),
    transforms.Ensure2D(),
    transforms.ToLogMel(sample_rate=44100, n_mels=64, win_len_ms=25, hop_ms=10),
    transforms.ZScore(),
])

# UrbanSound8K: 1s-4s @44100Hz → log-mel(64)
URBAN8K_PIPELINE = transforms.Compose([
    transforms.ToFloat32(),
    transforms.Ensure2D(),
    transforms.ToLogMel(sample_rate=44100, n_mels=64, win_len_ms=25, hop_ms=10),
    transforms.ZScore(),
])

# PAMAP2 (IMU)
PAMAP2_PIPELINE = transforms.Compose([
    transforms.ToFloat32(),
    transforms.ZScore(),
])

# MIT-BIH (ECG)
MITBIH_PIPELINE = transforms.Compose([
    transforms.ToFloat32(),
    transforms.ZScore(),
])

# Large-Scale Audio: 3s @16kHz → log-mel(64)
LARGE_SCALE_AUDIO_PIPELINE = transforms.Compose([
    transforms.ToFloat32(),
    transforms.ToLogMel(sample_rate=16000, n_mels=64, win_len_ms=25, hop_ms=10),
    transforms.ZScore(),
])

# DVS128 Gesture (neuromorphic events)
DVS_GESTURE_PIPELINE = transforms.Compose([
    transforms.DownsampleEvents(factor=4),
    transforms.EventToVoxel(H=32, W=32, bins=200, polarity=True),
    transforms.ZScore(),
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
    # ── Speech Commands
    # dict(ID="sc",
    #      DATASET="speech_commands",
    #      SPLITS=["train","test"],
    #      DATA_ROOT="./data",
    #      MAX_SAMPLES=2000,
    #      TRANSFORM=SC_PIPELINE,
    #      DATASET_KW={"class_count":10, "equal_per_class":True},
    #      NOTES="Speech Commands log-mel + ZScore",
    #      SEED=SEED),

    # # ── ESC-50 (environmental audio)
    # dict(ID="esc50",
    #      DATASET="esc50", SPLITS=["train","test"], DATA_ROOT="./data",
    #      MAX_SAMPLES=2000,
    #      TRANSFORM=ESC50_PIPELINE,
    #      DATASET_KW={"class_count": 10, "equal_per_class": True, "time_steps": 44100},
    #      NOTES="ESC-50 5s crops → log-mel(64)",
    #      SEED=SEED),

    # # ── UrbanSound8K (urban audio)
    # dict(ID="urban8k",
    #      DATASET="urban8k",
    #      SPLITS=["train","test"],
    #      DATA_ROOT="./data",
    #      MAX_SAMPLES=2000,
    #      TRANSFORM=URBAN8K_PIPELINE,
    #      DATASET_KW={"equal_per_class": True},
    #      NOTES="Urban8K 1s-4s crops → log-mel(64)",
    #      SEED=SEED),
    #
    # # ── PAMAP2 (IMU)
    # dict(ID="pamap2",
    #      DATASET="pamap2",
    #      SPLITS=["train","test"],
    #      DATA_ROOT="./data",
    #      MAX_SAMPLES=None,
    #      TRANSFORM=PAMAP2_PIPELINE,
    #      NOTES="PAMAP2 ZScore + SlidingWindow(128,64)",
    #      DATASET_KW={"equal_per_class": True, "time_steps": 128},
    #      SEED=SEED),
    #
    # # ── MIT-BIH (ECG)
    # dict(ID="mitbih",
    #      DATASET="mitbih",
    #      SPLITS=["train","test"],
    #      DATA_ROOT="./data",
    #      MAX_SAMPLES=10000,
    #      TRANSFORM=MITBIH_PIPELINE,
    #      DATASET_KW={"two_class": True, "equal_per_class": True},
    #      NOTES="MIT-BIH ZScore + 1s windows (360) hop 180",
    #      SEED=SEED),

    # # ── DVS128 Gesture (neuromorphic)
    # dict(ID="dvs-post",
    #      DATASET="dvs_gesture", SPLITS=["train","test"], DATA_ROOT="./data",
    #      MAX_SAMPLES=400, TRANSFORM=DVS_GESTURE_PIPELINE,
    #      NOTES="DVS events → voxel (128×128×bins) → SlidingWindow(50,25) + ZScore",
    #      SEED=SEED),

    # # ── Large-Scale Audio (emergency siren vs road noise)
    # dict(ID="large-scale-audio",
    #      DATASET="large_scale_audio",
    #      SPLITS=["train","test"],
    #      DATA_ROOT="./data",
    #      MAX_SAMPLES=2000,
    #      TRANSFORM=LARGE_SCALE_AUDIO_PIPELINE,
    #      DATASET_KW={"duration": 3.0, "equal_per_class": True},
    #      NOTES="Large-Scale Audio 3s crops → log-mel(64)",
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
            artifacts = build_dataset_viz(
                base_dir=BASE_DIR,
                tag=TAG,
                DATASET_KWARGS=dataset_kw,
                **job,
            )
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