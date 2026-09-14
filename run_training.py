# run_training.py
from utils.runner import run_one, summarize
from visualization.training_results import save_results
import timeseries.transforms as transforms

# ──────────────────────────────────────────────────────────────────────────────
# DATASET PIPELINES

# HAR (128 timesteps, 9 channels)
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

# Speech Commands (101 timesteps, 64 channels)
SC_PIPELINE = transforms.Compose([
    transforms.ToFloat32(),
    transforms.Ensure2D(),                                 # waveform -> [T,1]
    transforms.ToLogMel(sample_rate=16000, n_mels=64, win_len_ms=25, hop_ms=10),         # -> [F,M]
    transforms.ZScore(),
])

# ESC-50: 5s @44100Hz → log-mel(64) (101 timesteps, 64 channels)
ESC50_PIPELINE = transforms.Compose([
    transforms.ToFloat32(),
    transforms.Ensure2D(),
    transforms.ToLogMel(sample_rate=44100, n_mels=64, win_len_ms=25, hop_ms=10),
    transforms.ZScore(),
])

# UrbanSound8K: 1s-4s @44100Hz → log-mel(64) (401 timesteps, 64 channels)
URBAN8K_PIPELINE = transforms.Compose([
    transforms.ToFloat32(),
    transforms.Ensure2D(),
    transforms.ToLogMel(sample_rate=44100, n_mels=64, win_len_ms=25, hop_ms=10),
    transforms.ZScore(),
])

# PAMAP2 (IMU) (128 timesteps, 27 channels)
PAMAP2_PIPELINE = transforms.Compose([
    transforms.ToFloat32(),
    transforms.ZScore(),
])

# MIT-BIH (ECG) (360 timesteps, 1 channel)
MITBIH_PIPELINE = transforms.Compose([
    transforms.ToFloat32(),
    transforms.ZScore(),
])

# DVS128 Gesture (neuromorphic events) (200 timesteps, 2048 channels)
DVS_GESTURE_PIPELINE = transforms.Compose([
    transforms.DownsampleEvents(factor=4),
    transforms.EventToVoxel(H=32, W=32, bins=200, polarity=True),
    transforms.ZScore(),
])

# Large-Scale Audio Dataset (901 timesteps, 3s segments -> log-mel 64)
LARGE_SCALE_AUDIO_PIPELINE = transforms.Compose([
    transforms.ToFloat32(),
    transforms.ToLogMel(sample_rate=16000, n_mels=64, win_len_ms=25, hop_ms=10),
    transforms.ZScore(),
])

# ──────────────────────────────────────────────────────────────────────────────
# DEFAULTS (tunable per run)
DEFAULT = dict(
    # Setup
    DATA_ROOT                   = "./data",
    NUM_WORKERS                 = 0,        # subprocesses for data loading; 0 loads data in main process
    PIN_MEMORY                  = False,    # speeds CPU-to-GPU transfer when using CUDA
    SEED                        = 123,
    BATCH_SIZE                  = 128,
    EPOCHS                      = 10,       # safe default; each run overrides up to max 10
    MAX_SAMPLES                 = None,
    DTYPE                       = "fp32",   # "fp32", "fp16", "bf16" for both train + eval
    THEORY_ALPHA                = 1.0,      # arithmetic-cost coefficient for theoretical time proxy
    THEORY_BETA                 = 1.0,      # memory-access coefficient for theoretical time proxy

    # Evaluation
    TEST_EVERY_EPOCH            = False,
    EVAL_INT8_WEIGHTS           = False,    # weight-only int8 for evaluation
    TIME_EVAL                   = True,
    TIME_EVAL_FRACS             = (0.25, 0.50, 1.0),
    TIME_EVAL_INCLUDE_T1        = True,

    # Network
    HIDDEN_SIZES     = [128],
    BETA             = 0.9,
    SPIKE_GRAD       = "fast_sigmoid",
    SLOPE            = 25.0,
    THRESHOLD        = 1.0,
    HEAD             = "logits",
    RECURRENT        = False,
    INIT_TYPE        = "default",
    NORM             = None,
    HIDDEN_BIAS      = False,
    HEAD_BIAS        = False,

    # Backprop
    BP_AGG           = "mean",
    BP_LR            = 2e-3,
    BP_OPTIMIZER     = "adam",

    # FF
    FF_ALPHA         = 0.6,
    FF_LR            = 2e-3,
    FF_OPTIMIZER     = "adam",

    # E-Prop
    EP_LR_IN         = 1e-3,
    EP_LR_REC        = 1e-3,
    EP_LR_OUT        = 2e-3,
    EP_OPTIMIZER     = "adam",
    EP_DROP_DIAG     = True,
    EP_WEIGHT_CLIP   = 1.5,

    # PEPITA
    PEP_MODE         = "accum",
    PEP_LR           = 1e-2,
    PEP_OPTIMIZER    = "adam",
    PEP_MAX_REL_STEP = 0.05,
    PEP_MOD_RATIO    = 0.1,
)

# ──────────────────────────────────────────────────────────────────────────────
# EXPERIMENTS
#
# Keep the existing per-dataset configuration in one place, then expand it over
# every learner/method below. This avoids maintaining separate hand-written RUNS
# entries for each dataset × method combination.

METHODS = (
    "bp",
    "ff",
    "eprop",
    "pepita",
)

DATASET_CONFIGS = [
    # ── HAR (Human Activity Recognition) ─────────────────────────────────────
    {
        "RUN_ID_PREFIX": "har",
        "DATASET": "har",
        "EPOCHS": 10,
        "BATCH_SIZE": 128,
        "HIDDEN_SIZES": [128],
        "TRANSFORM": HAR_PIPELINE,
    },

    # ── MNIST (static repeated frames) ───────────────────────────────────────
    {
        "RUN_ID_PREFIX": "mnist-static",
        "DATASET": "mnist",
        "EPOCHS": 10,
        "BATCH_SIZE": 128,
        "HIDDEN_SIZES": [128],
        "TRANSFORM": MNIST_STATIC_PIPELINE,
    },

    # ── Speech Commands ──────────────────────────────────────────────────────
    {
        "RUN_ID_PREFIX": "sc",
        "DATASET": "speech_commands",
        "EPOCHS": 10,
        "BATCH_SIZE": 128,
        "HIDDEN_SIZES": [128],
        "MAX_SAMPLES": 10000,
        "TRANSFORM": SC_PIPELINE,
        "DATASET_KW": {
            "class_filter": ["yes", "no", "stop"],
            "equal_per_class": True,
        },
    },

    # ── ESC-50 (environmental audio) ─────────────────────────────────────────
    {
        "RUN_ID_PREFIX": "esc50",
        "DATASET": "esc50",
        "EPOCHS": 10,
        "BATCH_SIZE": 128,
        "HIDDEN_SIZES": [128],
        "TRANSFORM": ESC50_PIPELINE,
        "DATASET_KW": {
            "class_filter": ["dog", "rain", "siren", "helicopter"],
            "equal_per_class": True,
            "duration": 1,
        },
    },

    # ── UrbanSound8K (urban audio) ───────────────────────────────────────────
    {
        "RUN_ID_PREFIX": "urban8k",
        "DATASET": "urban8k",
        "EPOCHS": 10,
        "BATCH_SIZE": 128,
        "HIDDEN_SIZES": [128],
        "TRANSFORM": URBAN8K_PIPELINE,
        "DATASET_KW": {
            "class_filter": ["dog_bark", "siren", "gun_shot"],
            "equal_per_class": True,
        },
    },

    # ── PAMAP2 (physical activity) ───────────────────────────────────────────
    {
        "RUN_ID_PREFIX": "pamap2",
        "DATASET": "pamap2",
        "EPOCHS": 10,
        "BATCH_SIZE": 128,
        "HIDDEN_SIZES": [128],
        "TRANSFORM": PAMAP2_PIPELINE,
        "DATASET_KW": {
            "equal_per_class": True,
            "time_steps": 128,
        },
    },

    # ── MIT-BIH (ECG) ────────────────────────────────────────────────────────
    {
        "RUN_ID_PREFIX": "mitbih",
        "DATASET": "mitbih",
        "EPOCHS": 10,
        "BATCH_SIZE": 128,
        "HIDDEN_SIZES": [128],
        "TRANSFORM": MITBIH_PIPELINE,
        "MAX_SAMPLES": 2000,
        "DATASET_KW": {
            "two_class": True,
            "equal_per_class": True,
        },
    },

    # ── DVS128 Gesture (neuromorphic) ────────────────────────────────────────
    {
        "RUN_ID_PREFIX": "dvs",
        "DATASET": "dvs_gesture",
        "EPOCHS": 10,
        "BATCH_SIZE": 128,
        "HIDDEN_SIZES": [128],
        "TRANSFORM": DVS_GESTURE_PIPELINE,
        "DATASET_KW": {
            "class_filter": [
                "hand_clap",
                "right_hand_wave",
                "left_hand_wave",
                "right_arm_cw",
            ],
        },
    },

    # ── Large-Scale Audio Dataset ────────────────────────────────────────────
    {
        "RUN_ID_PREFIX": "large-scale-audio",
        "DATASET": "large_scale_audio",
        "EPOCHS": 10,
        "BATCH_SIZE": 128,
        "HIDDEN_SIZES": [128],
        "TRANSFORM": LARGE_SCALE_AUDIO_PIPELINE,
        "DATASET_KW": {
            "duration": 3.0,
            "equal_per_class": True,
        },
    },
]


def build_runs(dataset_configs=DATASET_CONFIGS, methods=METHODS):
    """Build all dataset × method experiment configs from existing defaults."""
    runs = []
    for dataset_cfg in dataset_configs:
        run_id_prefix = dataset_cfg["RUN_ID_PREFIX"]
        for method in methods:
            run = {
                **DEFAULT,
                **dataset_cfg,
                "RUN_ID": f"{run_id_prefix}-{method}",
                "LEARNER": method,
            }
            run.pop("RUN_ID_PREFIX")

            # Keep each run isolated if a runner mutates DATASET_KW internally.
            if "DATASET_KW" in run:
                run["DATASET_KW"] = dict(run["DATASET_KW"])

            runs.append(run)
    return runs


RUNS = build_runs()

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = []
    for run in RUNS:
        result = run_one(run)
        save_results([result])
        results.append(result)
    summarize(results)
