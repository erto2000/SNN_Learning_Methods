# run_training.py
from utils.runner import run_one, summarize
from visualization.training_results import save_results
import timeseries.transforms as transforms

# ──────────────────────────────────────────────────────────────────────────────
# DATASET PIPELINES

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

# DVS128 Gesture (neuromorphic events)
DVS_GESTURE_PIPELINE = transforms.Compose([
    transforms.DownsampleEvents(factor=4),
    transforms.EventToVoxel(H=32, W=32, bins=200, polarity=True),
    transforms.ZScore(),
])

# ──────────────────────────────────────────────────────────────────────────────
# DEFAULTS (tunable per run)
DEFAULT = dict(
    # General
    DATA_ROOT           = "./data",
    BATCH_SIZE          = 128,
    EPOCHS              = 10,      # safe default; each run overrides up to max 10
    MAX_SAMPLES         = None,
    TEST_EVERY_EPOCH    = False,
    EVAL_DTYPE          = "fp32",   # "fp32", "fp16", "bf16" for evaluation,
    EVAL_INT8_WEIGHTS   = False,    # weight-only int8 for evaluation
    SEED                = 123,
    NUM_WORKERS         = 0,
    PIN_MEMORY          = False,

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

    # Time gating
    TIME_GATING_ENABLED = False,
    TIME_GATING_START_U = 0.5,          # normalized 0..1
    TIME_GATING_MODE    = "hard",       # hard|linear|sigmoid|cosine
    TIME_GATING_RAMP_U  = 0.0,
    TIME_GATING_SHARP   = 20.0,

    # Backprop
    BP_AGG           = "mean",
    BP_LR            = 2e-3,

    # FF
    FF_ALPHA         = 0.6,
    FF_LR            = 2e-3,

    # E-Prop
    EP_LR_IN         = 1e-3,
    EP_LR_REC        = 1e-3,
    EP_LR_OUT        = 2e-3,
    EP_DROP_DIAG     = True,
    EP_WEIGHT_CLIP   = 1.5,

    # PEPITA
    PEP_MODE         = "original",
    PEP_LR           = 1e-2,
    PEP_MAX_REL_STEP = 0.05,
    PEP_MOD_RATIO    = 0.1,
)

# ──────────────────────────────────────────────────────────────────────────────
# EXPERIMENTS
RUNS = [
        # # ── HAR (Human Activity Recognition) ───────────────────────────────────────────────────────
        # {
        #     **DEFAULT,
        #     "RUN_ID": "har-bp",
        #     "DATASET": "har",
        #     "LEARNER": "bp",
        #     "EPOCHS": 10,
        #     "BATCH_SIZE": 128,
        #     "HIDDEN_SIZES": [128],
        #     "TRANSFORM": HAR_PIPELINE,
        # },
        #
        # # ── MNIST (static repeated frames) ───────────────────────────────────────
        # {
        #     **DEFAULT,
        #     "RUN_ID": "mnist-static-bp",
        #     "DATASET": "mnist",
        #     "LEARNER": "bp",
        #     "EPOCHS": 10,
        #     "BATCH_SIZE": 128,
        #     "HIDDEN_SIZES": [128],
        #     "TRANSFORM": MNIST_STATIC_PIPELINE,
        # },
        #
        # # ── MNIST (rate-coded spikes)  ────────────────────────────────
        # {
        #     **DEFAULT,
        #     "RUN_ID": "mnist-rate-bp",
        #     "DATASET": "mnist",
        #     "LEARNER": "bp",
        #     "EPOCHS": 10,
        #     "BATCH_SIZE": 128,
        #     "HIDDEN_SIZES": [128],
        #     "TRANSFORM": MNIST_RATE_PIPELINE,
        # },
        #
        # # ── Speech Commands ────────────────────────────────────────
        # {
        #     **DEFAULT,
        #     "RUN_ID": "sc-bp",
        #     "DATASET": "speech_commands",
        #     "LEARNER": "bp",
        #     "EPOCHS": 10,
        #     "BATCH_SIZE": 128,
        #     "HIDDEN_SIZES": [128],
        #     "MAX_SAMPLES": 10000,
        #     "TRANSFORM": SC_PIPELINE,
        #     "DATASET_KW": {
        #         "class_count": 10,
        #         "equal_per_class":True,
        #     },
        # },
        #
        # ── ESC-50 (environmental audio) ─────────────────────────────────────────
        {
            **DEFAULT,
            "RUN_ID": "esc50-bp",
            "DATASET": "esc50",
            "LEARNER": "bp",
            "EPOCHS": 10,
            "BATCH_SIZE": 128,
            "HIDDEN_SIZES": [512],
            "TRANSFORM": ESC50_PIPELINE,
            "DATASET_KW": {
                "class_count": 10,
                "equal_per_class": True,
                "duration": 1
            },
        },
        {
            **DEFAULT,
            "RUN_ID": "esc50-bp-timegate",
            "DATASET": "esc50",
            "LEARNER": "bp",
            "EPOCHS": 10,
            "BATCH_SIZE": 128,
            "HIDDEN_SIZES": [512],
            "TRANSFORM": ESC50_PIPELINE,
            "DATASET_KW": {
                "class_count": 10,
                "equal_per_class": True,
                "duration": 1
            },
            "TIME_GATING_ENABLED": True,
        },
        {
            **DEFAULT,
            "RUN_ID": "esc50-eprop",
            "DATASET": "esc50",
            "LEARNER": "eprop",
            "EPOCHS": 10,
            "BATCH_SIZE": 128,
            "HIDDEN_SIZES": [512],
            "TRANSFORM": ESC50_PIPELINE,
            "DATASET_KW": {
                "class_count": 10,
                "equal_per_class": True,
                "duration": 1
            },
        },
        {
            **DEFAULT,
            "RUN_ID": "esc50-eprop-timegate",
            "DATASET": "esc50",
            "LEARNER": "eprop",
            "EPOCHS": 10,
            "BATCH_SIZE": 128,
            "HIDDEN_SIZES": [512],
            "TRANSFORM": ESC50_PIPELINE,
            "DATASET_KW": {
                "class_count": 10,
                "equal_per_class": True,
                "duration": 1
            },
            "TIME_GATING_ENABLED": True,
        },
        {
            **DEFAULT,
            "RUN_ID": "esc50-ff",
            "DATASET": "esc50",
            "LEARNER": "ff",
            "EPOCHS": 10,
            "BATCH_SIZE": 128,
            "HIDDEN_SIZES": [512],
            "TRANSFORM": ESC50_PIPELINE,
            "DATASET_KW": {
                "class_count": 10,
                "equal_per_class": True,
                "duration": 1
            },
        },
        {
            **DEFAULT,
            "RUN_ID": "esc50-ff-timegate",
            "DATASET": "esc50",
            "LEARNER": "ff",
            "EPOCHS": 10,
            "BATCH_SIZE": 128,
            "HIDDEN_SIZES": [512],
            "TRANSFORM": ESC50_PIPELINE,
            "DATASET_KW": {
                "class_count": 10,
                "equal_per_class": True,
                "duration": 1
            },
            "TIME_GATING_ENABLED": True,
        },
        {
            **DEFAULT,
            "RUN_ID": "esc50-pepita",
            "DATASET": "esc50",
            "LEARNER": "pepita",
            "EPOCHS": 10,
            "BATCH_SIZE": 128,
            "HIDDEN_SIZES": [512],
            "TRANSFORM": ESC50_PIPELINE,
            "DATASET_KW": {
                "class_count": 10,
                "equal_per_class": True,
                "duration": 1
            },
        },
        {
            **DEFAULT,
            "RUN_ID": "esc50-pepita-timegate",
            "DATASET": "esc50",
            "LEARNER": "pepita",
            "EPOCHS": 10,
            "BATCH_SIZE": 128,
            "HIDDEN_SIZES": [512],
            "TRANSFORM": ESC50_PIPELINE,
            "DATASET_KW": {
                "class_count": 10,
                "equal_per_class": True,
                "duration": 1
            },
            "TIME_GATING_ENABLED": True,
        },
        #
        # # ── UrbanSound8K (urban audio) ───────────────────────────────────────────
        # {
        #     **DEFAULT,
        #     "RUN_ID": "urban8k-bp-timegate",
        #     "DATASET": "urban8k",
        #     "LEARNER": "bp",
        #     "EPOCHS": 10,
        #     "BATCH_SIZE": 128,
        #     "HIDDEN_SIZES": [512],
        #     "TRANSFORM": URBAN8K_PIPELINE,
        #     "DATASET_KW": {
        #         "equal_per_class": True,
        #     },
        #     "TIME_GATING_ENABLED": True,
        #     "TIME_GATING_START_U": 0.2,
        #     "TIME_GATING_MODE": "linear",
        # },
        #
        # # ── PAMAP2 (physical activity) ────────────────────────────────────
        # {
        #     **DEFAULT,
        #     "RUN_ID": "pamap2-bp",
        #     "DATASET": "pamap2",
        #     "LEARNER": "bp",
        #     "EPOCHS": 10,
        #     "BATCH_SIZE": 128,
        #     "HIDDEN_SIZES": [128],
        #     "TRANSFORM": PAMAP2_PIPELINE,
        #     "DATASET_KW": {
        #         "equal_per_class": True,
        #         "time_steps": 128,
        #     },
        # },
        #
        # # ── MIT-BIH (ECG) ─────────────────────────────────────────────
        # {
        #     **DEFAULT,
        #     "RUN_ID": "mitbih-bp",
        #     "DATASET": "mitbih",
        #     "LEARNER": "bp",
        #     "EPOCHS": 10,
        #     "BATCH_SIZE": 128,
        #     "HIDDEN_SIZES": [128],
        #     "TRANSFORM": MITBIH_PIPELINE,
        #     "MAX_SAMPLES": 2000,
        #     "DATASET_KW": {
        #         "two_class": True,
        #         "equal_per_class": True
        #     },
        # },
        #
        # # ── DVS128 Gesture (neuromorphic) ────────────────────────────────────────
        # {
        #     **DEFAULT,
        #     "RUN_ID": "dvs-bp",
        #     "DATASET": "dvs_gesture",
        #     "LEARNER": "bp",
        #     "EPOCHS": 10,
        #     "BATCH_SIZE": 128,
        #     "HIDDEN_SIZES": [512],
        #     "TRANSFORM": DVS_GESTURE_PIPELINE,
        # },
    ]

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = []
    for run in RUNS:
        result = run_one(run)
        save_results([result])
        results.append(result)
    summarize(results)
