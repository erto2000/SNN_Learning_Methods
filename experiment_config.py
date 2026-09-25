"""Shared dataset pipelines and configurations for training and window searches."""
import timeseries.transforms as transforms


# ──────────────────────────────────────────────────────────────────────────────
# DATASET PIPELINES

# HAR: 128 timesteps, 9 channels
HAR_PIPELINE = transforms.Compose([
    transforms.ToFloat32(),
    transforms.Ensure2D(),
    transforms.ZScore(),
])

# MNIST static: 50 timesteps, 784 pixels
MNIST_STATIC_PIPELINE = transforms.Compose([
    transforms.ToFloat32(),
    transforms.RepeatStatic(T=50),
])

# MNIST rate: 20 timesteps, 784 pixels
MNIST_RATE_PIPELINE = transforms.Compose([
    transforms.ToFloat32(),
    transforms.DeterministicSpikes(gain=0.7, T=20),
])

# Speech Commands: 1 s audio -> 101 log-mel frames, 64 bands
SC_PIPELINE = transforms.Compose([
    transforms.ToFloat32(),
    transforms.Ensure2D(),
    transforms.Resample(new_sr=16000),
    transforms.ToLogMel(sample_rate=16000, n_mels=64, win_len_ms=25, hop_ms=10),
    transforms.ZScore(),
])

# ESC-50: 1 s segments -> 101 log-mel frames, 64 bands
ESC50_PIPELINE = transforms.Compose([
    transforms.ToFloat32(),
    transforms.Ensure2D(),
    transforms.Resample(new_sr=44100),
    transforms.ToLogMel(sample_rate=44100, n_mels=64, win_len_ms=25, hop_ms=10),
    transforms.ZScore(),
])

# UrbanSound8K: variable-length clips -> variable log-mel frames, 64 bands
URBAN8K_PIPELINE = transforms.Compose([
    transforms.ToFloat32(),
    transforms.Ensure2D(),
    transforms.Resample(new_sr=44100),
    transforms.ToLogMel(sample_rate=44100, n_mels=64, win_len_ms=25, hop_ms=10),
    transforms.ZScore(),
])

# PAMAP2 (IMU): 128 timesteps, 27 channels
PAMAP2_PIPELINE = transforms.Compose([
    transforms.ToFloat32(),
    transforms.ZScore(),
])

# MIT-BIH (ECG): 360 timesteps, 1 channel
MITBIH_PIPELINE = transforms.Compose([
    transforms.ToFloat32(),
    transforms.ZScore(),
])

# DVS128 Gesture: 200 bins, 32 x 32 pixels with two polarities (2048 features)
DVS_GESTURE_PIPELINE = transforms.Compose([
    transforms.DownsampleEvents(factor=4),
    transforms.EventToVoxel(H=32, W=32, bins=200, polarity=True),
    transforms.ZScore(),
])

# Large-scale audio: 3 s segments -> 301 log-mel frames, 64 bands
LARGE_SCALE_AUDIO_PIPELINE = transforms.Compose([
    transforms.ToFloat32(),
    transforms.Resample(new_sr=16000),
    transforms.ToLogMel(sample_rate=16000, n_mels=64, win_len_ms=25, hop_ms=10),
    transforms.ZScore(),
])

# ──────────────────────────────────────────────────────────────────────────────
# DEFAULTS (shared across runs)
DEFAULT = dict(
    # Setup
    DATA_ROOT            = "./data",
    NUM_WORKERS          = 0,        # 0 loads data in the main process
    PIN_MEMORY           = False,    # speeds host-to-GPU transfer when enabled
    SEED                 = 123,
    BATCH_SIZE           = 128,
    EPOCHS               = 10,
    MAX_SAMPLES          = None,
    DTYPE                = "fp32",   # training and evaluation precision
    COST_COMPUTE_WEIGHT  = 1.0,      # weight of arithmetic work in the time estimate
    COST_ACCESS_WEIGHT   = 1.0,      # weight of memory traffic in the time estimate

    # Evaluation
    DATA_SPLIT           = {"train": 80, "validation": 10, "test": 10},
    TEST_EVERY_EPOCH     = False,
    EVAL_INT8_WEIGHTS    = False,               # quantize weights only during evaluation
    TIME_EVAL            = True,
    TIME_EVAL_FRACS      = (0.25, 0.50, 1.0),   # fractions of a sequence to evaluate
    TIME_EVAL_INCLUDE_T1 = True,                # also evaluate after the first timestep
    WINDOW_BOUND_SAMPLES = 256,                 # samples inspected to bound window search

    # Network
    HIDDEN_SIZES         = [128],
    BETA                 = 0.9,             # membrane-state decay
    SPIKE_GRAD           = "fast_sigmoid",  # surrogate gradient for spikes
    SLOPE                = 25.0,            # surrogate-gradient sharpness
    THRESHOLD            = 1.0,             # membrane potential needed to spike
    HEAD                 = "logits",
    RECURRENT            = False,
    INIT_TYPE            = "default",
    NORM                 = None,
    HIDDEN_BIAS          = False,
    HEAD_BIAS            = False,

    # Backprop
    BP_AGG               = "sum",   # accumulate output evidence across timesteps
    BP_LR                = 2e-3,
    BP_OPTIMIZER         = "adam",

    # FF
    FF_ALPHA             = 0.6,      # scales the positive/negative goodness gap in the loss
    FF_LR                = 2e-3,
    FF_OPTIMIZER         = "adam",

    # E-Prop
    EP_LR_IN             = 1e-3,
    EP_LR_REC            = 1e-3,
    EP_LR_OUT            = 2e-3,
    EP_OPTIMIZER         = "adam",
    EP_DROP_DIAG         = True,     # omit self-connections from recurrent updates
    EP_WEIGHT_CLIP       = 1.5,      # maximum magnitude of e-prop weights

    # PEPITA
    PEP_MODE             = "accum",  # use spike-rate differences for one update per sequence
    PEP_LR               = 1e-3,
    PEP_OPTIMIZER        = "adam",
    PEP_MAX_REL_STEP     = None,     # optional relative update cap; None disables it
    PEP_MOD_RATIO        = 0.1,      # target std of perturbation / std of input
)


METHODS = (
    "bp",
    "ff",
    "eprop",
    "pepita",
)

# ──────────────────────────────────────────────────────────────────────────────
# EXPERIMENTS
# Dataset entries override the shared defaults and expand over METHODS.
DATASET_CONFIGS = [
    # HAR (Human Activity Recognition)
    {
        "RUN_ID_PREFIX": "har",
        "DATASET": "har",
        "TRANSFORM": HAR_PIPELINE,
        "PEP_LR": 1e-4,  # Full-length HAR sequences need a smaller step.
    },

    # MNIST (static repeated frames)
    {
        "RUN_ID_PREFIX": "mnist-static",
        "DATASET": "mnist",
        "MAX_SAMPLES": 10000,
        "PEP_LR": 1e-4,
        "TRANSFORM": MNIST_STATIC_PIPELINE,
    },

    # MNIST (rate-coded spikes)
    {
        "RUN_ID_PREFIX": "mnist-rate",
        "DATASET": "mnist",
        "MAX_SAMPLES": 10000,
        "PEP_LR": 1e-4,
        "TRANSFORM": MNIST_RATE_PIPELINE,
    },

    # Speech Commands
    {
        "RUN_ID_PREFIX": "sc",
        "DATASET": "speech_commands",
        "MAX_SAMPLES": 10000,
        "TRANSFORM": SC_PIPELINE,
        "DATASET_KW": {
            "class_filter": ["yes", "no", "stop"],
            "equal_per_class": True,
        },
    },

    # ESC-50 (environmental audio)
    {
        "RUN_ID_PREFIX": "esc50",
        "DATASET": "esc50",
        "TRANSFORM": ESC50_PIPELINE,
        "DATASET_KW": {
            "class_filter": ["dog", "rain", "siren", "helicopter"],
            "equal_per_class": True,
            "duration": 1,
        },
    },

    # UrbanSound8K (urban audio)
    {
        "RUN_ID_PREFIX": "urban8k",
        "DATASET": "urban8k",
        "TRANSFORM": URBAN8K_PIPELINE,
        "DATASET_KW": {
            "class_filter": ["dog_bark", "siren", "gun_shot"],
            "equal_per_class": True,
        },
    },

    # PAMAP2 (physical activity)
    {
        "RUN_ID_PREFIX": "pamap2",
        "DATASET": "pamap2",
        "TRANSFORM": PAMAP2_PIPELINE,
        "DATASET_KW": {
            "equal_per_class": True,
            "time_steps": 128,
        },
    },

    # MIT-BIH (ECG)
    {
        "RUN_ID_PREFIX": "mitbih",
        "DATASET": "mitbih",
        "TRANSFORM": MITBIH_PIPELINE,
        "MAX_SAMPLES": 2000,
        "DATASET_KW": {
            "two_class": True,
            "equal_per_class": True,
        },
    },

    # DVS128 Gesture (neuromorphic events)
    {
        "RUN_ID_PREFIX": "dvs",
        "DATASET": "dvs_gesture",
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

    # Large-scale audio
    {
        "RUN_ID_PREFIX": "large-scale-audio",
        "DATASET": "large_scale_audio",
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
            # Give each run its own mutable settings.
            run["DATA_SPLIT"] = dict(run["DATA_SPLIT"])

            if "DATASET_KW" in run:
                run["DATASET_KW"] = dict(run["DATASET_KW"])

            runs.append(run)
    return runs


RUNS = build_runs()
