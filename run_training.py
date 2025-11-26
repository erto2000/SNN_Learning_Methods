# run_training.py
from utils.runner import run_one, summarize
from visualization.training_results import save_results
import timeseries.transforms as transforms

# ──────────────────────────────────────────────────────────────────────────────
# DATASET PIPELINES  (balanced for a 4 GB GPU and quick iterations)

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
    transforms.ToLogMel(sample_rate=16000, n_mels=64, win_len_ms=25, hop_ms=10),         # -> [F,M]
    transforms.ZScore(),
])

# Audio (ESC-50 / UrbanSound8K): 4s @16k → log-mel(64)
AUDIO_SR = 16000
ESC50_PIPELINE = transforms.Compose([
    transforms.ToFloat32(),
    transforms.Ensure2D(),
    transforms.ToLogMel(sample_rate=AUDIO_SR, n_mels=64, win_len_ms=25, hop_ms=10),
    transforms.ZScore(),   # auto-fit on train
    transforms.SlidingWindow(length=128, hop=128),
])
URBAN8K_PIPELINE = ESC50_PIPELINE  # identical defaults

# PAMAP2 (IMU): shorter windows to keep VRAM low
PAMAP2_PIPELINE = transforms.Compose([
    transforms.ToFloat32(),
    transforms.ZScore(),
    transforms.SlidingWindow(length=128, hop=128),
])

# MIT-BIH (ECG): 1s@360Hz windows with overlap
MITBIH_PIPELINE = transforms.Compose([
    transforms.ToFloat32(),
    transforms.ZScore(),
    transforms.SlidingWindow(length=360, hop=180),
])

# DVS128 Gesture (neuromorphic events): events → voxel → short windows
DVS_GESTURE_PIPELINE = transforms.Compose([
    transforms.DownsampleEvents(factor=4),
    transforms.EventToVoxel(H=32, W=32, bins=200, polarity=True),
    transforms.ZScore(),
    # transforms.SlidingWindow(length=50, hop=25),
])

# ──────────────────────────────────────────────────────────────────────────────
# DEFAULTS (tunable per run)
DEFAULT = dict(
    # General
    DATA_ROOT           = "./data",
    BATCH_SIZE          = 128,
    EPOCHS              = 8,      # safe default; each run overrides up to max 10
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

    # Backprop
    BP_AGG           = "mean",
    BP_LR            = 2e-3,

    # FF
    FF_ALPHA         = 0.6,
    FF_LR            = 2e-3,

    # E-Prop
    EP_LR_IN         = 6e-4,
    EP_LR_REC        = 6e-4,
    EP_LR_OUT        = 1e-3,
    EP_DROP_DIAG     = True,
    EP_WEIGHT_CLIP   = 1.5,

    # PEPITA (unused here but kept for completeness)
    PEP_MODE         = "original",
    PEP_LR           = 1e-2,
    PEP_MAX_REL_STEP = 0.05,
    PEP_MOD_RATIO    = 0.1,
)

# ──────────────────────────────────────────────────────────────────────────────
# EXPERIMENTS: ≤ 10 epochs each, VRAM-friendly batches/hidden sizes
RUNS = [
    # # ── HAR (Inertial) ───────────────────────────────────────────────────────
    # {
    #     **DEFAULT,
    #     "RUN_ID": "har-bp",
    #     "DATASET": "har",
    #     "LEARNER": "bp",
    #     "EPOCHS": 8,
    #     "BATCH_SIZE": 128,
    #     "HIDDEN_SIZES": [128],
    #     "TRANSFORM": HAR_PIPELINE,
    # },

    # # ── MNIST (static repeated frames) ───────────────────────────────────────
    # {
    #     **DEFAULT,
    #     "RUN_ID": "mnist-static-bp",
    #     "DATASET": "mnist",
    #     "LEARNER": "bp",
    #     "EPOCHS": 8,
    #     "BATCH_SIZE": 256,
    #     "HIDDEN_SIZES": [256, 256],
    #     "BP_LR": 3e-3,
    #     "TRANSFORM": MNIST_STATIC_PIPELINE,
    # },

    # # ── MNIST (rate-coded spikes) with E-Prop ────────────────────────────────
    # {
    #     **DEFAULT,
    #     "RUN_ID": "mnist-rate-eprop",
    #     "DATASET": "mnist",
    #     "LEARNER": "eprop",
    #     "EPOCHS": 8,
    #     "BATCH_SIZE": 256,
    #     "HIDDEN_SIZES": [512],
    #     "EP_LR_IN": 1e-3,
    #     "EP_LR_REC": 1e-3,
    #     "EP_LR_OUT": 2e-3,
    #     "TRANSFORM": MNIST_RATE_PIPELINE,
    # },

    # # ── Speech Commands (small, fast) ────────────────────────────────────────
    # {
    #     **DEFAULT,
    #     "RUN_ID": "sc-bp",
    #     "DATASET": "speech_commands",
    #     "LEARNER": "bp",
    #     "EPOCHS": 10,
    #     "BATCH_SIZE": 128,
    #     "HIDDEN_SIZES": [128],
    #     "BP_LR": 3e-3,
    #     "MAX_SAMPLES": 10000,
    #     "TRANSFORM": SC_PIPELINE,
    #     "DATASET_KW": {
    #         "class_count": 10,
    #     },
    # },

    # # ── ESC-50 (environmental audio) ─────────────────────────────────────────
    # {
    #     **DEFAULT,
    #     "RUN_ID": "esc50-bp",
    #     "DATASET": "esc50",
    #     "LEARNER": "bp",
    #     "EPOCHS": 10,
    #     "BATCH_SIZE": 128,
    #     "HIDDEN_SIZES": [128],
    #     "TRANSFORM": ESC50_PIPELINE,
    #     "DATASET_KW": {
    #         "class_count": 10,
    #     },
    # },

    # # ── UrbanSound8K (urban audio) ───────────────────────────────────────────
    # {
    #     **DEFAULT,
    #     "RUN_ID": "urban8k-bp",
    #     "DATASET": "urban8k",
    #     "LEARNER": "bp",
    #     "EPOCHS": 10,
    #     "BATCH_SIZE": 128,
    #     "HIDDEN_SIZES": [128],
    #     "TRANSFORM": URBAN8K_PIPELINE,
    #     # "MAX_SAMPLES": 2000,
    # },

    # # ── PAMAP2 (IMU; tuned for 4 GB VRAM) ────────────────────────────────────
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
    #         "train_subjects": [101, 102, 103, 104, 105, 106, 107, 108],
    #         "test_subjects": [109, 105, 106],
    #         "min_len": 100,
    #     },
    # },

    # # ── MIT-BIH (ECG; small cap) ─────────────────────────────────────────────
    # {
    #     **DEFAULT,
    #     "RUN_ID": "mitbih-bp",
    #     "DATASET": "mitbih",
    #     "LEARNER": "bp",
    #     "EPOCHS": 8,
    #     "BATCH_SIZE": 32,
    #     "HIDDEN_SIZES": [256],
    #     "BP_LR": 1.5e-3,
    #     "TRANSFORM": MITBIH_PIPELINE,
    #     "MAX_SAMPLES": 2000,  # keeps it light
    #     "DATASET_KW": {
    #         "train_records": [101, 106, 108],  # small subset
    #         "test_records": [100],
    #         "win_samples": 360,
    #         "local_only": True,
    #     },
    # },

    # # ── DVS128 Gesture (neuromorphic) ────────────────────────────────────────
    # {
    #     **DEFAULT,
    #     "RUN_ID": "dvs-bp",
    #     "DATASET": "dvs_gesture",
    #     "LEARNER": "bp",
    #     "EPOCHS": 10,
    #     "BATCH_SIZE": 128,
    #     "HIDDEN_SIZES": [128],
    #     "BP_LR": 1e-3,
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
