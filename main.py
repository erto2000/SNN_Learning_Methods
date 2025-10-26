# main.py
# Keep this file minimal: only parameters + high-level flow.
from networks.specs import HeadType
from utils.runner import run_all, summarize
from utils.results import save_results

# ──────────────────────────────────────────────────────────────────────────────
# DEFAULTS (you can override per-run below)
DEFAULT = dict(
    # General
    DATA_ROOT        = "./data",
    BATCH_SIZE       = 128,
    EPOCHS           = 10,
    MAX_SAMPLES      = None,
    SAMPLE_LENGTH    = None,
    STRIDE           = None,
    TEST_EVERY_EPOCH = False,
    SEED             = None,

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
    BP_AGG           = "sum",
    BP_LR            = 1e-3,

    # FF
    FF_ALPHA         = 0.6,
    FF_LR            = 1e-3,

    # E-Prop
    EP_LR_IN         = 5e-4,
    EP_LR_REC        = 5e-4,
    EP_LR_OUT        = 1e-3,
    EP_DROP_DIAG     = True,
    EP_WEIGHT_CLIP   = 1.5,

    # PEPITA
    PEP_MODE         = "original",
    PEP_LR           = 1e-2,
    PEP_F_FACTOR     = 0.05,
)

# ──────────────────────────────────────────────────────────────────────────────
# RUNS — configure multiple datasets/learners/variants here (no argparse)
RUNS = [
    {
        **DEFAULT,
        "RUN_ID": "har-bp",
        "DATASET": "har",
        "LEARNER": "bp",
        "EPOCHS": 5,
        "HIDDEN_SIZES": [128],
    },
    # {
    #     **DEFAULT,
    #     "RUN_ID": "har-eprop",
    #     "DATASET": "har",
    #     "LEARNER": "eprop",
    #     "EPOCHS": 5,
    #     "HIDDEN_SIZES": [128],
    # },
    # {
    #     **DEFAULT,
    #     "RUN_ID": "har-ff",
    #     "DATASET": "har",
    #     "LEARNER": "ff",
    #     "EPOCHS": 5,
    #     "HIDDEN_SIZES": [512],
    # },
    # {
    #     **DEFAULT,
    #     "RUN_ID": "har-pepita",
    #     "DATASET": "har",
    #     "LEARNER": "pepita",
    #     "EPOCHS": 5,
    #     "HIDDEN_SIZES": [128],
    # },
]

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = run_all(RUNS)
    summarize(results)
    save_results(results)
