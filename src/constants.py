LOG_DIR = "./logs/"

DEMO_NUM_SAMPLES_DEFAULT = 16
DEMO_SEED_DEFAULT = 0
DEMO_OUT_DEFAULT = "demo.png"
DEMO_LEVELS = ["Excellent", "Good", "Fair", "Poor", "Bad"]
DEMO_PRED_COLOR = "steelblue"
DEMO_GT_COLOR = "coral"

TRAIN_SAMPLE_SEED_DEFAULT = 42

TRAIN_STEPS_DEFAULT = 500
TRAIN_WARMUP_STEPS_DEFAULT = 5
TRAIN_VIT_WARMUP_STEPS_DEFAULT = 100
TRAIN_BATCH_SIZE_DEFAULT = 4
TRAIN_LR_DEFAULT = 1e-3
TRAIN_GRAD_ACCUM_DEFAULT = 32
TRAIN_LOG_EVERY_DEFAULT = 5
TRAIN_OSC_THRESHOLD_DEFAULT = 0.3
TRAIN_OSC_FACTOR_DEFAULT = 0.5
TRAIN_OSC_COOLDOWN_DEFAULT = 50
TRAIN_OSC_MIN_LR_DEFAULT = 1e-6
TRAIN_BACKBONE_LR_SCALE_DEFAULT = 1.0
TRAIN_PRETRAINED_BACKBONE_LR_SCALE_DEFAULT = 0.01
TRAIN_FREEZE_BACKBONE_STEPS_DEFAULT = 150
TRAIN_BACKBONE_WARMUP_STEPS_DEFAULT = 20
TRAIN_WEIGHT_DECAY_DEFAULT = 0.05
TRAIN_HEAD_WEIGHT_DECAY_DEFAULT = 0.0
TRAIN_PRETRAINED_DEFAULT = None
PRETRAINED_TYPES = ("dinov2",)

MODEL_TYPES = ("vit", "cnn")
TRAIN_MODEL_TYPE_DEFAULT = "vit"

# Mirrors the archives referenced by the DeQA-Score meta files (image paths under
# data/Data-DeQA-Score/<dir>/metas/*.json) back to their public source images, since
# zhiyuanyou/Data-DeQA-Score on the HF Hub only ships labels, not pixels.
IQA_DATASETS_HF_REPO_ID = "chaofengc/IQA-PyTorch-Datasets"
IQA_DATASET_ARCHIVES = {
    "koniq": ("koniq10k.tgz", "KONIQ"),
    "spaq": ("spaq.tgz", "SPAQ"),
    "kadid": ("kadid10k.tgz", "KADID10K"),
    "tid2013": ("tid2013.tgz", "TID2013"),
    "csiq": ("csiq.tgz", "CSIQ"),
    "pipal": ("pipal.tar", "PIPAL"),
    "livewild": ("live_challenge.tgz", "LIVE-WILD"),
    "agiqa3k": ("AGIQA-3K.zip", "AGIQA3K"),
}
DATA_DEQA_SCORE_DIR_DEFAULT = "data/Data-DeQA-Score"

# Soft-label generation (see src/datasets/gen_soft_label.py) needs each dataset's raw
# mos.json + split.json, which only the "authentic distortion" MOS datasets ship —
# the others arrive as already-built metas with no raw MOS to regenerate from.
# density_type follows the DeQA-Score appendix: pdf fits larger-std datasets
# (KonIQ, KADID) better, cdf fits smaller-std ones (SPAQ) better.
SOFT_LABEL_DATASET_PARAMS = {
    "koniq": {"density_type": "pdf", "thre_std": 0.2, "thre_diff": 0.1},
    "spaq": {"density_type": "cdf", "thre_std": 0.2, "thre_diff": 0.1},
    "kadid": {"density_type": "pdf", "thre_std": 0.2, "thre_diff": 0.1},
}

DOWNLOAD_DATASETS_ARG_SPECS = [
    {
        "flags": ["--datasets"],
        "kwargs": {
            "nargs": "+",
            "choices": sorted(IQA_DATASET_ARCHIVES.keys()),
            "default": None,
            "help": "Which datasets to download (default: all of them)",
        },
    },
    {
        "flags": ["--data-root"],
        "kwargs": {
            "default": DATA_DEQA_SCORE_DIR_DEFAULT,
            "help": "Path to the Data-DeQA-Score directory containing <DATASET>/metas/*.json",
        },
    },
    {
        "flags": ["--force"],
        "kwargs": {
            "action": "store_true",
            "help": "Re-extract images even if the destination file already exists",
        },
    },
]
