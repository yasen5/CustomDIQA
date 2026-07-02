import os

LOG_DIR = "./logs/"

DEMO_NUM_SAMPLES_DEFAULT = 16
DEMO_SEED_DEFAULT = 0
DEMO_OUT_DEFAULT = "demo.png"
DEMO_LEVELS = ["Excellent", "Good", "Fair", "Poor", "Bad"]
DEMO_PRED_COLOR = "steelblue"
DEMO_GT_COLOR = "coral"

EVAL_BATCH_SIZE_DEFAULT = 16
EVAL_SPLIT_DEFAULT = "test"
EVAL_OUT_DEFAULT = "eval_cross_dataset.png"
EVAL_SRCC_COLOR = "steelblue"
EVAL_PLCC_COLOR = "coral"

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
DATASET_KEYS_DEFAULT = sorted(IQA_DATASET_ARCHIVES.keys())

# metas/*.json filenames actually shipped per dataset — not uniform. koniq/spaq/kadid use
# the train.json/test.json this repo generates (see generate_soft_labels below); pipal
# ships its own pre-built train/test metas; tid2013/csiq/livewild/agiqa3k ship test-only
# benchmark metas with no train split and no level_probs (eval-only, see resolve_dataset_paths).
DATASET_META_FILENAMES = {
    "koniq": {"train": "train.json", "test": "test.json"},
    "spaq": {"train": "train.json", "test": "test.json"},
    "kadid": {"train": "train.json", "test": "test.json"},
    "pipal": {"train": "train_pipal_19k.json", "test": "test_pipal_5k.json"},
    "tid2013": {"train": None, "test": "test_tid2013_3k.json"},
    "csiq": {"train": None, "test": "test_csiq_866.json"},
    "livewild": {"train": None, "test": "test_livew_1k.json"},
    "agiqa3k": {"train": None, "test": "test_agiqa_3k.json"},
}

# Shared by run_train.py and run_demo.py: dataset selection defaults to every dataset
# with downloadable images (IQA_DATASET_ARCHIVES), excluded from rather than opted into.
DATASET_SELECT_ARG_SPECS = [
    {
        "flags": ["--datasets"],
        "kwargs": {
            "nargs": "+",
            "choices": DATASET_KEYS_DEFAULT,
            "default": DATASET_KEYS_DEFAULT,
            "help": "Datasets to use (default: all datasets with downloadable images)",
        },
    },
    {
        "flags": ["--exclude-datasets"],
        "kwargs": {
            "nargs": "+",
            "choices": DATASET_KEYS_DEFAULT,
            "default": [],
            "help": "Datasets to exclude from --datasets",
        },
    },
    {
        "flags": ["--data-root"],
        "kwargs": {
            "default": DATA_DEQA_SCORE_DIR_DEFAULT,
            "help": "Path to the Data-DeQA-Score directory containing <DATASET>/metas/*.json "
                    "(doubles as the shared image root)",
        },
    },
]


def _usability_warning(key, split, reason):
    banner = "!" * 100
    print(banner)
    print(f"!!! WARNING: dropping dataset {key!r} ({split} split) — {reason}")
    print(f"!!! It will NOT be used for this run.")
    print(banner)


def resolve_dataset_paths(datasets, exclude_datasets, data_root, split):
    """Turn --datasets/--exclude-datasets/--data-root into (keys, meta json paths) for
    `split` ("train" or "test"), per the real per-dataset filenames in DATASET_META_FILENAMES.
    Datasets with no meta file for `split` (e.g. the test-only benchmark sets have no train
    split) are skipped rather than erroring, since --datasets defaults to every dataset.

    Before accepting a dataset, checks that every image its metadata references actually
    exists under `data_root` — an incomplete download/extract otherwise surfaces only
    gradually, as a trickle of "image not found" warnings during training/eval. Any
    mismatch drops the whole dataset (with a loud warning) rather than silently evaluating
    on a partial, likely-biased subset of it."""
    from src.datasets.gen_soft_label import load_soft_label_samples

    keys, paths = [], []
    for k in datasets:
        if k in exclude_datasets:
            continue
        filename = DATASET_META_FILENAMES[k][split]
        if filename is None:
            print(f"NOTE: {k} has no {split!r} split, skipping.")
            continue
        path = os.path.join(data_root, IQA_DATASET_ARCHIVES[k][1], "metas", filename)

        try:
            samples = load_soft_label_samples(path)
        except (OSError, ValueError) as ex:
            _usability_warning(k, split, f"could not read metadata at {path}: {ex}")
            continue

        n_meta = len(samples)
        n_found = sum(1 for s in samples if os.path.isfile(os.path.join(data_root, s.image)))
        if n_found != n_meta:
            _usability_warning(
                k, split,
                f"only {n_found}/{n_meta} images referenced in {path} were found on disk "
                f"under {data_root!r}. This usually means the dataset was only partially "
                f"downloaded/extracted."
            )
            continue

        keys.append(k)
        paths.append(path)
    if not keys:
        raise ValueError(f"No selected dataset has a {split!r} split (after --exclude-datasets)")
    return keys, paths


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
            "choices": DATASET_KEYS_DEFAULT,
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
