import os

LOG_DIR = "./logs/"
ANALYSIS_RESULTS_DIR = "analysis-results"


def analysis_result_path(filename):
    return os.path.join(ANALYSIS_RESULTS_DIR, filename)

DEMO_NUM_SAMPLES_DEFAULT = 16
DEMO_SEED_DEFAULT = 0
DEMO_OUT_DEFAULT = analysis_result_path("demo.png")
DEMO_LEVELS = ["Excellent", "Good", "Fair", "Poor", "Bad"]
DEMO_PRED_COLOR = "steelblue"
DEMO_GT_COLOR = "coral"

# Canonical KADID-10k 25 synthetic distortion types, in original-paper order.
# Index with KADID_DISTORTION_NAMES[distortion_type - 1] (distortion_type is 1-25, from filenames).
KADID_DISTORTION_NAMES = [
    "Gaussian blur", "Lens blur", "Motion blur", "Color diffusion", "Color shift",
    "Color quantization", "Color saturation 1", "Color saturation 2", "JPEG2000 compression",
    "JPEG compression", "White noise", "White noise in color component", "Impulse noise",
    "Multiplicative noise", "Denoise", "Brighten", "Darken", "Mean shift", "Jitter",
    "Non-eccentricity patch", "Pixelate", "Quantization", "Color block", "High sharpen",
    "Contrast change",
]

EVAL_BATCH_SIZE_DEFAULT = 16
EVAL_SPLIT_DEFAULT = "test"
EVAL_OUT_DEFAULT = analysis_result_path("eval_cross_dataset.png")
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
PRETRAINED_TYPES = ("dinov2", "imagenet", "topiq_nr", "musiq_koniq")

MODEL_TYPES = ("vit", "cnn", "hybrid", "musiq")
TRAIN_MODEL_TYPE_DEFAULT = "vit"

# Mirrors the archives referenced by the DeQA-Score meta files (image paths under
# data/Data-DeQA-Score/<dir>/metas/*.json) back to their public source images, since
# zhiyuanyou/Data-DeQA-Score on the HF Hub only ships labels, not pixels.
IQA_DATASETS_HF_REPO_ID = "chaofengc/IQA-PyTorch-Datasets"
IQA_DATASET_ARCHIVES = {
    # The upstream KonIQ archive contains both 512x384 and 1024x768 copies with the same
    # basenames. Keep them as separate dataset keys/paths so a basename-only extract cannot
    # silently overwrite or skip the intended resolution.
    "koniq512": ("koniq10k.tgz", "KONIQ512"),
    "koniq1024": ("koniq10k.tgz", "KONIQ1024"),
    # Legacy path used by older local runs. It is intentionally excluded from defaults below;
    # prefer koniq1024 for official-resolution evaluation, or koniq512 for the small images.
    "koniq": ("koniq10k.tgz", "KONIQ"),
    "spaq": ("spaq.tgz", "SPAQ"),
    "kadid": ("kadid10k.tgz", "KADID10K"),
    "tid2013": ("tid2013.tgz", "TID2013"),
    "csiq": ("csiq.tgz", "CSIQ"),
    "pipal": ("pipal.tar", "PIPAL"),
    "livewild": ("live_challenge.tgz", "LIVE-WILD"),
    "agiqa3k": ("AGIQA-3K.zip", "AGIQA3K"),
    "flive": ("flive.tgz", "FLIVE"),
}
KONIQ_ARCHIVE_IMAGE_DIRS = {
    "koniq512": "koniq10k/512x384",
    "koniq1024": "koniq10k/1024x768",
}
# koniq512/koniq1024 reuse the same raw MOS/split files as the legacy KONIQ metadata, but
# write their generated train/test metas under separate output dirs with separate image paths.
DATASET_META_SOURCE_DIRS = {
    "koniq512": "KONIQ",
    "koniq1024": "KONIQ",
}
DATA_DEQA_SCORE_DIR_DEFAULT = "data/Data-DeQA-Score"
DATASET_KEYS = sorted(IQA_DATASET_ARCHIVES.keys())
# flive is excluded like kadid: a ~5GB archive that needs its own meta generation
# (see generate_pyiqa_mos_labels in gen_soft_label.py) rather than being ready for every default run.
# koniq1024 is the default KonIQ variant because it matches the official-resolution benchmark.
DATASET_KEYS_DEFAULT = [key for key in DATASET_KEYS if key not in ("kadid", "flive", "koniq", "koniq512")]
DOWNLOAD_DATASET_KEYS_DEFAULT = [key for key in DATASET_KEYS if key != "koniq"]

# metas/*.json filenames actually shipped per dataset — not uniform. koniq/spaq/kadid use
# the train.json/test.json this repo generates (see generate_soft_labels below); pipal
# ships its own pre-built train/test metas; tid2013/csiq/livewild/agiqa3k ship a single
# test-only benchmark meta (see ORIGINAL_TEST_ONLY_META_FILENAMES) that scripts/split_train_test.py
# partitions into metas/train.json + metas/test.json (run it once after downloading); flive has
# no train split at all — see generate_pyiqa_mos_labels in gen_soft_label.py (it's eval-only, no
# per-image std upstream, and has no entry in has_zhiyuanyou_metas below).
DATASET_META_FILENAMES = {
    "koniq512": {"train": "train.json", "test": "test.json"},
    "koniq1024": {"train": "train.json", "test": "test.json"},
    "koniq": {"train": "train.json", "test": "test.json"},
    "spaq": {"train": "train.json", "test": "test.json"},
    "kadid": {"train": "train.json", "test": "test.json"},
    "pipal": {"train": "train_pipal_19k.json", "test": "test_pipal_5k.json"},
    "tid2013": {"train": "train.json", "test": "test.json"},
    "csiq": {"train": "train.json", "test": "test.json"},
    "livewild": {"train": "train.json", "test": "test.json"},
    "agiqa3k": {"train": "train.json", "test": "test.json"},
    "flive": {"train": None, "test": "test.json"},
}

# The single pre-built benchmark meta each test-only dataset ships with (under its own
# metas/ dir), before scripts/split_train_test.py partitions it into train.json/test.json.
ORIGINAL_TEST_ONLY_META_FILENAMES = {
    "tid2013": "test_tid2013_3k.json",
    "csiq": "test_csiq_866.json",
    "livewild": "test_livew_1k.json",
    "agiqa3k": "test_agiqa_3k.json",
}

# Shared by run_train.py and run_demo.py: dataset selection defaults to every dataset
# with downloadable images except KADID, excluded from rather than opted into.
DATASET_SELECT_ARG_SPECS = [
    {
        "flags": ["--datasets"],
        "kwargs": {
            "nargs": "+",
            "choices": DATASET_KEYS,
            "default": DATASET_KEYS_DEFAULT,
            "help": "Datasets to use (default: standard eval set with koniq1024; excludes kadid, flive, "
                    "legacy koniq, and koniq512)",
        },
    },
    {
        "flags": ["--exclude-datasets"],
        "kwargs": {
            "nargs": "+",
            "choices": DATASET_KEYS,
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
    "koniq512": {"density_type": "pdf", "thre_std": 0.2, "thre_diff": 0.1},
    "koniq1024": {"density_type": "pdf", "thre_std": 0.2, "thre_diff": 0.1},
    "koniq": {"density_type": "pdf", "thre_std": 0.2, "thre_diff": 0.1},
    "spaq": {"density_type": "cdf", "thre_std": 0.2, "thre_diff": 0.1},
    "kadid": {"density_type": "pdf", "thre_std": 0.2, "thre_diff": 0.1},
}


def has_zhiyuanyou_metas(key):
    """Whether zhiyuanyou/Data-DeQA-Score ships *any* metadata for this dataset key: raw
    mos.json+split.json (SOFT_LABEL_DATASET_PARAMS), pipal's own pre-built train/test metas, or
    a single test-only benchmark meta (ORIGINAL_TEST_ONLY_META_FILENAMES). False only for flive
    today — any other dataset key added to IQA_DATASET_ARCHIVES without a matching entry in one
    of those three places falls through the same way, so scripts/download_datasets.py's call to
    gen_soft_label.generate_pyiqa_mos_labels picks it up automatically instead of needing a
    bespoke per-dataset special case."""
    return key in SOFT_LABEL_DATASET_PARAMS or key == "pipal" or key in ORIGINAL_TEST_ONLY_META_FILENAMES

DOWNLOAD_DATASETS_ARG_SPECS = [
    {
        "flags": ["--datasets"],
        "kwargs": {
            "nargs": "+",
            "choices": DATASET_KEYS,
            "default": None,
            "help": "Which datasets to download (default: all non-legacy dataset keys)",
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
