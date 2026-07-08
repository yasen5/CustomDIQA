"""Evaluate DeQA-Score-Mix3 against the paper's reported SRCC/PLCC. Must run under deqa_venv, not
the main custom_venv:

    deqa_venv/bin/python3 scripts/run_eval_deqa.py --sample-fraction 0.25

Everything (dataset loading, model, scoring, metrics) runs in this one process -- no subprocess,
no cross-process image serialization. This deliberately avoids `import src...`: src/__init__.py
eagerly imports src.model (which imports pyiqa- and torchvision-dependent code not installed in
deqa_venv -- see src/model/deqa/constants.py for why deqa_venv is a separate, minimal venv in the
first place). Instead: src/model/deqa is imported as a bare top-level package (its internal
relative imports don't care what its parent package is called), and the small dataset-loading /
metric helpers are pulled directly from src/constants.py and src/datasets/gen_soft_label.py via
importlib, bypassing src/__init__.py entirely.
"""
import argparse
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "model"))
from deqa import DeQAScoreIQA  # noqa: E402  (src/model/deqa, imported as a bare top-level package)


def _load_standalone_module(name, path):
    """Load a single .py file as its own module, without importing its parent package(s) --
    i.e. without triggering src/__init__.py's pyiqa/torchvision-dependent imports. Safe here only
    because the specific names we pull from these two files (below) have no such imports at
    module level -- any heavier imports in these files (e.g. scipy in calculate_srcc_plcc) are
    deferred inside function bodies, so they only matter if/when that function is actually called."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_constants = _load_standalone_module("_deqa_eval_src_constants", str(REPO_ROOT / "src" / "constants.py"))
_gen_soft_label = _load_standalone_module(
    "_deqa_eval_gen_soft_label", str(REPO_ROOT / "src" / "datasets" / "gen_soft_label.py")
)

OUT_JSON_DEFAULT = "analysis-results/eval_deqa.json"


def resolve_dataset_paths(datasets, exclude_datasets, data_root, split):
    """Same contract as src.constants.resolve_dataset_paths (see there for the reasoning) --
    reimplemented here rather than imported, since the real one's body does
    `from src.datasets.gen_soft_label import ...`, which (unlike our standalone-loaded copy above)
    would trigger the very src/__init__.py import chain this script exists to avoid."""
    keys, paths = [], []
    for k in datasets:
        if k in exclude_datasets:
            continue
        filename = _constants.DATASET_META_FILENAMES[k][split]
        if filename is None:
            print(f"NOTE: {k} has no {split!r} split, skipping.")
            continue
        path = os.path.join(data_root, _constants.IQA_DATASET_ARCHIVES[k][1], "metas", filename)

        try:
            samples = _gen_soft_label.load_soft_label_samples(path)
        except (OSError, ValueError) as ex:
            print(f"WARNING: dropping dataset {k!r} ({split} split) -- could not read metadata at {path}: {ex}")
            continue

        n_meta = len(samples)
        n_found = sum(1 for s in samples if os.path.isfile(os.path.join(data_root, s.image)))
        if n_found != n_meta:
            print(f"WARNING: dropping dataset {k!r} ({split} split) -- only {n_found}/{n_meta} images "
                  f"referenced in {path} were found on disk under {data_root!r}.")
            continue

        keys.append(k)
        paths.append(path)
    if not keys:
        raise ValueError(f"No selected dataset has a {split!r} split (after --exclude-datasets)")
    return keys, paths


def predict_dataset_timed(iqa_model, key, path, image_folder, batch_size, max_samples, sample_fraction, sample_seed):
    samples = _gen_soft_label.load_soft_label_samples(path)
    if sample_fraction is not None:
        max_samples = max(1, round(sample_fraction * len(samples)))
    if max_samples is not None and len(samples) > max_samples:
        samples = random.Random(sample_seed).sample(samples, max_samples)
    print(f"\n[{key}] evaluating {len(samples)} samples from {path}")

    preds, gts = [], []
    total_time = 0.0
    n_total, n_skipped = len(samples), 0
    for i in range(0, len(samples), batch_size):
        chunk = samples[i:i + batch_size]
        images, kept = [], []
        for s in chunk:
            try:
                images.append(Image.open(os.path.join(image_folder, s.image)).convert("RGB"))
            except (FileNotFoundError, OSError) as ex:
                print(f"WARNING: skipping {s.image}: {ex}")
                n_skipped += 1
                continue
            kept.append(s)
        if not images:
            continue

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        _, scores = iqa_model.predict(images)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_time += time.perf_counter() - start

        preds.extend(scores.tolist())
        gts.extend(s.gt_score_norm for s in kept)
    if n_skipped:
        print(f"  ({n_skipped}/{n_total} images skipped -- not found locally)")
    return np.array(preds, dtype=np.float64), np.array(gts, dtype=np.float64), total_time


def compute_metrics(preds, gts):
    if len(preds) < 2:
        return None
    srcc, plcc = _gen_soft_label.calculate_srcc_plcc(preds, gts)
    return {
        "n": len(preds),
        "srcc": float(srcc),
        "plcc": float(plcc),
        "mae": float(np.mean(np.abs(preds - gts))),
        "rmse": float(np.sqrt(np.mean((preds - gts) ** 2))),
    }


def print_table(results, pooled):
    header = f"{'Dataset':<12}{'N':>6}{'SRCC':>8}{'PLCC':>8}{'MAE':>8}{'RMSE':>8}{'ms/img':>10}"
    print("\n[deqa]")
    print(header)
    print("-" * len(header))
    for key, m in results.items():
        print(f"{key:<12}{m['n']:>6}{m['srcc']:>8.3f}{m['plcc']:>8.3f}{m['mae']:>8.3f}{m['rmse']:>8.3f}"
              f"{m['avg_ms_per_image']:>10.1f}")
    print("-" * len(header))
    print(f"{'ALL (pooled)':<12}{pooled['n']:>6}{pooled['srcc']:>8.3f}{pooled['plcc']:>8.3f}"
          f"{pooled['mae']:>8.3f}{pooled['rmse']:>8.3f}{pooled['avg_ms_per_image']:>10.1f}")


def evaluate(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    iqa_model = DeQAScoreIQA(device, load_in_8bit=not args.no_8bit)

    results = {}
    all_preds, all_gts, total_time_all = [], [], 0.0
    for key, path in zip(args.dataset_keys, args.data_path):
        preds, gts, total_time = predict_dataset_timed(
            iqa_model, key, path, args.image_folder, args.batch_size,
            args.max_samples, args.sample_fraction, args.sample_seed)
        metrics = compute_metrics(preds, gts)
        if metrics is None:
            print(f"  WARNING: fewer than 2 usable samples for {key!r}, skipping")
            continue
        metrics["avg_ms_per_image"] = 1000 * total_time / metrics["n"]
        results[key] = metrics
        all_preds.append(preds)
        all_gts.append(gts)
        total_time_all += total_time
        print(f"  n={metrics['n']}  srcc={metrics['srcc']:.3f}  plcc={metrics['plcc']:.3f}"
              f"  mae={metrics['mae']:.3f}  rmse={metrics['rmse']:.3f}"
              f"  avg_inference={metrics['avg_ms_per_image']:.1f}ms/img")

    if not results:
        raise ValueError("No dataset produced usable evaluation results")

    pooled = compute_metrics(np.concatenate(all_preds), np.concatenate(all_gts))
    pooled["avg_ms_per_image"] = 1000 * total_time_all / pooled["n"]
    print_table(results, pooled)

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump({"split": args.split, "results": {"deqa": {"per_dataset": results, "pooled": pooled}}},
                       f, indent=2)
        print(f"\nResults saved to {args.out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=_constants.DATASET_KEYS,
                         default=_constants.DATASET_KEYS_DEFAULT,
                         help="Datasets to evaluate (default: standard eval set with koniq1024)")
    parser.add_argument("--exclude-datasets", nargs="+", choices=_constants.DATASET_KEYS, default=[])
    parser.add_argument("--data-root", default=_constants.DATA_DEQA_SCORE_DIR_DEFAULT)
    parser.add_argument("--split", choices=["train", "test"], default=_constants.EVAL_SPLIT_DEFAULT)
    parser.add_argument("--max-samples", type=int, default=None,
                         help="Cap the number of samples evaluated per dataset (random subset). "
                              "Ignored if --sample-fraction is given.")
    parser.add_argument("--sample-fraction", type=float, default=None,
                         help="Keep this fraction (0, 1] of each dataset's samples (random subset), "
                              "computed per-dataset.")
    parser.add_argument("--sample-seed", type=int, default=_constants.TRAIN_SAMPLE_SEED_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=_constants.EVAL_BATCH_SIZE_DEFAULT)
    parser.add_argument("--no-8bit", action="store_true",
                         help="Load in fp16/fp32 instead of 8-bit (needs ~14GB VRAM)")
    parser.add_argument("--out-json", default=OUT_JSON_DEFAULT,
                         help="Path to dump per-dataset metrics as JSON (empty string to skip)")
    args = parser.parse_args()
    args.dataset_keys, args.data_path = resolve_dataset_paths(
        args.datasets, args.exclude_datasets, args.data_root, args.split)
    args.image_folder = args.data_root
    evaluate(args)
