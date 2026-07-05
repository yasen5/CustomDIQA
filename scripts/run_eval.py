import argparse
import json
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")
from src.constants import (
    DATASET_SELECT_ARG_SPECS,
    EVAL_BATCH_SIZE_DEFAULT,
    EVAL_OUT_DEFAULT,
    EVAL_PLCC_COLOR,
    EVAL_SPLIT_DEFAULT,
    EVAL_SRCC_COLOR,
    TRAIN_SAMPLE_SEED_DEFAULT,
    resolve_dataset_paths,
)
from src.datasets.gen_soft_label import calculate_srcc_plcc, load_soft_label_samples
from src.trainer import get_device
import script_utils


def predict_dataset(iqa_model, key, path, image_folder, batch_size, max_samples, sample_seed):
    """Returns (pred_scores, gt_scores) over dataset `key`'s samples at `path`, both as float
    arrays aligned index-for-index. All image loading/sampling is delegated to
    script_utils.build_dataset (SingleDataset — cache-aware) for fixed-size-processor models, or
    script_utils.predict_raw_items (raw PIL, native resolution) for topiq_nr — see script_utils.py
    for why those two paths can't be unified further."""
    if iqa_model.model_type in ("topiq_nr", "musiq"):
        samples = load_soft_label_samples(path)
        if max_samples is not None and len(samples) > max_samples:
            samples = random.Random(sample_seed).sample(samples, max_samples)
        print(f"\n[{key}] evaluating {len(samples)} samples from {path}")
        preds, gts = [], []
        for kept, scores in script_utils.predict_raw_items(
                iqa_model, samples, image_folder, lambda s: s.image, batch_size):
            preds.extend(scores)
            gts.extend(s.gt_score_norm for s in kept)
        return np.array(preds, dtype=np.float64), np.array(gts, dtype=np.float64)

    dataset = script_utils.build_dataset(path, image_folder, iqa_model.processor)
    indices = list(range(len(dataset)))
    if max_samples is not None and len(indices) > max_samples:
        indices = random.Random(sample_seed).sample(indices, max_samples)
    n_cached = sum(1 for j in indices if dataset.preprocessed_images[j] is not None)
    cache_note = f" ({n_cached}/{len(indices)} from preprocessed cache)" if n_cached else ""
    print(f"\n[{key}] evaluating {len(indices)} samples from {path}{cache_note}")

    preds, gts = [], []
    for chunk, scores in script_utils.predict_dataset_items(iqa_model, dataset, indices, batch_size):
        preds.extend(scores)
        gts.extend(dataset.list_data_dict[j].gt_score_norm for j in chunk)
    return np.array(preds, dtype=np.float64), np.array(gts, dtype=np.float64)


def compute_metrics(preds, gts):
    if len(preds) < 2:
        return None
    srcc, plcc = calculate_srcc_plcc(preds, gts)
    return {
        "n": len(preds),
        "srcc": float(srcc),
        "plcc": float(plcc),
        "mae": float(np.mean(np.abs(preds - gts))),
        "rmse": float(np.sqrt(np.mean((preds - gts) ** 2))),
    }


def print_table(results, pooled):
    header = f"{'Dataset':<12}{'N':>6}{'SRCC':>8}{'PLCC':>8}{'MAE':>8}{'RMSE':>8}"
    print(f"\n{header}")
    print("-" * len(header))
    for key, m in results.items():
        print(f"{key:<12}{m['n']:>6}{m['srcc']:>8.3f}{m['plcc']:>8.3f}{m['mae']:>8.3f}{m['rmse']:>8.3f}")
    print("-" * len(header))
    print(f"{'ALL (pooled)':<12}{pooled['n']:>6}{pooled['srcc']:>8.3f}{pooled['plcc']:>8.3f}"
          f"{pooled['mae']:>8.3f}{pooled['rmse']:>8.3f}")


def make_plot(results, out_path, model_type):
    keys = list(results.keys())
    srcc = [results[k]["srcc"] for k in keys]
    plcc = [results[k]["plcc"] for k in keys]

    x = np.arange(len(keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(keys)), 4.5))
    bars_srcc = ax.bar(x - width / 2, srcc, width, color=EVAL_SRCC_COLOR, label="SRCC")
    bars_plcc = ax.bar(x + width / 2, plcc, width, color=EVAL_PLCC_COLOR, label="PLCC")

    for bars in (bars_srcc, bars_plcc):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", (bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(keys, fontsize=9)
    ax.set_ylabel("Correlation")
    lo = min(0.0, min(srcc + plcc) - 0.1)
    ax.set_ylim(lo, 1.05)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.legend(fontsize=9)
    ax.set_title(f"{model_type.upper()} — Cross-Dataset Performance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nChart saved to {out_path}")


def evaluate(args):
    device = get_device()
    print(f"Device: {device}")

    iqa_model = script_utils.load_model(args, device)

    if iqa_model.model_type == "topiq_nr":
        print("NOTE: topiq_nr scores are linearly rescaled from pyiqa's documented ~0-1 range onto "
              "this repo's 1-5 GT scale (an uncalibrated affine transform, not fit to this dataset's "
              "actual MOS distribution) — MAE/RMSE below are roughly comparable to vit/cnn runs but "
              "shouldn't be over-interpreted; SRCC/PLCC remain the most trustworthy comparison.")

    results = {}
    all_preds, all_gts = [], []
    for key, path in zip(args.dataset_keys, args.data_path):
        preds, gts = predict_dataset(iqa_model, key, path, args.image_folder, args.batch_size,
                                      args.max_samples, args.sample_seed)
        metrics = compute_metrics(preds, gts)
        if metrics is None:
            print(f"  WARNING: fewer than 2 usable samples for {key!r}, skipping")
            continue
        results[key] = metrics
        all_preds.append(preds)
        all_gts.append(gts)
        print(f"  n={metrics['n']}  srcc={metrics['srcc']:.3f}  plcc={metrics['plcc']:.3f}"
              f"  mae={metrics['mae']:.3f}  rmse={metrics['rmse']:.3f}")

    if not results:
        raise ValueError("No dataset produced usable evaluation results")

    pooled = compute_metrics(np.concatenate(all_preds), np.concatenate(all_gts))
    print_table(results, pooled)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        make_plot(results, args.out, iqa_model.model_type)

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump({"model_path": args.model_path, "model_type": iqa_model.model_type,
                       "split": args.split, "per_dataset": results, "pooled": pooled}, f, indent=2)
        print(f"Results saved to {args.out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint's SRCC/PLCC/MAE/RMSE per dataset")
    script_utils.add_model_args(parser)
    for arg_spec in DATASET_SELECT_ARG_SPECS:
        parser.add_argument(*arg_spec["flags"], **arg_spec["kwargs"])
    parser.add_argument("--split", choices=["train", "test"], default=EVAL_SPLIT_DEFAULT,
                        help="Evaluate against each dataset's train or test split")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap the number of samples evaluated per dataset (random subset)")
    parser.add_argument("--sample-seed", type=int, default=TRAIN_SAMPLE_SEED_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=EVAL_BATCH_SIZE_DEFAULT)
    parser.add_argument("--out", default=EVAL_OUT_DEFAULT, help="Bar chart output path (empty string to skip)")
    parser.add_argument("--out-json", default=None, help="Optional path to dump per-dataset metrics as JSON")
    args = parser.parse_args()
    args.dataset_keys, args.data_path = resolve_dataset_paths(args.datasets, args.exclude_datasets, args.data_root, args.split)
    args.image_folder = args.data_root
    evaluate(args)
