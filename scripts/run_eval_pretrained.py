import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, ".")
from src.constants import (
    ANALYSIS_RESULTS_DIR,
    DATASET_SELECT_ARG_SPECS,
    EVAL_BATCH_SIZE_DEFAULT,
    EVAL_SPLIT_DEFAULT,
    TRAIN_SAMPLE_SEED_DEFAULT,
    resolve_dataset_paths,
)
from src.datasets.gen_soft_label import load_soft_label_samples
from src.trainer import get_device
import script_utils
from run_eval import compute_metrics

OUT_JSON_DEFAULT = os.path.join(ANALYSIS_RESULTS_DIR, "eval_pretrained.json")


def predict_dataset_timed(iqa_model, key, path, image_folder, batch_size, max_samples, sample_seed,
                           sample_fraction=None):
    """Runs `iqa_model` (topiq_nr/musiq/qalign_mini — all native-resolution, raw-PIL models; see
    script_utils.predict_raw_items) over dataset `key`'s samples at `path`. Returns (pred_scores,
    gt_scores, total_seconds), where total_seconds is the summed wall-clock time spent inside
    iqa_model.predict (forward passes only — image decode/load happens before the timer starts
    each batch, so I/O speed doesn't pollute the inference-time comparison across models).

    sample_fraction (0, 1], if given, takes precedence over max_samples and is applied per-dataset
    (e.g. 0.25 keeps a random quarter of *this* dataset's samples, rather than max_samples' fixed
    count shared across every dataset regardless of size)."""
    samples = load_soft_label_samples(path)
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
        print(f"  ({n_skipped}/{n_total} images skipped — not found locally)")
    return np.array(preds, dtype=np.float64), np.array(gts, dtype=np.float64), total_time


def print_table(model_type, results, pooled):
    header = f"{'Dataset':<12}{'N':>6}{'SRCC':>8}{'PLCC':>8}{'MAE':>8}{'RMSE':>8}{'ms/img':>10}"
    print(f"\n[{model_type}]")
    print(header)
    print("-" * len(header))
    for key, m in results.items():
        print(f"{key:<12}{m['n']:>6}{m['srcc']:>8.3f}{m['plcc']:>8.3f}{m['mae']:>8.3f}{m['rmse']:>8.3f}"
              f"{m['avg_ms_per_image']:>10.1f}")
    print("-" * len(header))
    print(f"{'ALL (pooled)':<12}{pooled['n']:>6}{pooled['srcc']:>8.3f}{pooled['plcc']:>8.3f}"
          f"{pooled['mae']:>8.3f}{pooled['rmse']:>8.3f}{pooled['avg_ms_per_image']:>10.1f}")


def print_timing_summary(all_results, dataset_keys):
    model_types = list(all_results.keys())
    col_width = max(10, max(len(m) for m in model_types) + 2)
    header = f"{'Dataset':<12}" + "".join(f"{m:>{col_width}}" for m in model_types)
    print("\nAverage inference time per image (ms), by dataset:")
    print(header)
    print("-" * len(header))
    for key in dataset_keys:
        row = f"{key:<12}"
        for model_type in model_types:
            m = all_results[model_type]["per_dataset"].get(key)
            row += f"{m['avg_ms_per_image']:>{col_width}.1f}" if m else f"{'--':>{col_width}}"
        print(row)
    print("-" * len(header))
    row = f"{'ALL (pooled)':<12}"
    for model_type in model_types:
        row += f"{all_results[model_type]['pooled']['avg_ms_per_image']:>{col_width}.1f}"
    print(row)


def evaluate(args):
    device = get_device()
    print(f"Device: {device}")

    all_results = {}
    for model_type in args.model_types:
        print(f"\n{f' Evaluating pretrained model: {model_type} ':=^88}")
        model_args = argparse.Namespace(model_type=model_type, model_path=None)
        iqa_model = script_utils.load_model(model_args, device)

        results = {}
        all_preds, all_gts, total_time_all = [], [], 0.0
        for key, path in zip(args.dataset_keys, args.data_path):
            preds, gts, total_time = predict_dataset_timed(
                iqa_model, key, path, args.image_folder, args.batch_size,
                args.max_samples, args.sample_seed, sample_fraction=args.sample_fraction)
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

        del iqa_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not results:
            print(f"WARNING: no usable dataset results for model {model_type!r}, skipping entirely")
            continue

        pooled = compute_metrics(np.concatenate(all_preds), np.concatenate(all_gts))
        assert pooled is not None  # >=2 samples guaranteed: each dataset above already had n>=2
        pooled["avg_ms_per_image"] = 1000 * total_time_all / pooled["n"]
        print_table(model_type, results, pooled)
        all_results[model_type] = {"per_dataset": results, "pooled": pooled}

    if not all_results:
        raise ValueError("No pretrained model produced usable evaluation results")

    print_timing_summary(all_results, args.dataset_keys)

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump({"split": args.split, "results": all_results}, f, indent=2)
        print(f"\nResults saved to {args.out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate every pretrained (no-checkpoint-required) IQA model in this repo "
                     "— topiq_nr, musiq, and qalign_mini — across datasets, reporting SRCC/PLCC/"
                     "MAE/RMSE plus average per-image inference time. (DeQA-Score is evaluated "
                     "separately: see scripts/run_eval_deqa.py, run under deqa_venv.)")
    parser.add_argument("--model-types", nargs="+", choices=script_utils.PRETRAINED_MODEL_TYPES,
                         default=list(script_utils.PRETRAINED_MODEL_TYPES),
                         help="Which pretrained models to evaluate (default: all of them)")
    for arg_spec in DATASET_SELECT_ARG_SPECS:
        parser.add_argument(*arg_spec["flags"], **arg_spec["kwargs"])
    parser.add_argument("--split", choices=["train", "test"], default=EVAL_SPLIT_DEFAULT,
                        help="Evaluate against each dataset's train or test split")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap the number of samples evaluated per dataset (random subset). "
                             "Ignored if --sample-fraction is given.")
    parser.add_argument("--sample-fraction", type=float, default=None,
                        help="Keep this fraction (0, 1] of each dataset's samples (random subset), "
                             "computed per-dataset rather than --max-samples' fixed count shared "
                             "across every dataset regardless of size.")
    parser.add_argument("--sample-seed", type=int, default=TRAIN_SAMPLE_SEED_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=EVAL_BATCH_SIZE_DEFAULT,
                        help="Chunk size for image loading; these models score one image per "
                             "forward pass regardless (see script_utils.QAlignMiniModel/"
                             "TopiqNRModel/MusiqModel), so this only affects I/O batching.")
    parser.add_argument("--out-json", default=OUT_JSON_DEFAULT,
                        help="Path to dump per-model per-dataset metrics as JSON (empty string to skip)")
    args = parser.parse_args()
    args.dataset_keys, args.data_path = resolve_dataset_paths(
        args.datasets, args.exclude_datasets, args.data_root, args.split)
    args.image_folder = args.data_root
    evaluate(args)
