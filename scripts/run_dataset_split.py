import argparse
import json
import os
import random
import sys

from PIL import Image

sys.path.insert(0, ".")
from src.constants import (
    DATASET_SELECT_ARG_SPECS,
    EVAL_BATCH_SIZE_DEFAULT,
    EVAL_SPLIT_DEFAULT,
    TRAIN_SAMPLE_SEED_DEFAULT,
    analysis_result_path,
    resolve_dataset_paths,
)
from src.datasets.gen_soft_label import load_soft_label_samples
from src.trainer import get_device
import script_utils

FAILURE_THRESHOLD_DEFAULT = 1.0
SUCCESS_THRESHOLD_DEFAULT = 0.1


def load_pooled_samples(dataset_keys, data_paths, max_samples, sample_seed):
    """Loads samples from every dataset, tags each with its dataset key, caps each
    dataset at `max_samples` (random subset) if given, and shuffles the pooled list
    so a --target-count cutoff isn't biased toward whichever dataset comes first."""
    pooled = []
    for key, path in zip(dataset_keys, data_paths):
        samples = load_soft_label_samples(path)
        if max_samples is not None and len(samples) > max_samples:
            samples = random.Random(sample_seed).sample(samples, max_samples)
        pooled.extend((key, s) for s in samples)
    random.Random(sample_seed).shuffle(pooled)
    return pooled


def split_dataset(iqa_model, pooled, image_folder, batch_size, failure_threshold, success_threshold, target_count):
    """Runs the model over `pooled` [(dataset_key, sample), ...] in chunks of `batch_size`,
    bucketing each sample into failures (|pred - gt| >= failure_threshold) or successes
    (|pred - gt| < success_threshold). Samples in between are discarded from both buckets.
    Stops early once both buckets reach `target_count` (if given)."""
    failures, successes = [], []
    n_skipped = 0
    for i in range(0, len(pooled), batch_size):
        if target_count is not None and len(failures) >= target_count and len(successes) >= target_count:
            break
        chunk = pooled[i:i + batch_size]
        images, kept = [], []
        for key, s in chunk:
            try:
                images.append(Image.open(os.path.join(image_folder, s.image)).convert("RGB"))
            except (FileNotFoundError, OSError) as ex:
                print(f"WARNING: skipping {s.image}: {ex}")
                n_skipped += 1
                continue
            kept.append((key, s))
        if not images:
            continue
        _, scores = iqa_model.predict(images)
        for (key, s), pred in zip(kept, scores.tolist()):
            diff = abs(pred - s.gt_score_norm)
            record = {"dataset": key, "image": s.image, "pred": pred, "gt": s.gt_score_norm, "abs_error": diff}
            if diff >= failure_threshold:
                if target_count is None or len(failures) < target_count:
                    failures.append(record)
            elif diff < success_threshold:
                if target_count is None or len(successes) < target_count:
                    successes.append(record)
    if n_skipped:
        print(f"  ({n_skipped} images skipped — not found locally)")
    return failures, successes


def main(args):
    device = get_device()
    print(f"Device: {device}")

    iqa_model = script_utils.load_model(args, device)

    pooled = load_pooled_samples(args.dataset_keys, args.data_path, args.max_samples, args.sample_seed)
    print(f"Pooled {len(pooled)} samples across {len(args.dataset_keys)} dataset(s)")

    failures, successes = split_dataset(
        iqa_model, pooled, args.image_folder, args.batch_size,
        args.failure_threshold, args.success_threshold, args.target_count)

    print(f"\n{len(failures)} failures (|pred - gt| >= {args.failure_threshold}), "
          f"{len(successes)} successes (|pred - gt| < {args.success_threshold})")
    if args.target_count is not None and (len(failures) < args.target_count or len(successes) < args.target_count):
        print(f"WARNING: ran out of pooled samples before reaching --target-count {args.target_count} on both sides")

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({
            "model_path": args.model_path,
            "model_type": iqa_model.model_type,
            "split": args.split,
            "failure_threshold": args.failure_threshold,
            "success_threshold": args.success_threshold,
            "target_count": args.target_count,
            "failures": failures,
            "successes": successes,
        }, f, indent=2)
    print(f"Saved to {args.out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Like run_eval.py, but splits images into failure/success buckets "
                     "(|pred mos - gt mos| >= --failure-threshold) instead of computing correlations")
    script_utils.add_model_args(parser)
    for arg_spec in DATASET_SELECT_ARG_SPECS:
        parser.add_argument(*arg_spec["flags"], **arg_spec["kwargs"])
    parser.add_argument("--split", choices=["train", "test"], default=EVAL_SPLIT_DEFAULT,
                        help="Evaluate against each dataset's train or test split")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap the number of samples pooled per dataset (random subset) before shuffling")
    parser.add_argument("--sample-seed", type=int, default=TRAIN_SAMPLE_SEED_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=EVAL_BATCH_SIZE_DEFAULT)
    parser.add_argument("--failure-threshold", type=float, default=FAILURE_THRESHOLD_DEFAULT,
                        help="Minimum |pred - gt| MOS difference to count an image as a failure")
    parser.add_argument("--success-threshold", type=float, default=SUCCESS_THRESHOLD_DEFAULT,
                        help="Maximum |pred - gt| MOS difference to count an image as a success "
                             "(samples between this and --failure-threshold are discarded)")
    parser.add_argument("--target-count", type=int, default=None,
                        help="Stop once both the failure and success buckets reach this many images "
                             "(processes every pooled sample if omitted)")
    parser.add_argument("--out-json", default=analysis_result_path("dataset_split_failures.json"),
                        help="Path to write the failure/success image lists")
    args = parser.parse_args()
    args.dataset_keys, args.data_path = resolve_dataset_paths(args.datasets, args.exclude_datasets, args.data_root, args.split)
    args.image_folder = args.data_root
    main(args)
