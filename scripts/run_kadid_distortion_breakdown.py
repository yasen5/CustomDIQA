import argparse
import json
import os
import random
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")
from src.constants import (
    DATA_DEQA_SCORE_DIR_DEFAULT,
    EVAL_BATCH_SIZE_DEFAULT,
    EVAL_SPLIT_DEFAULT,
    EVAL_SRCC_COLOR,
    KADID_DISTORTION_NAMES,
    TRAIN_SAMPLE_SEED_DEFAULT,
    analysis_result_path,
    resolve_dataset_paths,
)
from src.datasets.gen_soft_label import calculate_srcc_plcc, load_soft_label_samples
from src.trainer import get_device
import script_utils

FAILURE_THRESHOLD_DEFAULT = 1.0
SUCCESS_THRESHOLD_DEFAULT = 0.1
OUT_TYPE_PLOT_DEFAULT = analysis_result_path("kadid_distortion_breakdown.png")
OUT_LEVEL_PLOT_DEFAULT = analysis_result_path("kadid_level_breakdown.png")
OUT_JSON_DEFAULT = analysis_result_path("kadid_distortion_breakdown.json")

# KADID images are named "I<ref>_<distortion 01-25>_<level 01-05>.png", e.g. "I48_16_05.png".
KADID_FILENAME_RE = re.compile(r"^I\d+_(\d+)_(\d+)\.png$")


def parse_kadid_filename(image_path):
    """Extracts (distortion_type, level) ints from a KADID image path's basename
    (e.g. 'KADID10K/images/I48_16_05.png' -> (16, 5)). Returns None if the basename
    doesn't match KADID's 'I<ref>_<type>_<level>.png' naming convention."""
    m = KADID_FILENAME_RE.match(os.path.basename(image_path))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _make_record(sample, distortion_type, level, pred):
    return {
        "image": sample.image,
        "distortion_type": distortion_type,
        "level": level,
        "pred": pred,
        "gt": sample.gt_score_norm,
        "abs_error": abs(pred - sample.gt_score_norm),
    }


def predict_kadid(iqa_model, path, image_folder, batch_size, max_samples, sample_seed):
    """Returns one record per usable sample: {image, distortion_type, level, pred, gt,
    abs_error}. Images whose filename doesn't match KADID's naming convention are dropped.
    All other image loading/sampling is delegated to script_utils (SingleDataset-backed cache
    for fixed-size-processor models, raw PIL for topiq_nr) — see script_utils.py."""
    if iqa_model.model_type == "topiq_nr":
        samples = load_soft_label_samples(path)
        parsed_samples = [(s, parse_kadid_filename(s.image)) for s in samples]
        n_unparsed = sum(1 for _, p in parsed_samples if p is None)
        parsed_samples = [(s, p) for s, p in parsed_samples if p is not None]
        if max_samples is not None and len(parsed_samples) > max_samples:
            parsed_samples = random.Random(sample_seed).sample(parsed_samples, max_samples)
        print(f"Evaluating {len(parsed_samples)} KADID samples from {path}")
        if n_unparsed:
            print(f"  ({n_unparsed} images skipped — filename didn't match KADID's naming convention)")

        records = []
        for kept, scores in script_utils.predict_raw_items(
                iqa_model, parsed_samples, image_folder, lambda item: item[0].image, batch_size):
            for (s, (distortion_type, level)), pred in zip(kept, scores):
                records.append(_make_record(s, distortion_type, level, pred))
        return records

    dataset = script_utils.build_dataset(path, image_folder, iqa_model.processor)
    parsed_by_index = {}
    for j, s in enumerate(dataset.list_data_dict):
        parsed = parse_kadid_filename(s.image)
        if parsed is not None:
            parsed_by_index[j] = parsed
    n_unparsed = len(dataset) - len(parsed_by_index)

    indices = list(parsed_by_index)
    if max_samples is not None and len(indices) > max_samples:
        indices = random.Random(sample_seed).sample(indices, max_samples)
    print(f"Evaluating {len(indices)} KADID samples from {path}")
    if n_unparsed:
        print(f"  ({n_unparsed} images skipped — filename didn't match KADID's naming convention)")

    records = []
    for chunk, scores in script_utils.predict_dataset_items(iqa_model, dataset, indices, batch_size):
        for j, pred in zip(chunk, scores):
            distortion_type, level = parsed_by_index[j]
            records.append(_make_record(dataset.list_data_dict[j], distortion_type, level, pred))
    return records


def summarize(records, key_fn, failure_threshold, success_threshold):
    """Groups `records` by key_fn(record) and computes per-group n, mean/median
    abs_error, failure_rate (fraction with abs_error >= failure_threshold),
    success_rate (fraction with abs_error < success_threshold), and srcc
    (None if fewer than 2 records in the group)."""
    groups = {}
    for r in records:
        groups.setdefault(key_fn(r), []).append(r)

    summary = {}
    for key, group in groups.items():
        errors = np.array([r["abs_error"] for r in group], dtype=np.float64)
        preds = np.array([r["pred"] for r in group], dtype=np.float64)
        gts = np.array([r["gt"] for r in group], dtype=np.float64)
        srcc = None
        if len(group) >= 2:
            srcc, _ = calculate_srcc_plcc(preds, gts)
            srcc = float(srcc)
        summary[key] = {
            "n": len(group),
            "mean_abs_error": float(np.mean(errors)),
            "median_abs_error": float(np.median(errors)),
            "failure_rate": float(np.mean(errors >= failure_threshold)),
            "success_rate": float(np.mean(errors < success_threshold)),
            "srcc": srcc,
        }
    return summary


def _fmt_srcc(srcc):
    return f"{srcc:>8.3f}" if srcc is not None else f"{'n/a':>8}"


def print_type_table(by_type):
    header = f"{'Type':<5}{'Name':<28}{'N':>6}{'MeanErr':>9}{'MedErr':>8}{'FailRate':>9}{'SuccRate':>9}{'SRCC':>9}"
    print(f"\n{header}")
    print("-" * len(header))
    for t, m in sorted(by_type.items(), key=lambda kv: kv[1]["mean_abs_error"], reverse=True):
        name = KADID_DISTORTION_NAMES[t - 1] if 1 <= t <= len(KADID_DISTORTION_NAMES) else f"type {t}"
        print(f"{t:<5}{name:<28}{m['n']:>6}{m['mean_abs_error']:>9.3f}{m['median_abs_error']:>8.3f}"
              f"{m['failure_rate']:>9.3f}{m['success_rate']:>9.3f}{_fmt_srcc(m['srcc'])}")


def print_level_table(by_level):
    header = f"{'Level':<7}{'N':>6}{'MeanErr':>9}{'MedErr':>8}{'FailRate':>9}{'SuccRate':>9}{'SRCC':>9}"
    print(f"\n{header}")
    print("-" * len(header))
    for level in sorted(by_level):
        m = by_level[level]
        print(f"{level:<7}{m['n']:>6}{m['mean_abs_error']:>9.3f}{m['median_abs_error']:>8.3f}"
              f"{m['failure_rate']:>9.3f}{m['success_rate']:>9.3f}{_fmt_srcc(m['srcc'])}")


def _bar_plot(labels, mean_errors, failure_rates, out_path, title):
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(labels)), 5))
    bars = ax.bar(x, mean_errors, color=EVAL_SRCC_COLOR)
    for bar, rate in zip(bars, failure_rates):
        h = bar.get_height()
        ax.annotate(f"{rate:.2f}", (bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=60, ha="right")
    ax.set_ylabel("Mean |pred - gt|")
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.set_title(f"{title} (numbers above bars = failure rate)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Chart saved to {out_path}")


def make_type_plot(by_type, out_path, model_type):
    ordered = sorted(by_type.items(), key=lambda kv: kv[1]["mean_abs_error"], reverse=True)
    labels = [KADID_DISTORTION_NAMES[t - 1] if 1 <= t <= len(KADID_DISTORTION_NAMES) else f"type {t}"
              for t, _ in ordered]
    mean_errors = [m["mean_abs_error"] for _, m in ordered]
    failure_rates = [m["failure_rate"] for _, m in ordered]
    _bar_plot(labels, mean_errors, failure_rates, out_path,
              f"{model_type.upper()} — KADID error by distortion type")


def make_level_plot(by_level, out_path, model_type):
    levels = sorted(by_level)
    labels = [f"Level {level}" for level in levels]
    mean_errors = [by_level[level]["mean_abs_error"] for level in levels]
    failure_rates = [by_level[level]["failure_rate"] for level in levels]
    _bar_plot(labels, mean_errors, failure_rates, out_path,
              f"{model_type.upper()} — KADID error by severity level")


def main(args):
    device = get_device()
    print(f"Device: {device}")

    iqa_model = script_utils.load_model(args, device)

    _, data_paths = resolve_dataset_paths(["kadid"], [], args.data_root, args.split)
    records = predict_kadid(iqa_model, data_paths[0], args.data_root, args.batch_size,
                             args.max_samples, args.sample_seed)
    if len(records) < 2:
        raise ValueError("Fewer than 2 usable KADID samples, cannot summarize")

    by_type = summarize(records, lambda r: r["distortion_type"], args.failure_threshold, args.success_threshold)
    by_level = summarize(records, lambda r: r["level"], args.failure_threshold, args.success_threshold)
    overall = summarize(records, lambda r: "overall", args.failure_threshold, args.success_threshold)["overall"]

    print_type_table(by_type)
    print_level_table(by_level)
    print(f"\nOverall: n={overall['n']}  mean_err={overall['mean_abs_error']:.3f}  "
          f"median_err={overall['median_abs_error']:.3f}  failure_rate={overall['failure_rate']:.3f}  "
          f"success_rate={overall['success_rate']:.3f}  srcc={overall['srcc']}")

    if args.out:
        make_type_plot(by_type, args.out, iqa_model.model_type)
    if args.out_level:
        make_level_plot(by_level, args.out_level, iqa_model.model_type)

    if args.out_json:
        by_type_list = [
            {"type": t, "name": KADID_DISTORTION_NAMES[t - 1] if 1 <= t <= len(KADID_DISTORTION_NAMES) else None, **m}
            for t, m in sorted(by_type.items())
        ]
        by_level_list = [{"level": level, **by_level[level]} for level in sorted(by_level)]
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump({
                "model_path": args.model_path,
                "model_type": iqa_model.model_type,
                "split": args.split,
                "failure_threshold": args.failure_threshold,
                "success_threshold": args.success_threshold,
                "n_total": overall["n"],
                "by_distortion_type": by_type_list,
                "by_level": by_level_list,
                "overall": overall,
            }, f, indent=2)
        print(f"Results saved to {args.out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Break down a model's KADID prediction error by synthetic distortion type and severity level")
    script_utils.add_model_args(parser)
    parser.add_argument("--data-root", default=DATA_DEQA_SCORE_DIR_DEFAULT,
                         help="Path to the Data-DeQA-Score directory (doubles as the image root)")
    parser.add_argument("--split", choices=["train", "test"], default=EVAL_SPLIT_DEFAULT,
                         help="Evaluate against KADID's train or test split")
    parser.add_argument("--max-samples", type=int, default=None,
                         help="Cap the number of KADID samples evaluated (random subset)")
    parser.add_argument("--sample-seed", type=int, default=TRAIN_SAMPLE_SEED_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=EVAL_BATCH_SIZE_DEFAULT)
    parser.add_argument("--failure-threshold", type=float, default=FAILURE_THRESHOLD_DEFAULT,
                         help="Minimum |pred - gt| MOS difference to count an image as a failure")
    parser.add_argument("--success-threshold", type=float, default=SUCCESS_THRESHOLD_DEFAULT,
                         help="Maximum |pred - gt| MOS difference to count an image as a success")
    parser.add_argument("--out", default=OUT_TYPE_PLOT_DEFAULT,
                         help="Per-distortion-type bar chart output path (empty string to skip)")
    parser.add_argument("--out-level", default=OUT_LEVEL_PLOT_DEFAULT,
                         help="Per-severity-level bar chart output path (empty string to skip)")
    parser.add_argument("--out-json", default=OUT_JSON_DEFAULT,
                         help="Path to write the full breakdown as JSON (empty string to skip)")
    args = parser.parse_args()
    args.image_folder = args.data_root
    main(args)
