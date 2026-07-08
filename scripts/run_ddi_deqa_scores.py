"""Scores DDI images with DeQA-Score-Mix3 and caches them to CSV, so that scripts running under
the main custom_venv (e.g. run_ddi_pairwise_agreement.py, run_ddi_model_agreement.py) can include
DeQA in cross-model comparisons without constructing it in-process -- see
src/model/deqa/constants.py for why DeQA can only be built inside deqa_venv, and
scripts/run_eval_deqa.py for the same one-process-per-venv split applied to dataset eval.

Must run under deqa_venv, not the main custom_venv:

    deqa_venv/bin/python3 scripts/run_ddi_deqa_scores.py

Like run_eval_deqa.py, this deliberately avoids `import src...` (src/__init__.py eagerly imports
pyiqa-/torchvision-dependent code not installed in deqa_venv): src/model/deqa is imported as a bare
top-level package.

Usage:
    deqa_venv/bin/python3 scripts/run_ddi_deqa_scores.py [--image-dir DIR] [--metadata CSV]
        [--files FILE [FILE ...]] [--out-csv CSV] [--batch-size N] [--no-8bit]
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "model"))
from deqa import DeQAScoreIQA  # noqa: E402  (src/model/deqa, imported as a bare top-level package)

OUT_CSV_DEFAULT = str(REPO_ROOT / "validation" / "ddi_deqa_scores.csv")


def load_ddi_files(metadata_path):
    import csv as _csv
    with open(metadata_path, newline="") as f:
        return [row["DDI_file"] for row in _csv.DictReader(f)]


def load_existing_cache(out_csv):
    if not os.path.exists(out_csv):
        return {}
    with open(out_csv, newline="") as f:
        return {row["DDI_file"]: float(row["deqa_score"]) for row in csv.DictReader(f)}


def score_files(iqa_model, files, image_dir, batch_size):
    scores = {}
    for i in range(0, len(files), batch_size):
        chunk = files[i:i + batch_size]
        images, kept = [], []
        for fname in chunk:
            try:
                images.append(Image.open(os.path.join(image_dir, fname)).convert("RGB"))
            except (FileNotFoundError, OSError) as ex:
                print(f"WARNING: skipping {fname}: {ex}")
                continue
            kept.append(fname)
        if not images:
            continue
        _, batch_scores = iqa_model.predict(images)
        for fname, score in zip(kept, batch_scores.tolist()):
            scores[fname] = score
        print(f"  scored {min(i + batch_size, len(files))}/{len(files)}")
    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Score DDI images with DeQA-Score-Mix3 and cache the results to CSV, for "
                     "later use by pairwise/inter-model agreement scripts run under custom_venv.")
    parser.add_argument("--image-dir", default=str(REPO_ROOT / "ddi_data"))
    parser.add_argument("--metadata", default=str(REPO_ROOT / "ddi_metadata.csv"))
    parser.add_argument("--files", nargs="+", default=None,
                         help="Score only these DDI filenames (default: every image in --metadata)")
    parser.add_argument("--out-csv", default=OUT_CSV_DEFAULT,
                         help="Cache path (CSV with DDI_file,deqa_score columns). Existing entries "
                              "are kept and only missing files are (re-)scored, unless --overwrite.")
    parser.add_argument("--overwrite", action="store_true",
                         help="Re-score every requested file even if already present in --out-csv")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-8bit", action="store_true",
                         help="Load in fp16/fp32 instead of 8-bit (needs ~14GB VRAM)")
    args = parser.parse_args()

    files = args.files if args.files is not None else load_ddi_files(args.metadata)
    print(f"Requested {len(files)} DDI files")

    cache = {} if args.overwrite else load_existing_cache(args.out_csv)
    todo = [f for f in files if f not in cache]
    print(f"{len(files) - len(todo)} already cached in {args.out_csv}, {len(todo)} left to score")

    if todo:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")
        iqa_model = DeQAScoreIQA(device, load_in_8bit=not args.no_8bit)
        new_scores = score_files(iqa_model, todo, args.image_dir, args.batch_size)
        cache.update(new_scores)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["DDI_file", "deqa_score"])
        for fname in sorted(cache):
            writer.writerow([fname, cache[fname]])
    n_missing = len(files) - sum(1 for f in files if f in cache)
    print(f"Saved {len(cache)} cached scores to {args.out_csv}"
          + (f" ({n_missing} requested files could not be scored)" if n_missing else ""))


if __name__ == "__main__":
    main()
