"""Visualize which DDI images Q-Align Mini classifies into each of its five quality levels
(excellent/good/fair/poor/bad), so you can eyeball what "poor quality" vs. "good quality" means
to the model on dermatology images.

Runs Q-Align Mini over every image in ddi_metadata.csv, takes the argmax of its 5-way level
distribution per image as the predicted label, and saves:
  - a per-image CSV of predicted level/score/confidence (merged with skin_tone/malignant/disease)
  - a bar chart of how many images fall into each level
  - a thumbnail grid with a row per level, so you can look at actual example images per bucket

Usage:
    python scripts/visualize_ddi_qalign_quality.py [--image-dir DIR] [--metadata CSV]
        [--output-dir DIR] [--max-samples N] [--samples-per-level N]
"""
import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, ".")
from src.model.qalign import constants as qalign_constants
from src.trainer import get_device
import script_utils

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEVEL_LABELS = [level.capitalize() for level in qalign_constants.QALIGN_LEVELS]


def score_ddi_images(iqa_model, df, image_dir, batch_size):
    """Runs `iqa_model` over every DDI_file in `df`, returning a new dataframe with quality_score,
    predicted_level, and confidence (the winning level's probability) columns appended. Drops
    (with a warning) any row whose image is missing/unreadable."""
    rows = list(df.itertuples(index=False))
    results = {}
    n_skipped = 0
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        images, kept = [], []
        for row in chunk:
            try:
                images.append(Image.open(os.path.join(image_dir, row.DDI_file)).convert("RGB"))
            except (FileNotFoundError, OSError) as ex:
                print(f"WARNING: skipping {row.DDI_file}: {ex}")
                n_skipped += 1
                continue
            kept.append(row)
        if not images:
            continue

        probs, scores = iqa_model.predict(images)
        for row, prob, score in zip(kept, probs, scores):
            level_idx = int(np.argmax(prob))
            results[row.DDI_file] = {
                "quality_score": float(score),
                "predicted_level": LEVEL_LABELS[level_idx],
                "confidence": float(prob[level_idx]),
            }
        print(f"  scored {min(i + batch_size, len(rows))}/{len(rows)}")

    if n_skipped:
        print(f"({n_skipped}/{len(rows)} images skipped — not found locally)")

    out = df[df["DDI_file"].isin(results)].copy()
    out["quality_score"] = out["DDI_file"].map(lambda f: results[f]["quality_score"])
    out["predicted_level"] = out["DDI_file"].map(lambda f: results[f]["predicted_level"])
    out["confidence"] = out["DDI_file"].map(lambda f: results[f]["confidence"])
    return out.reset_index(drop=True)


def plot_level_counts(df, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = df["predicted_level"].value_counts().reindex(LEVEL_LABELS, fill_value=0)

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.bar(counts.index, counts.values, color="steelblue")
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Q-Align Mini predicted quality level")
    ax.set_ylabel("Number of DDI images")
    ax.set_title(f"Q-Align Mini predicted quality level distribution (DDI, n={len(df)})")
    plot_path = os.path.join(output_dir, "ddi_qalign_level_counts.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved level count bar chart to {plot_path}")


def plot_level_grid(df, image_dir, output_dir, samples_per_level, seed, thumb_size):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    rng = random.Random(seed)
    rows = []
    for level in LEVEL_LABELS:
        level_files = df.loc[df["predicted_level"] == level, "DDI_file"].tolist()
        if not level_files:
            continue
        sample = rng.sample(level_files, min(samples_per_level, len(level_files)))
        rows.append((level, len(level_files), sample))

    if not rows:
        print("No predictions to plot in the level grid.")
        return

    n_cols = max(len(sample) for _, _, sample in rows)
    fig = plt.figure(figsize=(2.0 * n_cols, 2.3 * len(rows)))
    gs = gridspec.GridSpec(len(rows), n_cols, hspace=0.35, wspace=0.1)

    for r, (level, count, sample) in enumerate(rows):
        for c in range(n_cols):
            ax = fig.add_subplot(gs[r, c])
            ax.axis("off")
            if c >= len(sample):
                continue
            filename = sample[c]
            img = Image.open(os.path.join(image_dir, filename)).convert("RGB")
            img.thumbnail((thumb_size, thumb_size))
            ax.imshow(img)
            if c == 0:
                ax.set_ylabel(f"{level}\n(n={count})", rotation=0, ha="right", va="center",
                              fontsize=10, labelpad=30)
                ax.axis("on")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

    fig.suptitle("Q-Align Mini predicted quality level, with example DDI images", fontsize=13)
    plot_path = os.path.join(output_dir, "ddi_qalign_level_grid.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved example-image grid to {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize which DDI images Q-Align Mini labels as poor/fair/good/etc. quality.")
    parser.add_argument("--image-dir", default=os.path.join(REPO_ROOT, "ddi_data"))
    parser.add_argument("--metadata", default=os.path.join(REPO_ROOT, "ddi_metadata.csv"))
    parser.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "validation", "ddi_qalign"))
    parser.add_argument("--batch-size", type=int, default=16,
                         help="Chunk size for image loading; Q-Align Mini scores one image per "
                              "forward pass regardless, so this only affects I/O batching.")
    parser.add_argument("--max-samples", type=int, default=None,
                         help="Cap the number of DDI images scored (for a quick smoke test)")
    parser.add_argument("--samples-per-level", type=int, default=8,
                         help="How many example thumbnails to show per predicted quality level")
    parser.add_argument("--thumb-size", type=int, default=160,
                         help="Max width/height (pixels) of each thumbnail in the example grid")
    parser.add_argument("--seed", type=int, default=0,
                         help="Random seed for sampling which images appear in the example grid")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.metadata)[["DDI_file", "skin_tone", "malignant", "disease"]]
    if args.max_samples is not None and len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=args.seed).reset_index(drop=True)
    print(f"Loaded {len(df)} DDI images from {args.metadata}")

    device = get_device()
    print(f"Device: {device}")

    model_args = argparse.Namespace(model_type="qalign_mini", model_path=None)
    iqa_model = script_utils.load_model(model_args, device)

    df = score_ddi_images(iqa_model, df, args.image_dir, args.batch_size)

    del iqa_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if len(df) == 0:
        raise SystemExit("No DDI images could be scored")

    predictions_path = os.path.join(args.output_dir, "ddi_qalign_predictions.csv")
    df.to_csv(predictions_path, index=False)
    print(f"Saved per-image Q-Align Mini predictions to {predictions_path}")

    print("\n=== Q-Align Mini predicted quality level counts (DDI) ===")
    counts = df["predicted_level"].value_counts().reindex(LEVEL_LABELS, fill_value=0)
    for level, count in counts.items():
        print(f"{level:<10} {count:>5}  ({100 * count / len(df):.1f}%)")

    plot_level_counts(df, args.output_dir)
    plot_level_grid(df, args.image_dir, args.output_dir, args.samples_per_level, args.seed,
                     args.thumb_size)


if __name__ == "__main__":
    main()
