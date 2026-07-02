import argparse
import os
import random
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

sys.path.insert(0, ".")
from src.constants import (
    DATA_DEQA_SCORE_DIR_DEFAULT,
    DEMO_GT_COLOR,
    DEMO_PRED_COLOR,
    DEMO_SEED_DEFAULT,
    analysis_result_path,
    resolve_dataset_paths,
)
from src.datasets.gen_soft_label import load_soft_label_samples
from src.trainer import get_device
import script_utils

# KADID images are named "I<ref>_<distortion>_<level>.png"; a "progression" is every
# level (severity 01, 02, 03, ...) of one <ref>_<distortion> pair, e.g. I48_05_01..05.
PROGRESSION_RE = re.compile(r"^(.*)_(\d+)\.png$")


def load_tagged_samples(data_root):
    """Pool KADID samples from both the train and test splits, tagging each with
    the split it actually came from (so a progression can span both)."""
    tagged = []
    for split in ("train", "test"):
        _, data_paths = resolve_dataset_paths(["kadid"], [], data_root, split)
        for s in load_soft_label_samples(data_paths[0]):
            if s.level_probs is not None:
                tagged.append((s, split))
    return tagged


def group_progressions(tagged_samples):
    groups = {}
    for s, split in tagged_samples:
        m = PROGRESSION_RE.match(os.path.basename(s.image))
        if not m:
            continue
        groups.setdefault(m.group(1), []).append((int(m.group(2)), s, split))
    return {p: [(s, split) for _, s, split in sorted(items)]
            for p, items in groups.items() if len(items) > 1}


def demo(args):
    device = get_device()
    print(f"Device: {device}")

    iqa_model = script_utils.load_model(args, device)

    tagged_samples = load_tagged_samples(args.data_root)
    groups = group_progressions(tagged_samples)
    mixed_prefixes = [p for p, g in groups.items() if len({split for _, split in g}) > 1]
    print(f"Found {len(groups)} KADID progressions ({len(mixed_prefixes)} span both train and test splits)")

    rng = random.Random(args.seed)
    # Prefer progressions that actually span both splits; fall back to any progression.
    chosen = rng.sample(mixed_prefixes, min(args.num_sets, len(mixed_prefixes)))
    remaining = [p for p in groups if p not in chosen]
    chosen += rng.sample(remaining, min(args.num_sets - len(chosen), len(remaining)))

    fig = plt.figure(figsize=(11, 4 * len(chosen)))
    outer = gridspec.GridSpec(len(chosen), 2, figure=fig, width_ratios=[2.2, 1], hspace=0.6)

    for row, prefix in enumerate(chosen):
        group = groups[prefix]
        samples = [s for s, _ in group]
        splits = [split for _, split in group]
        images = [Image.open(os.path.join(args.data_root, s.image)).convert("RGB") for s in samples]
        _, pred_scores = iqa_model.predict(images)
        gt_scores = [s.gt_score_norm for s in samples]
        levels = list(range(1, len(group) + 1))

        for name, split, pred, gt in zip((os.path.basename(s.image) for s in samples), splits, pred_scores, gt_scores):
            print(f"{name:<20} ({split:<5}) pred={pred:.2f}  gt={gt:.2f}")

        img_gs = outer[row, 0].subgridspec(1, len(group), wspace=0.08)
        for j, img in enumerate(images):
            ax_img = fig.add_subplot(img_gs[0, j])
            ax_img.imshow(img)
            ax_img.axis("off")
            label = f"L{levels[j]} ({splits[j]})"
            ax_img.set_title(f"{prefix}\n{label}" if j == 0 else label, fontsize=8)

        ax_plot = fig.add_subplot(outer[row, 1])
        ax_plot.plot(levels, pred_scores, marker="o", color=DEMO_PRED_COLOR, label="Pred")
        ax_plot.plot(levels, gt_scores, marker="o", color=DEMO_GT_COLOR, label="GT")
        ax_plot.set_xlabel("Distortion level")
        ax_plot.set_ylabel("Score")
        ax_plot.set_xticks(levels)
        ax_plot.legend(fontsize=8)
        ax_plot.grid(alpha=0.25)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    script_utils.add_model_args(parser)
    parser.add_argument("--data-root", default=DATA_DEQA_SCORE_DIR_DEFAULT,
                         help="Path to the Data-DeQA-Score directory (doubles as the image root)")
    parser.add_argument("--num-sets", type=int, default=3, help="Number of distortion progressions to show")
    parser.add_argument("--seed", type=int, default=DEMO_SEED_DEFAULT)
    parser.add_argument("--out", default=analysis_result_path("demo_kadid_progression.png"))
    args = parser.parse_args()
    demo(args)
