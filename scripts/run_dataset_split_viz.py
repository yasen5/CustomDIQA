import argparse
import json
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, ".")
from src.constants import DATA_DEQA_SCORE_DIR_DEFAULT
from src.constants import analysis_result_path

SAMPLE_COUNT_DEFAULT = 5
INPUT_JSON_DEFAULT = analysis_result_path("dataset_split_failures.json")
OUT_DEFAULT = analysis_result_path("dataset_split_samples.png")


def make_grid(failures, successes, image_folder, out_path):
    rows = [("Failures", failures), ("Successes", successes)]
    n_cols = max(len(failures), len(successes))
    fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 6.5), squeeze=False)

    for row_idx, (label, records) in enumerate(rows):
        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if col_idx >= len(records):
                continue
            r = records[col_idx]
            img = Image.open(os.path.join(image_folder, r["image"])).convert("RGB")
            ax.imshow(img)
            ax.set_title(f"{r['dataset']}\npred={r['pred']:.2f}  gt={r['gt']:.2f}", fontsize=8)
        axes[row_idx][0].set_ylabel(label, fontsize=12)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved to {out_path}")


def visualize(args):
    with open(args.input_json) as f:
        data = json.load(f)

    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2 ** 32)
    print(f"Using seed {seed} (pass --seed {seed} to reproduce this sample)")

    failures = random.Random(seed).sample(data["failures"], min(args.num_samples, len(data["failures"])))
    successes = random.Random(seed).sample(data["successes"], min(args.num_samples, len(data["successes"])))
    print(f"Sampled {len(failures)} failures and {len(successes)} successes from {args.input_json}")

    make_grid(failures, successes, args.data_root, args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Samples images from each bucket of a run_dataset_split.py output "
                     "(failures/successes) and shows them side-by-side in a grid image")
    parser.add_argument("--input-json", default=INPUT_JSON_DEFAULT,
                        help="Path to the run_dataset_split.py output JSON")
    parser.add_argument("--data-root", default=DATA_DEQA_SCORE_DIR_DEFAULT,
                        help="Path to the Data-DeQA-Score directory (image root)")
    parser.add_argument("--num-samples", type=int, default=SAMPLE_COUNT_DEFAULT,
                        help="Number of images to sample from each bucket")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for sampling which images to show (random each run if omitted)")
    parser.add_argument("--out", default=OUT_DEFAULT)
    args = parser.parse_args()
    visualize(args)
