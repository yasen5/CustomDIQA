import argparse
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, ".")
from src.constants import (
    DATASET_SELECT_ARG_SPECS,
    DEMO_GT_COLOR,
    DEMO_LEVELS,
    DEMO_NUM_SAMPLES_DEFAULT,
    DEMO_OUT_DEFAULT,
    DEMO_PRED_COLOR,
    DEMO_SEED_DEFAULT,
    TRAIN_SAMPLE_SEED_DEFAULT,
    resolve_dataset_paths,
)
from src.datasets.gen_soft_label import load_soft_label_samples
from src.model import build_model, load_checkpoint, load_model_type
from src.trainer import SimpleImageProcessor, get_device
from src.utils import expand2square

SCORE_WEIGHTS = np.array([5, 4, 3, 2, 1], dtype=np.float32)


@torch.inference_mode()
def run_model(model, processor, pil_images, device):
    tensors = []
    for img in pil_images:
        img = expand2square(img, tuple(int(x * 255) for x in processor.image_mean))
        t = processor.preprocess(img, return_tensors="pt")["pixel_values"][0]
        tensors.append(t)
    batch = torch.stack(tensors).to(device=device, dtype=torch.float32)
    probs = F.softmax(model(batch), dim=-1).cpu().numpy()
    scores = (probs * SCORE_WEIGHTS).sum(axis=-1)
    return probs, scores


def make_plot(titles, images, gt_scores, gt_probs_list, pred_probs, pred_scores, out_path, model_type):
    n = len(titles)
    x = np.arange(len(DEMO_LEVELS))

    fig = plt.figure(figsize=(4.8 * n, 7.2))
    gs = gridspec.GridSpec(2, n, height_ratios=[1, 1.5], hspace=0.38, wspace=0.38)

    for i in range(n):
        gt_p = np.array(gt_probs_list[i])

        ax_img = fig.add_subplot(gs[0, i])
        ax_img.imshow(images[i])
        ax_img.axis("off")
        ax_img.set_title(titles[i], fontsize=9, pad=4)

        ax_dist = fig.add_subplot(gs[1, i])
        ax_dist.fill_between(x, pred_probs[i], color=DEMO_PRED_COLOR, alpha=0.24)
        ax_dist.plot(x, pred_probs[i], color=DEMO_PRED_COLOR, marker="o", linewidth=2.0,
                     label=f"Pred ({pred_scores[i]:.2f})")
        ax_dist.fill_between(x, gt_p, color=DEMO_GT_COLOR, alpha=0.24)
        ax_dist.plot(x, gt_p, color=DEMO_GT_COLOR, marker="o", linewidth=2.0,
                     label=f"GT ({gt_scores[i]:.2f})")
        ax_dist.set_xticks(x)
        ax_dist.set_xticklabels(DEMO_LEVELS, fontsize=8)
        ax_dist.set_xlim(x[0], x[-1])
        ax_dist.set_ylim(0, 1)
        ax_dist.set_ylabel("Probability", fontsize=9)
        ax_dist.grid(axis="y", alpha=0.25, linewidth=0.8)
        ax_dist.tick_params(axis="y", labelsize=8)
        ax_dist.legend(fontsize=7.5, loc="upper right")
        ax_dist.set_title("Label Distributions", fontsize=9, pad=4)

    fig.suptitle(f"{model_type.upper()} — Predicted vs Ground-Truth", fontsize=13, y=1.01)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved to {out_path}")


def demo(args):
    device = get_device()
    print(f"Device: {device}")

    if os.path.isdir(args.model_path):
        model_type = args.model_type or load_model_type(args.model_path)
        weights_path = os.path.join(args.model_path, "weights.pt")
    else:
        if args.model_type is None:
            raise ValueError("--model-type is required when --model-path is a weights file, not a checkpoint dir")
        model_type = args.model_type
        weights_path = args.model_path

    model, model_constants = build_model(model_type)
    model = model.to(device=device, dtype=torch.float32)
    model_state, _ = load_checkpoint(weights_path, map_location="cpu")
    model.load_state_dict(model_state)
    model.eval()
    print(f"Loaded {model_type} model from {weights_path}")

    processor = SimpleImageProcessor(model_constants.img_size)

    # Pool samples across every selected dataset's --split. With --split train and
    # matching --sample-size/--sample-seed, this replicates the exact pool the trainer used.
    all_data = []
    for path in args.data_path:
        all_data += [s for s in load_soft_label_samples(path) if s.level_probs is not None]
    if args.sample_size is not None:
        pool_indices = random.Random(args.sample_seed).sample(range(len(all_data)), args.sample_size)
    else:
        pool_indices = list(range(len(all_data)))
    pool = [all_data[i] for i in pool_indices]
    print(f"Pool: {len(pool)} samples from {len(args.data_path)} dataset(s) ({args.split} split)")

    shuffled = pool[:]
    random.Random(args.seed).shuffle(shuffled)

    titles, images, gt_scores, gt_probs_list = [], [], [], []
    for s in shuffled:
        if len(titles) >= args.num_samples:
            break
        try:
            img = Image.open(os.path.join(args.image_folder, s.image)).convert("RGB")
        except (FileNotFoundError, OSError) as ex:
            print(f"WARNING: skipping {s.image}: {ex}")
            continue
        titles.append(os.path.basename(s.image))
        images.append(img)
        gt_scores.append(s.gt_score_norm)
        gt_probs_list.append(s.level_probs)

    if len(titles) < args.num_samples:
        print(f"WARNING: only found {len(titles)}/{args.num_samples} requested images "
              f"(some dataset images may not be downloaded locally)")

    pred_probs, pred_scores = run_model(model, processor, images, device)

    for title, gt_score, gt_p, score, probs in zip(titles, gt_scores, gt_probs_list, pred_scores, pred_probs):
        dist = "  ".join(f"{l}={p:.2f}" for l, p in zip(DEMO_LEVELS, probs))
        print(f"{title:<30}  pred={score:.2f}  gt={gt_score:.2f}  [{dist}]")

    if args.out:
        make_plot(titles, images, gt_scores, gt_probs_list, pred_probs, pred_scores, args.out, model_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-type", choices=["vit", "cnn"], default=None,
                        help="Overrides the checkpoint's recorded model type; required if --model-path is a weights file")
    for arg_spec in DATASET_SELECT_ARG_SPECS:
        parser.add_argument(*arg_spec["flags"], **arg_spec["kwargs"])
    parser.add_argument("--split", choices=["train", "test"], default="test",
                        help="Pool samples from each dataset's train or test split")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Subsample the pool to this many samples before drawing --num-samples; "
                             "with --split train, must match --sample-size used during training to "
                             "replicate its exact pool")
    parser.add_argument("--sample-seed", type=int, default=TRAIN_SAMPLE_SEED_DEFAULT,
                        help="Must match --sample-seed used during training if replicating its pool")
    parser.add_argument("--num-samples", type=int, default=DEMO_NUM_SAMPLES_DEFAULT)
    parser.add_argument("--seed", type=int, default=DEMO_SEED_DEFAULT)
    parser.add_argument("--out", default=DEMO_OUT_DEFAULT)
    args = parser.parse_args()
    _, args.data_path = resolve_dataset_paths(args.datasets, args.exclude_datasets, args.data_root, args.split)
    args.image_folder = args.data_root
    demo(args)
