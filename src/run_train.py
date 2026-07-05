import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, ".")
from src.constants import (
    DATASET_SELECT_ARG_SPECS,
    MODEL_TYPES,
    PRETRAINED_TYPES,
    TRAIN_BACKBONE_LR_SCALE_DEFAULT,
    TRAIN_BACKBONE_WARMUP_STEPS_DEFAULT,
    TRAIN_BATCH_SIZE_DEFAULT,
    TRAIN_FREEZE_BACKBONE_STEPS_DEFAULT,
    TRAIN_GRAD_ACCUM_DEFAULT,
    TRAIN_HEAD_WEIGHT_DECAY_DEFAULT,
    TRAIN_LOG_EVERY_DEFAULT,
    TRAIN_LR_DEFAULT,
    TRAIN_MODEL_TYPE_DEFAULT,
    TRAIN_OSC_COOLDOWN_DEFAULT,
    TRAIN_OSC_FACTOR_DEFAULT,
    TRAIN_OSC_MIN_LR_DEFAULT,
    TRAIN_OSC_THRESHOLD_DEFAULT,
    TRAIN_PRETRAINED_BACKBONE_LR_SCALE_DEFAULT,
    TRAIN_PRETRAINED_DEFAULT,
    TRAIN_SAMPLE_SEED_DEFAULT,
    TRAIN_STEPS_DEFAULT,
    TRAIN_VIT_WARMUP_STEPS_DEFAULT,
    TRAIN_WARMUP_STEPS_DEFAULT,
    TRAIN_WEIGHT_DECAY_DEFAULT,
    resolve_dataset_paths,
)
from src.trainer import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for arg_spec in DATASET_SELECT_ARG_SPECS:
        parser.add_argument(*arg_spec["flags"], **arg_spec["kwargs"])
    parser.add_argument("--data-weights", type=int, nargs="+", default=None,
                        help="Integer replication weight per selected dataset (default: 1 each)")
    parser.add_argument("--model-type", choices=MODEL_TYPES, default=TRAIN_MODEL_TYPE_DEFAULT,
                        help="Backbone architecture to train")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--checkpoint-path", default=None,
                        help="Path to a checkpoint dir or weights.pt file to resume training from")
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE_DEFAULT)
    parser.add_argument("--grad-accum", type=int, default=TRAIN_GRAD_ACCUM_DEFAULT)
    parser.add_argument("--lr", type=float, default=TRAIN_LR_DEFAULT)
    parser.add_argument("--warmup-steps", type=int, default=None,
                        help=f"Defaults to {TRAIN_VIT_WARMUP_STEPS_DEFAULT} for vit, "
                             f"{TRAIN_WARMUP_STEPS_DEFAULT} for cnn/hybrid, unless set explicitly")
    parser.add_argument("--log-every", type=int, default=TRAIN_LOG_EVERY_DEFAULT)
    parser.add_argument("--osc-factor", type=float, default=TRAIN_OSC_FACTOR_DEFAULT)
    parser.add_argument("--osc-threshold", type=float, default=TRAIN_OSC_THRESHOLD_DEFAULT)
    parser.add_argument("--osc-cooldown", type=int, default=TRAIN_OSC_COOLDOWN_DEFAULT)
    parser.add_argument("--osc-min-lr", type=float, default=TRAIN_OSC_MIN_LR_DEFAULT)
    parser.add_argument("--backbone-lr-scale", type=float, default=None,
                        help=f"Defaults to {TRAIN_PRETRAINED_BACKBONE_LR_SCALE_DEFAULT} when --pretrained is set "
                             f"(a freshly-unfrozen pretrained backbone gets large, near-uniform Adam updates "
                             f"in its first few post-unfreeze steps regardless of gradient size — full head LR "
                             f"is enough to wreck the transferred features), else {TRAIN_BACKBONE_LR_SCALE_DEFAULT} "
                             f"(no reason to scale down an already-random backbone).")
    parser.add_argument("--weight-decay", type=float, default=TRAIN_WEIGHT_DECAY_DEFAULT,
                        help="Weight decay for backbone params")
    parser.add_argument("--head-weight-decay", type=float, default=TRAIN_HEAD_WEIGHT_DECAY_DEFAULT)
    parser.add_argument("--freeze-backbone-steps", type=int, default=None,
                        help=f"Freeze all non-head params for this many steps, then unfreeze. "
                             f"Defaults to {TRAIN_FREEZE_BACKBONE_STEPS_DEFAULT} when --pretrained is set "
                             f"(protects the transferred weights from a randomly-initialized head's early "
                             f"gradients), else 0 (no benefit to freezing an already-random backbone).")
    parser.add_argument("--backbone-warmup-steps", type=int, default=TRAIN_BACKBONE_WARMUP_STEPS_DEFAULT,
                        help="LR re-warmup steps for the backbone group after it unfreezes")
    parser.add_argument("--pretrained", choices=PRETRAINED_TYPES, default=TRAIN_PRETRAINED_DEFAULT,
                        help="Initialize the backbone from an external pretrained model: "
                             "'dinov2' (vit only, first N transformer blocks + patch-embed), "
                             "'imagenet' (cnn only, full EfficientNet-B0 backbone transfer), "
                             "'topiq_nr' (hybrid only, full TOPIQ-NR CFANet backbone transfer), or "
                             "'musiq_koniq' (musiq only, full MUSIQ koniq10k tokenizer+encoder transfer). "
                             "One-time network fetch on first use, cached afterward.")
    parser.add_argument("--augment", action="store_true",
                        help="Apply mild random-crop + horizontal-flip augmentation (recommended for vit)")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Use a fixed random subset of this many samples from the dataset")
    parser.add_argument("--sample-seed", type=int, default=TRAIN_SAMPLE_SEED_DEFAULT,
                        help="Seed for subset selection and mini-batch sampling")
    args = parser.parse_args()

    _, args.data_path = resolve_dataset_paths(args.datasets, args.exclude_datasets, args.data_root, "train")
    args.image_folder = args.data_root

    if args.data_weights is None:
        args.data_weights = [1] * len(args.data_path)
    elif len(args.data_weights) != len(args.data_path):
        raise ValueError(
            f"--data-weights has {len(args.data_weights)} entries, expected one per "
            f"--data-path ({len(args.data_path)})"
        )
    if args.warmup_steps is None:
        args.warmup_steps = TRAIN_VIT_WARMUP_STEPS_DEFAULT if args.model_type == "vit" else TRAIN_WARMUP_STEPS_DEFAULT
    if args.freeze_backbone_steps is None:
        args.freeze_backbone_steps = TRAIN_FREEZE_BACKBONE_STEPS_DEFAULT if args.pretrained else 0
    if args.backbone_lr_scale is None:
        args.backbone_lr_scale = TRAIN_PRETRAINED_BACKBONE_LR_SCALE_DEFAULT if args.pretrained else TRAIN_BACKBONE_LR_SCALE_DEFAULT
    if args.warmup_steps > args.freeze_backbone_steps > 0:
        print(f"WARNING: --warmup-steps ({args.warmup_steps}) > --freeze-backbone-steps "
              f"({args.freeze_backbone_steps}) — the head's own warmup won't finish before backbone unfreeze.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    args.save_path = os.path.join(args.checkpoint_dir, f"run_{args.model_type}_{timestamp}_steps{args.steps}")
    train(args)
