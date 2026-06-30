import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, ".")
from src.constants import (
    TRAIN_BACKBONE_LR_SCALE_DEFAULT,
    TRAIN_BATCH_SIZE_DEFAULT,
    TRAIN_GRAD_ACCUM_DEFAULT,
    TRAIN_LOG_EVERY_DEFAULT,
    TRAIN_LR_DEFAULT,
    TRAIN_OSC_COOLDOWN_DEFAULT,
    TRAIN_OSC_FACTOR_DEFAULT,
    TRAIN_OSC_MIN_LR_DEFAULT,
    TRAIN_OSC_THRESHOLD_DEFAULT,
    TRAIN_SAMPLE_SEED_DEFAULT,
    TRAIN_STEPS_DEFAULT,
    TRAIN_WARMUP_STEPS_DEFAULT,
)
from src.trainer import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE_DEFAULT)
    parser.add_argument("--grad-accum", type=int, default=TRAIN_GRAD_ACCUM_DEFAULT)
    parser.add_argument("--lr", type=float, default=TRAIN_LR_DEFAULT)
    parser.add_argument("--warmup-steps", type=int, default=TRAIN_WARMUP_STEPS_DEFAULT)
    parser.add_argument("--log-every", type=int, default=TRAIN_LOG_EVERY_DEFAULT)
    parser.add_argument("--osc-factor", type=float, default=TRAIN_OSC_FACTOR_DEFAULT)
    parser.add_argument("--osc-threshold", type=float, default=TRAIN_OSC_THRESHOLD_DEFAULT)
    parser.add_argument("--osc-cooldown", type=int, default=TRAIN_OSC_COOLDOWN_DEFAULT)
    parser.add_argument("--osc-min-lr", type=float, default=TRAIN_OSC_MIN_LR_DEFAULT)
    parser.add_argument("--backbone-lr-scale", type=float, default=TRAIN_BACKBONE_LR_SCALE_DEFAULT)
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Use a fixed random subset of this many samples from the dataset")
    parser.add_argument("--sample-seed", type=int, default=TRAIN_SAMPLE_SEED_DEFAULT,
                        help="Seed for subset selection and mini-batch sampling")
    args = parser.parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    args.save_path = os.path.join(args.checkpoint_dir, f"run_{timestamp}_steps{args.steps}")
    train(args)
