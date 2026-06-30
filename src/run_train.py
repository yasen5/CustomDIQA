import argparse
import sys

sys.path.insert(0, ".")
from src.trainer import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--save-path", default="model.pt")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--osc-factor", type=float, default=0.5)
    parser.add_argument("--osc-threshold", type=float, default=0.3)
    parser.add_argument("--osc-cooldown", type=int, default=50)
    parser.add_argument("--osc-min-lr", type=float, default=1e-6)
    parser.add_argument("--backbone-lr-scale", type=float, default=1.0)
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Use a fixed random subset of this many samples from the dataset")
    parser.add_argument("--sample-seed", type=int, default=42,
                        help="Seed for subset selection and mini-batch sampling")
    args = parser.parse_args()
    train(args)
