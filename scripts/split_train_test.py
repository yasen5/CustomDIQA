import argparse
import json
import os
import random
import sys

sys.path.insert(0, ".")
from src.constants import DATA_DEQA_SCORE_DIR_DEFAULT, IQA_DATASET_ARCHIVES, ORIGINAL_TEST_ONLY_META_FILENAMES
from src.datasets.gen_soft_label import load_soft_label_samples

TRAIN_FRAC_DEFAULT = 0.9
SPLIT_SEED_DEFAULT = 42


def split_dataset(key, data_root, train_frac, seed, force):
    _, dataset_dir = IQA_DATASET_ARCHIVES[key]
    metas_dir = os.path.join(data_root, dataset_dir, "metas")
    src_path = os.path.join(metas_dir, ORIGINAL_TEST_ONLY_META_FILENAMES[key])
    train_path = os.path.join(metas_dir, "train.json")
    test_path = os.path.join(metas_dir, "test.json")

    if os.path.isfile(train_path) and os.path.isfile(test_path) and not force:
        print(f"[{key}] {train_path} and {test_path} already exist, skipping (use --force to overwrite)")
        return

    samples = [s.to_json_dict() for s in load_soft_label_samples(src_path)]
    random.Random(seed).shuffle(samples)

    n_train = round(len(samples) * train_frac)
    train_samples, test_samples = samples[:n_train], samples[n_train:]

    with open(train_path, "w") as f:
        json.dump(train_samples, f, indent=4)
    with open(test_path, "w") as f:
        json.dump(test_samples, f, indent=4)
    print(f"[{key}] split {len(samples)} samples from {src_path} -> "
          f"{len(train_samples)} train / {len(test_samples)} test")


def main():
    parser = argparse.ArgumentParser(
        description="Partition datasets that ship only a single test-benchmark meta (no train "
                     "split) into a --train-frac/remainder train/test split, saved as "
                     "metas/train.json + metas/test.json")
    parser.add_argument("--datasets", nargs="+", choices=tuple(ORIGINAL_TEST_ONLY_META_FILENAMES),
                        default=tuple(ORIGINAL_TEST_ONLY_META_FILENAMES),
                        help="Datasets to split (default: all datasets with no existing train split)")
    parser.add_argument("--data-root", default=DATA_DEQA_SCORE_DIR_DEFAULT)
    parser.add_argument("--train-frac", type=float, default=TRAIN_FRAC_DEFAULT)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED_DEFAULT)
    parser.add_argument("--force", action="store_true", help="Overwrite existing metas/train.json + test.json")
    args = parser.parse_args()

    for key in args.datasets:
        split_dataset(key, args.data_root, args.train_frac, args.seed, args.force)


if __name__ == "__main__":
    main()
