import argparse
import multiprocessing as mp
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, ".")
from src.constants import DATASET_SELECT_ARG_SPECS, MODEL_TYPES, TRAIN_MODEL_TYPE_DEFAULT, resolve_dataset_paths
from src.datasets.gen_soft_label import load_soft_label_samples
from src.datasets.preprocessed import preprocessed_cache_path, save_preprocessed_cache
from src.model import MODEL_REGISTRY
from src.trainer import SimpleImageProcessor
from src.utils import expand2square


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
}

_WORKER_DATA_ROOT = None
_WORKER_PROCESSOR = None
_WORKER_IMAGE_ASPECT_RATIO = None
_WORKER_DTYPE = None


def _init_worker(data_root, img_size, image_aspect_ratio, dtype_name):
    global _WORKER_DATA_ROOT, _WORKER_PROCESSOR, _WORKER_IMAGE_ASPECT_RATIO, _WORKER_DTYPE
    torch.set_num_threads(1)
    _WORKER_DATA_ROOT = data_root
    _WORKER_PROCESSOR = SimpleImageProcessor(img_size, augment=False)
    _WORKER_IMAGE_ASPECT_RATIO = image_aspect_ratio
    _WORKER_DTYPE = DTYPES[dtype_name]


def _preprocess_one(sample):
    image_path = os.path.join(_WORKER_DATA_ROOT, sample.image)
    image = Image.open(image_path).convert("RGB")
    if _WORKER_IMAGE_ASPECT_RATIO == "pad":
        image = expand2square(image, tuple(int(x * 255) for x in _WORKER_PROCESSOR.image_mean))
    tensor = _WORKER_PROCESSOR.preprocess(image, return_tensors="pt")["pixel_values"][0].to(dtype=_WORKER_DTYPE)
    return tensor.numpy()


def preprocess_samples(samples, data_root, processor, image_aspect_ratio, dtype_name, num_workers):
    dtype = DTYPES[dtype_name]
    if not samples:
        return torch.empty(0, 3, processor.crop_size["height"], processor.crop_size["width"], dtype=dtype)

    arrays = []
    img_size = processor.crop_size["height"]
    if num_workers <= 1:
        _init_worker(data_root, img_size, image_aspect_ratio, dtype_name)
        iterator = map(_preprocess_one, samples)
        for idx, array in enumerate(iterator, start=1):
            arrays.append(array)
            if idx % 500 == 0:
                print(f"  processed {idx}/{len(samples)}", flush=True)
        return torch.from_numpy(np.stack(arrays))

    with mp.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(data_root, img_size, image_aspect_ratio, dtype_name),
    ) as pool:
        for idx, array in enumerate(pool.imap(_preprocess_one, samples, chunksize=16), start=1):
            arrays.append(array)
            if idx % 500 == 0:
                print(f"  processed {idx}/{len(samples)}", flush=True)
    return torch.from_numpy(np.stack(arrays))


def preprocess_dataset(meta_path, data_root, processor, image_aspect_ratio, dtype_name, num_workers, force):
    samples = load_soft_label_samples(meta_path)
    n_before = len(samples)
    samples = [sample for sample in samples if sample.level_probs is not None]
    if n_before != len(samples):
        print(f"  skipped {n_before - len(samples)}/{n_before} samples missing level_probs")

    cache_path = preprocessed_cache_path(meta_path, processor, image_aspect_ratio)
    if os.path.isfile(cache_path) and not force:
        print(f"Exists, skipping: {cache_path}")
        return

    print(f"Preprocessing {len(samples)} samples from {meta_path} with {num_workers} worker(s)")
    images = preprocess_samples(samples, data_root, processor, image_aspect_ratio, dtype_name, num_workers)
    save_preprocessed_cache(cache_path, samples, images, processor, image_aspect_ratio)
    size_mb = os.path.getsize(cache_path) / (1024 * 1024)
    print(f"Saved {cache_path} ({size_mb:.1f} MiB)")


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess selected dataset images into reusable tensor caches")
    for arg_spec in DATASET_SELECT_ARG_SPECS:
        parser.add_argument(*arg_spec["flags"], **arg_spec["kwargs"])
    parser.add_argument("--split", choices=("train", "test", "all"), default="train")
    parser.add_argument("--model-type", choices=MODEL_TYPES, default=TRAIN_MODEL_TYPE_DEFAULT)
    parser.add_argument("--image-aspect-ratio", choices=("pad", "none"), default="pad")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="float16",
                        help="Storage dtype for preprocessed tensors")
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1),
                        help="Parallel image decode/resize workers")
    parser.add_argument("--force", action="store_true", help="Overwrite existing preprocessed caches")
    return parser.parse_args()


def main():
    args = parse_args()
    _, model_constants = MODEL_REGISTRY[args.model_type]
    processor = SimpleImageProcessor(model_constants.img_size, augment=False)
    splits = ("train", "test") if args.split == "all" else (args.split,)

    for split in splits:
        print(f"Split: {split}")
        _, data_paths = resolve_dataset_paths(args.datasets, args.exclude_datasets, args.data_root, split)
        for meta_path in data_paths:
            preprocess_dataset(
                meta_path,
                args.data_root,
                processor,
                args.image_aspect_ratio,
                args.dtype,
                args.num_workers,
                args.force,
            )


if __name__ == "__main__":
    main()
