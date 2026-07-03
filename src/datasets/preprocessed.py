import os
from typing import List

import torch

from .gen_soft_label import SoftLabelSample

CACHE_VERSION = 1
PROCESSED_DIR_NAME = "processed"


def processor_size(processor):
    crop_size = processor.crop_size
    return int(crop_size["height"]), int(crop_size["width"])


def preprocessed_cache_path(meta_path: str, processor, image_aspect_ratio: str) -> str:
    height, width = processor_size(processor)
    metas_dir = os.path.dirname(meta_path)
    stem = os.path.splitext(os.path.basename(meta_path))[0]
    filename = f"{stem}_{height}x{width}_{image_aspect_ratio}_v{CACHE_VERSION}.pt"
    return os.path.join(metas_dir, PROCESSED_DIR_NAME, filename)


def _sample_cache_keys(samples: List[SoftLabelSample]):
    return [sample.image for sample in samples]


def save_preprocessed_cache(path: str, samples: List[SoftLabelSample], images: torch.Tensor, processor, image_aspect_ratio: str):
    height, width = processor_size(processor)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "version": CACHE_VERSION,
            "height": height,
            "width": width,
            "image_aspect_ratio": image_aspect_ratio,
            "sample_images": _sample_cache_keys(samples),
            "images": images,
        },
        path,
    )


def load_preprocessed_cache(path: str, samples: List[SoftLabelSample], processor, image_aspect_ratio: str):
    if not os.path.isfile(path):
        return None

    payload = torch.load(path, map_location="cpu")
    height, width = processor_size(processor)
    if payload.get("version") != CACHE_VERSION:
        return None
    if payload.get("height") != height or payload.get("width") != width:
        return None
    if payload.get("image_aspect_ratio") != image_aspect_ratio:
        return None
    if payload.get("sample_images") != _sample_cache_keys(samples):
        return None

    images = payload.get("images")
    if not isinstance(images, torch.Tensor) or len(images) != len(samples):
        return None
    return images
