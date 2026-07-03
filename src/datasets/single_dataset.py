import os
import random
from dataclasses import dataclass
from typing import List

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.utils import expand2square, rank0_print
from .gen_soft_label import load_soft_label_samples
from .preprocessed import load_preprocessed_cache, preprocessed_cache_path


@dataclass
class SingleSampleItem:
    image: torch.Tensor
    level_probs: List[float] # ground truth probability of each score [excellent, good, fair, poor, bad]


class SingleDataset(Dataset):
    """Dataset for single-image quality scoring."""

    def __init__(self, data_paths, data_weights, data_args):
        super().__init__()
        list_data_dict = []
        preprocessed_images = []
        for data_path, data_weight in zip(data_paths, data_weights):
            data_dict = load_soft_label_samples(data_path)
            n_before = len(data_dict)
            data_dict = [s for s in data_dict if s.level_probs is not None]
            n_skipped = n_before - len(data_dict)
            if n_skipped:
                rank0_print(f"WARNING: skipped {n_skipped}/{n_before} samples missing 'level_probs' in {data_path}")
            list_data_dict += data_dict * data_weight
            cache_images = self._load_preprocessed(data_path, data_dict, data_args)
            for _ in range(data_weight):
                if cache_images is None:
                    preprocessed_images.extend([None] * len(data_dict))
                else:
                    preprocessed_images.extend(cache_images)

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.preprocessed_images = preprocessed_images

    @staticmethod
    def _load_preprocessed(data_path, data_dict, data_args):
        processor = data_args.image_processor
        if getattr(processor, "augment", False):
            return None
        cache_path = preprocessed_cache_path(data_path, processor, data_args.image_aspect_ratio)
        try:
            images = load_preprocessed_cache(cache_path, data_dict, processor, data_args.image_aspect_ratio)
        except Exception as ex:
            rank0_print(f"WARNING: could not load preprocessed cache {cache_path}: {ex}")
            return None
        if images is not None:
            rank0_print(f"Using preprocessed cache: {cache_path}")
        return images

    def __len__(self):
        return len(self.list_data_dict)

    def next_rand(self):
        return random.randint(0, len(self) - 1)

    def __getitem__(self, i) -> SingleSampleItem:
        while True:
            try:
                cached_image = self.preprocessed_images[i]
                if cached_image is not None:
                    sample = self.list_data_dict[i]
                    return SingleSampleItem(
                        image=cached_image,
                        level_probs=sample.level_probs,
                    )

                sample = self.list_data_dict[i]
                image_folder = self.data_args.image_folder
                processor = self.data_args.image_processor

                image_file = sample.image
                if image_file is not None:
                    image_path = os.path.join(image_folder, image_file)
                    try:
                        image = Image.open(image_path).convert("RGB")
                    except Exception as ex:
                        print(ex)
                        i = self.next_rand()
                        continue

                    if self.data_args.image_aspect_ratio == "pad":
                        image = expand2square(
                            image, tuple(int(x * 255) for x in processor.image_mean)
                        )
                    image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
                else:
                    crop_size = processor.crop_size
                    image = torch.zeros(3, crop_size["height"], crop_size["width"])

                return SingleSampleItem(
                    image=image,
                    level_probs=sample.level_probs,
                )
            except Exception as ex:
                print(ex)
                i = self.next_rand()
                continue
