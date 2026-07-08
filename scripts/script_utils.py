import os
import types

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from src.datasets.single_dataset import SingleDataset
from src.model import build_model, load_checkpoint, load_model_type
from src.model.qalign import QAlignMiniIQA
from src.trainer import SimpleImageProcessor
from src.utils import expand2square

MODEL_CHOICES = ("vit", "cnn", "hybrid", "topiq_nr", "musiq", "qalign_mini")
# The subset of MODEL_CHOICES that load_model() can build with no --model-path at all — i.e. every
# pretrained-only IQA model this repo ships an off-the-shelf wrapper for (as opposed to vit/cnn/hybrid,
# and musiq-with-a-checkpoint, which score with a locally trained head). Used by
# scripts/run_eval_pretrained.py to eval "every pretrained model" without hardcoding the list twice.
PRETRAINED_MODEL_TYPES = ("topiq_nr", "musiq", "qalign_mini")
SCORE_WEIGHTS = np.array([5, 4, 3, 2, 1], dtype=np.float32)

# Matches CustomModel.predict()'s hardcoded expand2square call and scripts/preprocess_datasets.py's
# default --image-aspect-ratio, since that's the only aspect-ratio handling any of these scripts apply.
IMAGE_ASPECT_RATIO = "pad"

TOPIQ_NR_SCORE_RANGE = (0.0, 1.0)  # pyiqa's documented (approximate) score_range for topiq_nr: "~0, ~1"
GT_SCORE_RANGE = (1.0, 5.0)        # this repo's MOS scale (see gen_soft_label.py mos_norm, SCORE_WEIGHTS above)

# topiq_nr's CFANet has no built-in resize for this checkpoint (test_img_size=None in pyiqa's
# config), so its cross-attention token count grows with input pixel count. Raw phone photos
# (e.g. SPAQ, ~3000-5500px) OOM an 11GB GPU; this cap only kicks in above that size, so every
# other dataset here (already downscaled) still scores at true native resolution.
TOPIQ_NR_MAX_SIDE = 2048

# MUSIQ's native-resolution scale keeps every patch unpadded/untruncated (max_seq_len_from_original_res
# = -1, see src/model/musiq/constants.py), so — same OOM concern as topiq_nr above — raw phone photos
# would otherwise blow the sequence length up unboundedly.
MUSIQ_MAX_SIDE = 2048


def _cap_image_size(img, max_side):
    w, h = img.size
    longest = max(w, h)
    if longest <= max_side:
        return img
    scale = max_side / longest
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.BICUBIC)


def _rescale(x, src_range, dst_range):
    src_lo, src_hi = src_range
    dst_lo, dst_hi = dst_range
    return dst_lo + (x - src_lo) * (dst_hi - dst_lo) / (src_hi - src_lo)


def add_model_args(parser):
    parser.add_argument("--model-path", default=None,
                         help="Checkpoint dir or weights file. Required for --model-type vit/cnn/hybrid "
                              "(unless --model-type is omitted and can be read from the checkpoint "
                              "dir's model_type.txt); ignored for topiq_nr and for musiq without a "
                              "model path, which use "
                              "pretrained weights instead of a local checkpoint.")
    parser.add_argument("--model-type", choices=MODEL_CHOICES, default=None,
                         help="'vit'/'cnn'/'hybrid' load a trained checkpoint from --model-path (overrides "
                              "the checkpoint's recorded type; required if --model-path is a weights "
                              "file rather than a checkpoint dir). 'topiq_nr' uses pyiqa's pretrained "
                              "TOPIQ-NR no-reference IQA metric instead of a local checkpoint. 'musiq' "
                              "uses this repo's manual MUSIQ reimplementation with pyiqa's pretrained "
                              "koniq10k weights when --model-path is omitted, or a local checkpoint "
                              "when --model-path is provided. 'qalign_mini' runs this repo's manual "
                              "Q-Align Mini wrapper over the already-downloaded local "
                              "q-future/Q-ReAlign-Mini-0.8B weights.")


def load_model(args, device):
    """Factory returning a model wrapper with a uniform .predict(pil_images) -> (probs_or_None, scores)
    interface and a .model_type label, regardless of whether the backend is a local checkpoint or a
    pyiqa metric. 'musiq' with no --model-path falls back to MusiqModel (pretrained backbone + a
    freshly-initialized, un-fine-tuned head — the "untuned" baseline); with --model-path it's a
    fine-tuned checkpoint like vit/cnn/hybrid, so it goes through CustomModel instead."""
    if args.model_type == "topiq_nr":
        return TopiqNRModel(device)
    if args.model_type == "musiq" and args.model_path is None:
        return MusiqModel(device)
    if args.model_type == "qalign_mini":
        return QAlignMiniIQA(device)
    return CustomModel(args, device)


def build_dataset(meta_path, image_folder, processor):
    """Wraps a single meta file in a SingleDataset (see src/datasets/single_dataset.py) so every
    script that scores a fixed-size-processor model (vit/cnn/hybrid, via CustomModel) gets the
    exact same image loading behavior training does — preprocessed-tensor cache lookup
    (src/datasets/preprocessed.py) when available, PIL decode/resize/pad otherwise, and the same
    level_probs-based sample filtering — instead of every script reimplementing it. Not usable
    for topiq_nr: it scores at native resolution, incompatible with SingleDataset's fixed
    crop_size pipeline (see predict_raw_items below)."""
    data_args = types.SimpleNamespace(
        image_folder=image_folder,
        image_processor=processor,
        image_aspect_ratio=IMAGE_ASPECT_RATIO,
    )
    return SingleDataset(data_paths=[meta_path], data_weights=[1], data_args=data_args)


def predict_dataset_items(iqa_model, dataset, indices, batch_size):
    """Runs `iqa_model` (a CustomModel) over `indices` into `dataset` (from build_dataset) in
    chunks of `batch_size`, yielding (index_chunk, scores) per batch. Callers zip index_chunk
    against dataset.list_data_dict (or whatever they tagged those indices with, e.g. a pooled
    multi-dataset key) to pull gt scores/metadata — this function only knows about batching and
    prediction, not what a caller wants to do with the result."""
    for i in range(0, len(indices), batch_size):
        chunk = indices[i:i + batch_size]
        batch = torch.stack([dataset[j].image for j in chunk])
        _, scores = iqa_model.predict_tensors(batch)
        yield chunk, scores.tolist()


def predict_raw_items(iqa_model, items, image_folder, get_image_path, batch_size):
    """Fallback sampling path for models without a fixed-size processor (currently only
    topiq_nr — see build_dataset). Loads each item's image with PIL and calls iqa_model.predict
    directly, dropping (with a warning) any item whose image fails to load. `items` can be any
    objects; `get_image_path(item)` extracts the image's path relative to `image_folder`. Yields
    (kept_items, scores) per batch."""
    n_total, n_skipped = len(items), 0
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        images, kept = [], []
        for item in chunk:
            image_path = get_image_path(item)
            try:
                images.append(Image.open(os.path.join(image_folder, image_path)).convert("RGB"))
            except (FileNotFoundError, OSError) as ex:
                print(f"WARNING: skipping {image_path}: {ex}")
                n_skipped += 1
                continue
            kept.append(item)
        if not images:
            continue
        _, scores = iqa_model.predict(images)
        yield kept, scores.tolist()
    if n_skipped:
        print(f"  ({n_skipped}/{n_total} images skipped — not found locally)")


class CustomModel:
    def __init__(self, args, device):
        if args.model_path is None:
            raise ValueError("--model-path is required when --model-type is 'vit' or 'cnn'")

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

        self.model_type = model_type
        self.device = device
        self.model = model
        self.processor = SimpleImageProcessor(model_constants.img_size)

    @torch.inference_mode()
    def predict(self, pil_images):
        tensors = []
        for img in pil_images:
            img = expand2square(img, tuple(int(x * 255) for x in self.processor.image_mean))
            t = self.processor.preprocess(img, return_tensors="pt")["pixel_values"][0]
            tensors.append(t)
        return self.predict_tensors(torch.stack(tensors))

    @torch.inference_mode()
    def predict_tensors(self, batch):
        """Same as predict(), but skips PIL decode/resize for already-preprocessed tensors
        (e.g. from a preprocessed_cache_path() cache built by scripts/preprocess_datasets.py)."""
        batch = batch.to(device=self.device, dtype=torch.float32)
        probs = F.softmax(self.model(batch), dim=-1).cpu().numpy()
        scores = (probs * SCORE_WEIGHTS).sum(axis=-1)
        return probs, scores


class TopiqNRModel:
    def __init__(self, device):
        try:
            import pyiqa
        except ImportError:
            raise RuntimeError(
                "--model-type topiq_nr requires the 'pyiqa' package. Install it with `pip install pyiqa`."
            )
        from src.model.pyiqa_loader import configure_pyiqa_cache
        configure_pyiqa_cache()
        self.model_type = "topiq_nr"
        self.device = device
        self.metric = pyiqa.create_metric("topiq_nr", device=device)
        print("Loaded topiq_nr model (pyiqa)")

    @torch.inference_mode()
    def predict(self, pil_images):
        # Native-resolution, one image at a time: topiq_nr's distortion-sensitive scoring
        # depends on not resizing/cropping, and images vary in size so can't be batched.
        # Images above TOPIQ_NR_MAX_SIDE are downscaled to avoid OOM (see comment there).
        scores = []
        for img in pil_images:
            img = _cap_image_size(img, TOPIQ_NR_MAX_SIDE)
            tensor = TF.to_tensor(img).unsqueeze(0).to(self.device)
            scores.append(self.metric(tensor).item())
        scores = _rescale(np.array(scores, dtype=np.float32), TOPIQ_NR_SCORE_RANGE, GT_SCORE_RANGE)
        return None, scores


class MusiqModel:
    """Wraps this repo's manual MUSIQ reimplementation (src/model/musiq): pyiqa's pretrained
    koniq10k tokenizer+encoder weights feeding our own 5-way MeanOpinionScoreHead, which is left
    randomly initialized (never fine-tuned) — the "untuned" baseline behind the same uniform
    interface as TopiqNRModel. A fine-tuned musiq checkpoint is evaluated via CustomModel instead
    (see load_model above), not this class."""

    def __init__(self, device):
        from src.model.musiq import MUSIQModel
        from src.model.musiq.pretrained import load_musiq_koniq_pretrained
        self.model_type = "musiq"
        self.device = device
        model = MUSIQModel().to(device=device, dtype=torch.float32)
        load_musiq_koniq_pretrained(model)
        model.eval()
        self.model = model
        print("Loaded musiq model (manual implementation, pyiqa koniq10k backbone, untuned head)")

    @torch.inference_mode()
    def predict(self, pil_images):
        # Native-resolution, one image at a time: MUSIQ's multiscale patch extraction is defined
        # per-image and images vary in size so can't be batched. Images above MUSIQ_MAX_SIDE are
        # downscaled to avoid OOM (see comment there).
        scores = []
        for img in pil_images:
            img = _cap_image_size(img, MUSIQ_MAX_SIDE)
            tensor = TF.to_tensor(img).unsqueeze(0).to(self.device)
            probs = F.softmax(self.model(tensor), dim=-1).cpu().numpy()
            scores.append(float((probs * SCORE_WEIGHTS).sum()))
        return None, np.array(scores, dtype=np.float64)


MUSIQ_KONIQ_SCORE_RANGE = (0.0, 100.0)  # pyiqa's documented (approximate) score_range for musiq's
                                          # pretrained koniq10k head: "~0, ~100"


class MusiqNativeModel:
    """Wraps pyiqa's own pretrained MUSIQ metric (koniq10k checkpoint, including its native 1-d
    regression head) — unlike MusiqModel above, which reuses this repo's manual reimplementation
    with a freshly-initialized, never-fine-tuned head. Use this where MUSIQ's score needs to
    reflect what the pretrained model actually predicts (e.g. comparing it against other
    pretrained models), rather than MusiqModel's "untuned baseline" behind the same uniform
    interface as TopiqNRModel."""

    def __init__(self, device):
        try:
            import pyiqa
        except ImportError:
            raise RuntimeError(
                "musiq (native) requires the 'pyiqa' package. Install it with `pip install pyiqa`."
            )
        from src.model.pyiqa_loader import configure_pyiqa_cache
        configure_pyiqa_cache()
        self.model_type = "musiq"
        self.device = device
        self.metric = pyiqa.create_metric("musiq", device=device)
        print("Loaded musiq model (pyiqa's own pretrained MUSIQ metric, koniq10k checkpoint, native head)")

    @torch.inference_mode()
    def predict(self, pil_images):
        # Native-resolution, one image at a time: same rationale as TopiqNRModel/MusiqModel above.
        # Images above MUSIQ_MAX_SIDE are downscaled to avoid OOM (see comment there).
        scores = []
        for img in pil_images:
            img = _cap_image_size(img, MUSIQ_MAX_SIDE)
            tensor = TF.to_tensor(img).unsqueeze(0).to(self.device)
            scores.append(self.metric(tensor).item())
        scores = _rescale(np.array(scores, dtype=np.float32), MUSIQ_KONIQ_SCORE_RANGE, GT_SCORE_RANGE)
        return None, scores


