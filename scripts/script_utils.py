import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from src.model import build_model, load_checkpoint, load_model_type
from src.trainer import SimpleImageProcessor
from src.utils import expand2square

MODEL_CHOICES = ("vit", "cnn", "hybrid", "topiq_nr")
SCORE_WEIGHTS = np.array([5, 4, 3, 2, 1], dtype=np.float32)

TOPIQ_NR_SCORE_RANGE = (0.0, 1.0)  # pyiqa's documented (approximate) score_range for topiq_nr: "~0, ~1"
GT_SCORE_RANGE = (1.0, 5.0)        # this repo's MOS scale (see gen_soft_label.py mos_norm, SCORE_WEIGHTS above)

# topiq_nr's CFANet has no built-in resize for this checkpoint (test_img_size=None in pyiqa's
# config), so its cross-attention token count grows with input pixel count. Raw phone photos
# (e.g. SPAQ, ~3000-5500px) OOM an 11GB GPU; this cap only kicks in above that size, so every
# other dataset here (already downscaled) still scores at true native resolution.
TOPIQ_NR_MAX_SIDE = 2048


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
                              "dir's model_type.txt); ignored for topiq_nr, which uses pyiqa's own "
                              "pretrained weights.")
    parser.add_argument("--model-type", choices=MODEL_CHOICES, default=None,
                         help="'vit'/'cnn'/'hybrid' load a trained checkpoint from --model-path (overrides "
                              "the checkpoint's recorded type; required if --model-path is a weights "
                              "file rather than a checkpoint dir). 'topiq_nr' uses pyiqa's pretrained "
                              "TOPIQ-NR no-reference IQA metric instead of a local checkpoint.")


def load_model(args, device):
    """Factory returning a model wrapper with a uniform .predict(pil_images) -> (probs_or_None, scores)
    interface and a .model_type label, regardless of whether the backend is a local checkpoint or a
    pyiqa metric."""
    if args.model_type == "topiq_nr":
        return TopiqNRModel(device)
    return CustomModel(args, device)


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
        batch = torch.stack(tensors).to(device=self.device, dtype=torch.float32)
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
