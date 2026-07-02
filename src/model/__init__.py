import os

import torch

from .vit.model import EncoderModel
from .vit import constants as vit_constants
from .cnn.model import EfficientNet
from .cnn import constants as cnn_constants

MODEL_REGISTRY = {
    "vit": (EncoderModel, vit_constants),
    "cnn": (EfficientNet, cnn_constants),
}

MODEL_TYPE_FILENAME = "model_type.txt"


def build_model(model_type: str, pretrained: str | None = None):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type {model_type!r}, expected one of {list(MODEL_REGISTRY)}")
    model_class, constants = MODEL_REGISTRY[model_type]
    model = model_class()
    if pretrained == "dinov2":
        if model_type != "vit":
            raise ValueError("--pretrained dinov2 is only supported for --model-type vit")
        from .vit.pretrained import load_dinov2_partial
        load_dinov2_partial(model)
    elif pretrained == "imagenet":
        if model_type != "cnn":
            raise ValueError("--pretrained imagenet is only supported for --model-type cnn")
        from .cnn.pretrained import load_imagenet_efficientnet_b0
        load_imagenet_efficientnet_b0(model)
    elif pretrained is not None:
        raise ValueError(f"Unknown pretrained option {pretrained!r}")
    return model, constants


def save_model_type(checkpoint_dir: str, model_type: str):
    with open(os.path.join(checkpoint_dir, MODEL_TYPE_FILENAME), "w") as f:
        f.write(model_type)


def load_model_type(checkpoint_dir: str, default: str = "vit") -> str:
    marker_path = os.path.join(checkpoint_dir, MODEL_TYPE_FILENAME)
    if not os.path.isfile(marker_path):
        print(f"WARNING: {marker_path} not found, assuming model type {default!r}")
        return default
    with open(marker_path) as f:
        return f.read().strip()


def load_checkpoint(weights_path: str, map_location=None):
    """Load a weights.pt file, returning (model_state_dict, optimizer_state_dict)."""
    raw = torch.load(weights_path, map_location=map_location)
    return raw["model"], raw.get("optimizer")
