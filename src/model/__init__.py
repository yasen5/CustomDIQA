import os

from .vit.model import EncoderModel
from .vit import constants as vit_constants
from .cnn.model import EfficientNet
from .cnn import constants as cnn_constants

MODEL_REGISTRY = {
    "vit": (EncoderModel, vit_constants),
    "cnn": (EfficientNet, cnn_constants),
}

MODEL_TYPE_FILENAME = "model_type.txt"


def build_model(model_type: str):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type {model_type!r}, expected one of {list(MODEL_REGISTRY)}")
    model_class, constants = MODEL_REGISTRY[model_type]
    return model_class(), constants


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
