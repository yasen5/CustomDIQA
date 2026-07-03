from pyiqa.archs.arch_util import load_pretrained_network
from pyiqa.archs.topiq_arch import default_model_urls

from ..pyiqa_loader import configure_pyiqa_cache

TOPIQ_NR_MODEL_NAME = "cfanet_nr_koniq_res50"


def load_topiq_nr_pretrained(model, verbose=True):
    """Copies pyiqa's pretrained TOPIQ-NR weights (CFANet: ResNet50 semantic backbone +
    cross-scale attention, NR-trained on KonIQ) into `model.backbone`, in place. One-time
    network fetch, cached locally under pyiqa_loader.PYIQA_CACHE_DIR afterward.

    strict=False because `model.backbone.score_linear` was replaced with nn.Identity() in
    HybridModel.__init__ (TOPIQ's own 1-d regression head is discarded) — the checkpoint's
    score_linear.* keys are simply left unused. model.head (our 5-way head) is left as
    whatever HybridModel.__init__ already initialized it to.
    """
    configure_pyiqa_cache()
    load_pretrained_network(
        model.backbone, default_model_urls[TOPIQ_NR_MODEL_NAME], strict=False, weight_keys="params"
    )
    if verbose:
        transferred = sum(
            p.numel() for name, p in model.backbone.named_parameters() if not name.startswith("score_linear")
        )
        total = sum(p.numel() for p in model.parameters())
        print(f"Loaded pyiqa TOPIQ-NR (CFANet ResNet50 + cross-scale attention) pretrained "
              f"weights into backbone ({transferred}/{total} params, "
              f"{100 * transferred / total:.1f}%); model.head left randomly initialized")
