"""
Download every pretrained model weight this repo uses, into the exact local cache locations
the rest of the code expects — so a fresh machine can run both the manual reimplementations
(src/model/*/pretrained.py) and plain `transformers`/`pyiqa` calls without any code path
triggering an on-demand network fetch.

Fetches:
  - dinov2:      facebookresearch/dinov2 ViT-B/14 backbone, via torch.hub
                 (cache: ~/.cache/torch/hub)
  - imagenet:    torchvision EfficientNet-B0 IMAGENET1K_V1 weights
                 (cache: ~/.cache/torch/hub/checkpoints)
  - topiq_nr:    pyiqa's pretrained TOPIQ-NR (CFANet ResNet50) checkpoint
                 (cache: checkpoints/pyiqa_pretrained)
  - musiq_koniq: pyiqa's pretrained MUSIQ koniq10k checkpoint
                 (cache: checkpoints/pyiqa_pretrained)
  - qalign:      q-future/Q-ReAlign-Mini-0.8B full HF snapshot
                 (cache: checkpoints/qalign_pretrained) — the same local snapshot dir that both
                 this repo's manual QAlignMiniForQuality and a plain
                 transformers.AutoModelForImageTextToText.from_pretrained(..., local_files_only=True)
                 call load from.
  - deqa:        zhiyuanyou/DeQA-Score-Mix3 HF snapshot, safetensors only (the duplicate
                 pytorch_model*.bin shards are skipped to save ~16GB of disk)
                 (cache: checkpoints/deqa_pretrained) — loaded by src/model/deqa's
                 deqa_worker.py subprocess (see src/model/deqa/constants.py for why that runs in
                 its own venv instead of in-process like every other model here).

Run with no arguments to fetch everything, or --models to fetch a subset.
"""

import argparse
import sys

sys.path.insert(0, ".")

MODEL_KEYS = ("dinov2", "imagenet", "topiq_nr", "musiq_koniq", "qalign", "deqa")


def download_dinov2():
    import torch

    from src.model.vit.pretrained import DINOV2_HUB_REPO, DINOV2_HUB_MODEL

    print(f"[dinov2] fetching {DINOV2_HUB_REPO}:{DINOV2_HUB_MODEL} via torch.hub...")
    torch.hub.load(DINOV2_HUB_REPO, DINOV2_HUB_MODEL)
    print("[dinov2] done.")


def download_imagenet():
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    print("[imagenet] fetching torchvision efficientnet_b0 IMAGENET1K_V1 weights...")
    efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    print("[imagenet] done.")


def download_topiq_nr():
    from pyiqa.archs.topiq_arch import default_model_urls
    from pyiqa.utils.download_util import load_file_from_url

    from src.model.hybrid.pretrained import TOPIQ_NR_MODEL_NAME
    from src.model.pyiqa_loader import configure_pyiqa_cache

    configure_pyiqa_cache()
    url = default_model_urls[TOPIQ_NR_MODEL_NAME]
    print(f"[topiq_nr] fetching {url}...")
    load_file_from_url(url)
    print("[topiq_nr] done.")


def download_musiq_koniq():
    from pyiqa.utils.download_util import load_file_from_url

    from src.model.musiq.pretrained import MUSIQ_KONIQ_URL
    from src.model.pyiqa_loader import configure_pyiqa_cache

    configure_pyiqa_cache()
    print(f"[musiq_koniq] fetching {MUSIQ_KONIQ_URL}...")
    load_file_from_url(MUSIQ_KONIQ_URL)
    print("[musiq_koniq] done.")


def download_qalign():
    from huggingface_hub import snapshot_download

    from src.model.qalign.constants import QALIGN_CACHE_DIR, QALIGN_MODEL_ID

    print(f"[qalign] fetching {QALIGN_MODEL_ID} snapshot into {QALIGN_CACHE_DIR}...")
    snapshot_download(repo_id=QALIGN_MODEL_ID, cache_dir=QALIGN_CACHE_DIR)
    print("[qalign] done.")


def download_deqa():
    from huggingface_hub import snapshot_download

    from src.model.deqa.constants import DEQA_CACHE_DIR, DEQA_MODEL_ID

    print(f"[deqa] fetching {DEQA_MODEL_ID} snapshot (safetensors only) into {DEQA_CACHE_DIR}...")
    snapshot_download(
        repo_id=DEQA_MODEL_ID,
        cache_dir=DEQA_CACHE_DIR,
        ignore_patterns=["pytorch_model*.bin", "pytorch_model.bin.index.json", "*.md"],
    )
    print("[deqa] done.")


DOWNLOADERS = {
    "dinov2": download_dinov2,
    "imagenet": download_imagenet,
    "topiq_nr": download_topiq_nr,
    "musiq_koniq": download_musiq_koniq,
    "qalign": download_qalign,
    "deqa": download_deqa,
}


def main(models):
    for key in models or MODEL_KEYS:
        DOWNLOADERS[key]()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=MODEL_KEYS, default=None,
                         help="Which pretrained models to download (default: all)")
    args = parser.parse_args()
    main(args.models)
