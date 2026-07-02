"""
Download the classic IQA dataset images (KONIQ, SPAQ, KADID10K, TID2013, CSIQ,
PIPAL, LIVE-Wild, AGIQA3K) referenced by data/Data-DeQA-Score/<DATASET>/metas/*.json.

zhiyuanyou/Data-DeQA-Score on the HF Hub only ships label jsons, not pixels, so the
source images are pulled from chaofengc/IQA-PyTorch-Datasets (the pyiqa toolbox's
dataset mirror) and reorganized to match the exact relative paths the meta jsons
expect. Requires prior `huggingface-cli login` (or HF_TOKEN) for hf_hub_download to
pick up.
"""

import argparse
import json
import os
import shutil
import sys
import tarfile
import zipfile

from huggingface_hub import hf_hub_download

sys.path.insert(0, ".")
from src.constants import DOWNLOAD_DATASETS_ARG_SPECS, IQA_DATASET_ARCHIVES, IQA_DATASETS_HF_REPO_ID
from src.datasets.gen_soft_label import generate_soft_labels


def build_target_lookup(metas_dir):
    """Map each source image's basename to its expected path (e.g. "KADID10K/images/x.png"),
    as recorded in every list-of-samples meta json (train/test/soft-label splits). Dict-shaped
    metas like mos.json/split.json carry no per-item path and are skipped."""
    lookup = {}
    if not os.path.isdir(metas_dir):
        return lookup
    for fname in sorted(os.listdir(metas_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(metas_dir, fname)) as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue
        for item in data:
            image = item.get("image") if isinstance(item, dict) else None
            if image:
                lookup[os.path.basename(image)] = image
    return lookup


def extract_matching(archive_path, lookup, data_root, force):
    matched = set()
    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                basename = os.path.basename(info.filename)
                target = lookup.get(basename)
                if target is None:
                    continue
                dest = os.path.join(data_root, target)
                matched.add(basename)
                if os.path.exists(dest) and not force:
                    continue
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with zf.open(info) as src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out)
    else:
        with tarfile.open(archive_path, "r:*") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                basename = os.path.basename(member.name)
                target = lookup.get(basename)
                if target is None:
                    continue
                dest = os.path.join(data_root, target)
                matched.add(basename)
                if os.path.exists(dest) and not force:
                    continue
                src = tf.extractfile(member)
                if src is None:
                    continue
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with open(dest, "wb") as out:
                    shutil.copyfileobj(src, out)
    return matched


def download_dataset(key, data_root, force):
    archive_name, dataset_dir = IQA_DATASET_ARCHIVES[key]
    metas_dir = os.path.join(data_root, dataset_dir, "metas")
    lookup = build_target_lookup(metas_dir)
    if not lookup:
        print(f"[{key}] No meta jsons with an 'image' field found under {metas_dir}, skipping.")
        return

    print(f"[{key}] {len(lookup)} images expected. Downloading {archive_name} from "
          f"{IQA_DATASETS_HF_REPO_ID}...")
    archive_path = hf_hub_download(
        repo_id=IQA_DATASETS_HF_REPO_ID,
        filename=archive_name,
        repo_type="dataset",
    )

    print(f"[{key}] Extracting matching images into {os.path.join(data_root, dataset_dir, 'images')}...")
    matched = extract_matching(archive_path, lookup, data_root, force)

    missing = sorted(set(lookup) - matched)
    print(f"[{key}] {len(matched)}/{len(lookup)} images found in archive.")
    if missing:
        preview = ", ".join(missing[:10])
        print(f"[{key}] WARNING: {len(missing)} images missing from archive, e.g. {preview}")

    if generate_soft_labels(key, data_root, force=force):
        print(f"[{key}] Generated metas/train.json + metas/test.json from mos.json/split.json.")


def main(datasets, data_root, force):
    keys = datasets or sorted(IQA_DATASET_ARCHIVES.keys())
    for key in keys:
        download_dataset(key, data_root, force)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for arg_spec in DOWNLOAD_DATASETS_ARG_SPECS:
        parser.add_argument(*arg_spec["flags"], **arg_spec["kwargs"])
    args = parser.parse_args()
    main(args.datasets, args.data_root, args.force)
