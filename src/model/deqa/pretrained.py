from pathlib import Path

from .constants import DEQA_CACHE_DIR, DEQA_MODEL_ID


def get_deqa_snapshot_dir(cache_dir=DEQA_CACHE_DIR, model_id=DEQA_MODEL_ID):
    """Return the already-downloaded Hugging Face snapshot directory for DeQA-Score-Mix3."""
    cache_dir = Path(cache_dir)
    repo_dir = cache_dir / f"models--{model_id.replace('/', '--')}"
    refs_main = repo_dir / "refs" / "main"

    if refs_main.exists():
        revision = refs_main.read_text().strip()
        snapshot_dir = repo_dir / "snapshots" / revision
        if snapshot_dir.exists():
            return snapshot_dir

    snapshots_dir = repo_dir / "snapshots"
    if snapshots_dir.exists():
        snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
        if snapshots:
            return snapshots[-1]

    raise FileNotFoundError(
        f"DeQA-Score-Mix3 weights are not available locally under {cache_dir}. "
        f"Download {model_id} there before using DeQAScoreIQA."
    )
