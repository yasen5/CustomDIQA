from pathlib import Path

from safetensors.torch import load_model

from .constants import QALIGN_CACHE_DIR, QALIGN_MODEL_ID


def get_qalign_snapshot_dir(cache_dir=QALIGN_CACHE_DIR, model_id=QALIGN_MODEL_ID):
    """Return the already-downloaded Hugging Face snapshot directory for Q-Align Mini."""
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
        f"Q-Align Mini weights are not available locally under {cache_dir}. "
        f"Download {model_id} there before using the manual implementation."
    )


def load_qalign_safetensors(model, snapshot_dir, strict=True):
    """Load the local model.safetensors file into an instantiated Qwen3.5/Q-Align module."""
    snapshot_dir = Path(snapshot_dir)
    weights_path = snapshot_dir / "model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing Q-Align weights file: {weights_path}")

    missing, unexpected = load_model(model, str(weights_path), strict=False)

    # The Qwen3.5 config ties lm_head.weight to model.language_model.embed_tokens.weight.
    # safetensors stores both names, but the instantiated PyTorch module has one shared Parameter.
    expected_unexpected = {"model.language_model.embed_tokens.weight"}
    unexpected = set(unexpected) - expected_unexpected

    if strict and (missing or unexpected):
        raise RuntimeError(
            f"Failed to load Q-Align weights strictly: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    return missing, unexpected
