import os

# checkpoints/ is already gitignored and used for our own run outputs, so it doubles as a
# stable, project-local spot for third-party pretrained weights too.
PYIQA_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "checkpoints", "pyiqa_pretrained",
)

_configured = False


def configure_pyiqa_cache():
    """Redirect every pyiqa weight download to PYIQA_CACHE_DIR instead of pyiqa's default
    (~/.cache/torch/hub/pyiqa). Call this before any code path that may trigger a pyiqa
    download (pyiqa.create_metric(...), load_pretrained_network(...), or constructing a
    pyiqa arch with pretrained=True) — the first call downloads the weights file into
    PYIQA_CACHE_DIR, and every call after (including in a fresh process, or a sandboxed/
    ephemeral environment without a persistent home cache) finds it already there and skips
    the network fetch entirely.

    Safe to call repeatedly / from multiple call sites: only the first call has any effect.
    Must run before the download happens, but can run after `import pyiqa` — pyiqa's
    download helper re-reads its module-level cache-dir constant on every call rather than
    baking it in at import time, so patching that constant here still takes effect.
    """
    global _configured
    if _configured:
        return
    import pyiqa.utils.download_util as pyiqa_download_util

    os.makedirs(PYIQA_CACHE_DIR, exist_ok=True)
    pyiqa_download_util.DEFAULT_CACHE_DIR = PYIQA_CACHE_DIR
    _configured = True
