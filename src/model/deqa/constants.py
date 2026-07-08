from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEQA_CACHE_DIR = PROJECT_ROOT / "checkpoints" / "deqa_pretrained"
DEQA_MODEL_ID = "zhiyuanyou/DeQA-Score-Mix3"

# DeQA-Score's own documented scoring contract (see its vendored .score() in
# mplug_owl2/modeling_mplug_owl2.py): read the next-token distribution over these five
# leading words, weight by [5, 4, 3, 2, 1]. Unlike Q-Align (see src/model/qalign/constants.py),
# this is already on this repo's [1, 5] MOS scale, so no rescaling is needed.
DEQA_LEVELS = ["excellent", "good", "fair", "poor", "bad"]
DEQA_LEVEL_WEIGHTS = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32)
DEQA_SCORE_RANGE = (1.0, 5.0)
DEQA_PROMPT_TEMPLATE = (
    "USER: How would you rate the quality of this image?\n<|image|>\nASSISTANT: The quality of the image is"
)

# The mPLUG-Owl2/LLaMA-2-7B modeling code this model is built on (vendored under ./mplug_owl2)
# monkeypatches transformers.models.llama.modeling_llama internals that transformers 5.x removed
# (see mplug_owl2/modeling_llama2.py), so it only runs under an older pinned transformers -- see
# ../../../deqa_venv, pinned to transformers==4.46.3. Rather than downgrading the shared venv
# (breaking src/model/qalign's Qwen3.5 support), DeQAScoreIQA (model.py) must be imported and used
# from a process already running under deqa_venv: scripts/run_eval_deqa.py, invoked as
# `deqa_venv/bin/python3 scripts/run_eval_deqa.py`. Constructing it under the main custom_venv
# will fail (the same transformers-API mismatches deqa_venv exists to route around), so nothing
# in the main venv's scripts (script_utils.py etc.) references DeQAScoreIQA.
