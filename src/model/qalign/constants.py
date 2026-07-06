from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
QALIGN_CACHE_DIR = PROJECT_ROOT / "checkpoints" / "qalign_pretrained"
QALIGN_MODEL_ID = "q-future/Q-ReAlign-Mini-0.8B"

# Q-Align's documented prompt-scoring contract: read the next-token distribution over these
# five leading-space quality words, then collapse it to a scalar in [0, 1].
QALIGN_LEVELS = ["excellent", "good", "fair", "poor", "bad"]
QALIGN_LEVEL_WEIGHTS = np.array([1.0, 0.75, 0.5, 0.25, 0.0], dtype=np.float32)
QALIGN_SCORE_RANGE = (0.0, 1.0)
QALIGN_PROMPT = "How would you rate the quality of this image?"
QALIGN_ANSWER_STEM = "The quality of the image is"

# Same OOM/latency cap as the native-resolution TOPIQ and MUSIQ wrappers.
QALIGN_MAX_SIDE = 2048

# Fixed by the downloaded Q-ReAlign-Mini-0.8B config.
TEXT_HIDDEN_SIZE = 1024
VISION_HIDDEN_SIZE = 768
VISION_PATCH_VECTOR_SIZE = 3 * 2 * 16 * 16
VOCAB_SIZE = 248320
