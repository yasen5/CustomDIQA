import torch
import torch.nn as nn
import torch.nn.functional as F

from . import constants
from .pretrained import get_qalign_snapshot_dir, load_qalign_safetensors


def _cap_image_size(img, max_side):
    w, h = img.size
    longest = max(w, h)
    if longest <= max_side:
        return img
    scale = max_side / longest
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), resample=3)


def _rescale(x, src_range, dst_range):
    src_lo, src_hi = src_range
    dst_lo, dst_hi = dst_range
    return dst_lo + (x - src_lo) * (dst_hi - dst_lo) / (src_hi - src_lo)


class QAlignMiniForQuality(nn.Module):
    """Manual top-level Q-Align Mini forward pass over the local Qwen3.5 weights.

    This intentionally does not call AutoModelForImageTextToText.from_pretrained. It instantiates
    the concrete Qwen3.5 class from the local config, loads model.safetensors from the already
    downloaded snapshot, then spells out the image-feature insertion and language-model scoring
    path so the tensor dimensions are visible at each stage.
    """

    def __init__(self, snapshot_dir=None):
        super().__init__()
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

        self.snapshot_dir = get_qalign_snapshot_dir() if snapshot_dir is None else snapshot_dir
        config = Qwen3_5Config.from_pretrained(self.snapshot_dir, local_files_only=True)

        # Build parameters directly in the checkpoint dtype: bfloat16 for Q-ReAlign-Mini-0.8B.
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(config.dtype)
        try:
            self.qwen = Qwen3_5ForConditionalGeneration(config)
        finally:
            torch.set_default_dtype(default_dtype)

        load_qalign_safetensors(self.qwen, self.snapshot_dir)
        self.config = config

    def forward(
        self,
        input_ids,
        attention_mask=None,
        pixel_values=None,
        image_grid_thw=None,
        mm_token_type_ids=None,
        level_token_ids=None,
    ):
        """Return either full next-token logits or the five Q-Align level logits.

        Args:
            input_ids: token ids, (batch_size, seq_len)
            attention_mask: valid-token mask, (batch_size, seq_len)
            pixel_values: flattened vision patches, (sum(T * H * W), 1536)
            image_grid_thw: per-image vision grid, (num_images, 3) with columns T, H, W
            mm_token_type_ids: 0=text/1=image token types, (batch_size, seq_len)
            level_token_ids: optional quality word ids, (5,)
        """
        qwen_model = self.qwen.model

        # Text embedding: (batch_size, seq_len) -> (batch_size, seq_len, 1024).
        inputs_embeds = qwen_model.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            # Processor output: (sum(T * H * W), 1536), where 1536 = 3 * 2 * 16 * 16.
            # Visual patch embed inside Qwen: view to (-1, 3, 2, 16, 16), Conv3d to
            # (sum(T * H * W), 768), 12 vision blocks keep (sum(T * H * W), 768), then
            # 2x2 spatial merger returns one 1024-d vector per LLM image token:
            # tuple[(T * H * W / 4, 1024), ...].
            image_outputs = qwen_model.get_image_features(
                pixel_values,
                image_grid_thw,
                return_dict=True,
            )

            # Concatenate all images in the batch: (num_image_tokens, 1024).
            image_embeds = torch.cat(image_outputs.pooler_output, dim=0)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

            # Placeholder mask: (batch_size, seq_len, 1), true where input_ids are <|image_pad|>.
            image_mask, _ = qwen_model.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_embeds,
            )

            # Replace image placeholder token embeddings in-place logically, preserving shape:
            # (batch_size, seq_len, 1024).
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        # M-RoPE positions for text plus image tokens: (3, batch_size, seq_len).
        position_ids = qwen_model.compute_3d_position_ids(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            past_key_values=None,
            mm_token_type_ids=mm_token_type_ids,
        )

        # Qwen3.5 text stack: embeddings (batch_size, seq_len, 1024) through 24 decoder layers,
        # preserving width, then final RMSNorm -> (batch_size, seq_len, 1024).
        language_outputs = qwen_model.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=False,
        )
        hidden_states = language_outputs.last_hidden_state

        # LM head: (batch_size, seq_len, 1024) -> (batch_size, seq_len, 248320).
        logits = self.qwen.lm_head(hidden_states)

        if level_token_ids is None:
            return logits

        # Q-Align scoring reads only the next-token distribution at the final prompt position:
        # (batch_size, seq_len, 248320) -> (batch_size, 5).
        return logits[:, -1, level_token_ids]


class QAlignMiniIQA:
    """Q-Align Mini IQA wrapper with the same predict(pil_images) API used by eval scripts."""

    def __init__(self, device):
        from transformers import AutoProcessor

        self.model_type = "qalign_mini"
        self.device = device
        self.snapshot_dir = get_qalign_snapshot_dir()
        self.processor = AutoProcessor.from_pretrained(self.snapshot_dir, local_files_only=True)
        self.model = QAlignMiniForQuality(self.snapshot_dir).to(device).eval()

        # The checkpoint loads in bf16 (see QAlignMiniForQuality), but bf16 tensor cores only
        # exist from Ampere (compute capability 8.0) onward — on older cards (e.g. Turing/RTX
        # 2080Ti, cc 7.5) bf16 dtype is still *accepted* by CUDA (torch.cuda.is_bf16_supported()
        # returns True there too) but matmuls silently run through a slow emulated path (~100x
        # slower measured on a 2080Ti). fp16 runs at full tensor-core speed on those older cards
        # and is safe for this use: qalign_mini only does short-sequence inference, not training,
        # so fp16's narrower dynamic range isn't a practical concern.
        if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] < 8:
            print(f"NOTE: {device} ({torch.cuda.get_device_name(device)}) has no bf16 tensor cores"
                  " — running qalign_mini in fp16 instead")
            self.model = self.model.half()

        # Leading-space quality words are single-token ids in this tokenizer: (5,).
        self.level_token_ids = torch.tensor(
            [
                self.processor.tokenizer(" " + level, add_special_tokens=False).input_ids[0]
                for level in constants.QALIGN_LEVELS
            ],
            device=device,
            dtype=torch.long,
        )

        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": constants.QALIGN_PROMPT},
        ]}]
        self._prompt_text = (
            self.processor.apply_chat_template(messages, add_generation_prompt=True)
            + constants.QALIGN_ANSWER_STEM
        )
        print(f"Loaded {constants.QALIGN_MODEL_ID} (manual local Q-Align Mini)")

    @torch.inference_mode()
    def predict(self, pil_images):
        scores = []
        all_probs = []
        for img in pil_images:
            img = _cap_image_size(img, constants.QALIGN_MAX_SIDE)
            inputs = self.processor(
                text=[self._prompt_text],
                images=[img],
                return_tensors="pt",
            ).to(self.device)

            # input_ids/attention/mm types: (1, seq_len); pixel_values: (T*H*W, 1536);
            # image_grid_thw: (1, 3). level_logits: (1, 5).
            level_logits = self.model(**inputs, level_token_ids=self.level_token_ids)
            probs = F.softmax(level_logits.float(), dim=-1).cpu().numpy()[0]
            all_probs.append(probs)
            scores.append(float((probs * constants.QALIGN_LEVEL_WEIGHTS).sum()))

        import numpy as np

        scores = _rescale(
            np.array(scores, dtype=np.float64),
            constants.QALIGN_SCORE_RANGE,
            (1.0, 5.0),
        )
        return np.stack(all_probs), scores
