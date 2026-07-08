import torch

from . import constants
from .pretrained import get_deqa_snapshot_dir


class DeQAScoreIQA:
    """DeQA-Score-Mix3 wrapper with the same predict(pil_images) -> (probs, scores) API used by
    src/model/qalign's QAlignMiniIQA -- but unlike that one, this class only works when imported
    from a process already running under deqa_venv (see constants.py for why). It loads the model
    once and scores images directly in-process: no subprocess, no serialization round trip.

    Entry point: scripts/run_eval_deqa.py, invoked as `deqa_venv/bin/python3 scripts/run_eval_deqa.py`.
    """

    def __init__(self, device="cuda:0", load_in_8bit=True):
        # Deferred so merely importing this class (e.g. accidentally, under the main custom_venv)
        # doesn't itself explode -- constructing it does, loudly, with the real transformers error.
        from .mplug_owl2.configuration_mplug_owl2 import MPLUGOwl2Config
        from .mplug_owl2.modeling_llama2 import replace_llama_modality_adaptive
        from .mplug_owl2.modeling_mplug_owl2 import (
            IMAGE_TOKEN_INDEX,
            MPLUGOwl2LlamaForCausalLM,
            expand2square,
            tokenizer_image_token,
        )

        # This monkeypatches transformers.models.llama.modeling_llama process-wide (adds the
        # modality-adaptive k_proj/v_proj/layernorm mPLUG-Owl2 needs) -- fine as long as nothing
        # else in this process needs a plain, unpatched LlamaModel.
        replace_llama_modality_adaptive()

        self.model_type = "deqa"
        self.device = device
        snapshot_dir = get_deqa_snapshot_dir()

        config = MPLUGOwl2Config.from_pretrained(str(snapshot_dir))
        kwargs = {}
        if load_in_8bit:
            kwargs["load_in_8bit"] = True
            kwargs["device_map"] = {"": 0} if device.startswith("cuda") else {"": "cpu"}
        else:
            kwargs["torch_dtype"] = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = MPLUGOwl2LlamaForCausalLM.from_pretrained(
            str(snapshot_dir), config=config, low_cpu_mem_usage=True, **kwargs
        )
        if not load_in_8bit:
            self.model = self.model.to(device)
        self.model.eval()

        self._expand2square = expand2square
        self._tokenizer_image_token = tokenizer_image_token
        self._image_token_index = IMAGE_TOKEN_INDEX
        self._weight_tensor = torch.tensor(
            constants.DEQA_LEVEL_WEIGHTS.tolist(), dtype=torch.float32, device=device
        )
        print(f"Loaded {constants.DEQA_MODEL_ID} (native, {device}, "
              f"{'8-bit' if load_in_8bit else 'fp16/fp32'})")

    @torch.inference_mode()
    def predict(self, pil_images):
        images = [
            self._expand2square(img, tuple(int(x * 255) for x in self.model.image_processor.image_mean))
            for img in pil_images
        ]
        input_ids = self._tokenizer_image_token(
            constants.DEQA_PROMPT_TEMPLATE, self.model.tokenizer, self._image_token_index, return_tensors="pt"
        ).unsqueeze(0).to(self.device)
        image_tensor = self.model.image_processor.preprocess(images, return_tensors="pt")["pixel_values"]
        image_tensor = image_tensor.half().to(self.device)

        # (batch, seq_len, vocab) -> next-token logits at the 5 level words -> (batch, 5).
        level_logits = self.model(
            input_ids.repeat(image_tensor.shape[0], 1), images=image_tensor
        )["logits"][:, -1, self.model.preferential_ids_]
        probs = torch.softmax(level_logits.float(), dim=-1)
        scores = probs @ self._weight_tensor
        return probs.cpu().numpy(), scores.cpu().numpy()
