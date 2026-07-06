import argparse
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageTextToText

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.qalign import constants
from src.model.qalign.model import QAlignMiniForQuality
from src.model.qalign.pretrained import get_qalign_snapshot_dir


def build_prompt(processor):
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": constants.QALIGN_PROMPT},
    ]}]
    return processor.apply_chat_template(messages, add_generation_prompt=True) + constants.QALIGN_ANSWER_STEM


def main():
    parser = argparse.ArgumentParser(description="Compare manual Q-Align Mini against Transformers AutoModel.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from transformers import AutoProcessor

    snapshot_dir = get_qalign_snapshot_dir()
    processor = AutoProcessor.from_pretrained(snapshot_dir, local_files_only=True)
    prompt = build_prompt(processor)
    image = Image.new("RGB", (64, 48), (128, 64, 32))
    inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(args.device)
    level_token_ids = torch.tensor(
        [processor.tokenizer(" " + level, add_special_tokens=False).input_ids[0] for level in constants.QALIGN_LEVELS],
        device=args.device,
        dtype=torch.long,
    )

    manual = QAlignMiniForQuality(snapshot_dir).to(args.device).eval()
    reference = AutoModelForImageTextToText.from_pretrained(
        snapshot_dir,
        local_files_only=True,
        dtype="auto",
    ).to(args.device).eval()

    with torch.inference_mode():
        manual_logits = manual(**inputs, level_token_ids=level_token_ids)
        reference_logits = reference(**inputs).logits[:, -1, level_token_ids]

    logit_diff = (manual_logits - reference_logits).abs().max().item()
    manual_probs = F.softmax(manual_logits.float(), dim=-1)
    reference_probs = F.softmax(reference_logits.float(), dim=-1)
    prob_diff = (manual_probs - reference_probs).abs().max().item()
    manual_score = float((manual_probs.cpu().numpy() * constants.QALIGN_LEVEL_WEIGHTS).sum())
    reference_score = float((reference_probs.cpu().numpy() * constants.QALIGN_LEVEL_WEIGHTS).sum())

    print(f"manual logits:    {manual_logits.detach().cpu().tolist()}")
    print(f"reference logits: {reference_logits.detach().cpu().tolist()}")
    print(f"max logit diff:   {logit_diff}")
    print(f"max prob diff:    {prob_diff}")
    print(f"manual score:     {manual_score}")
    print(f"reference score:  {reference_score}")
    if logit_diff != 0.0 or prob_diff != 0.0 or manual_score != reference_score:
        raise SystemExit("manual Q-Align output does not exactly match AutoModel output")


if __name__ == "__main__":
    main()
