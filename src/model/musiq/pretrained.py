import torch

from ..pyiqa_loader import configure_pyiqa_cache

MUSIQ_KONIQ_URL = (
    "https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/musiq_koniq_ckpt-e95806b9.pth"
)


def _fetch_musiq_koniq_state_dict():
    # One-time network fetch (huggingface.co), cached under pyiqa_loader.PYIQA_CACHE_DIR afterward.
    from pyiqa.archs.arch_util import clean_state_dict
    from pyiqa.utils.download_util import load_file_from_url

    configure_pyiqa_cache()
    ckpt_path = load_file_from_url(MUSIQ_KONIQ_URL)
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return clean_state_dict(state_dict)


def _copy_bottleneck(dst, src, prefix):
    dst.conv1.weight.copy_(src[f"{prefix}.conv1.weight"])
    dst.norm1.weight.copy_(src[f"{prefix}.gn1.weight"])
    dst.norm1.bias.copy_(src[f"{prefix}.gn1.bias"])
    dst.conv2.weight.copy_(src[f"{prefix}.conv2.weight"])
    dst.norm2.weight.copy_(src[f"{prefix}.gn2.weight"])
    dst.norm2.bias.copy_(src[f"{prefix}.gn2.bias"])
    dst.conv3.weight.copy_(src[f"{prefix}.conv3.weight"])
    dst.norm3.weight.copy_(src[f"{prefix}.gn3.weight"])
    dst.norm3.bias.copy_(src[f"{prefix}.gn3.bias"])
    if dst.needs_projection:
        dst.proj_conv.weight.copy_(src[f"{prefix}.conv_proj.weight"])
        dst.proj_norm.weight.copy_(src[f"{prefix}.gn_proj.weight"])
        dst.proj_norm.bias.copy_(src[f"{prefix}.gn_proj.bias"])


def _copy_block(dst, src, prefix):
    dst.attention_layernorm.weight.copy_(src[f"{prefix}.norm1.weight"])
    dst.attention_layernorm.bias.copy_(src[f"{prefix}.norm1.bias"])
    dst.attention.query_key_value.weight.copy_(
        torch.cat(
            [
                src[f"{prefix}.attention.query.weight"],
                src[f"{prefix}.attention.key.weight"],
                src[f"{prefix}.attention.value.weight"],
            ],
            dim=0,
        )
    )
    dst.attention.query_key_value.bias.copy_(
        torch.cat(
            [
                src[f"{prefix}.attention.query.bias"],
                src[f"{prefix}.attention.key.bias"],
                src[f"{prefix}.attention.value.bias"],
            ],
            dim=0,
        )
    )
    dst.attention.out_proj.weight.copy_(src[f"{prefix}.attention.out.weight"])
    dst.attention.out_proj.bias.copy_(src[f"{prefix}.attention.out.bias"])
    dst.feedforward_layernorm.weight.copy_(src[f"{prefix}.norm2.weight"])
    dst.feedforward_layernorm.bias.copy_(src[f"{prefix}.norm2.bias"])
    dst.feed_forward.dense1.weight.copy_(src[f"{prefix}.mlp.fc1.weight"])
    dst.feed_forward.dense1.bias.copy_(src[f"{prefix}.mlp.fc1.bias"])
    dst.feed_forward.dense2.weight.copy_(src[f"{prefix}.mlp.fc2.weight"])
    dst.feed_forward.dense2.bias.copy_(src[f"{prefix}.mlp.fc2.bias"])


def load_musiq_koniq_pretrained(model, verbose=True):
    """Copies pyiqa's pretrained MUSIQ weights (koniq10k checkpoint: ResNet-token stem + multi-scale
    transformer, NR-trained on KonIQ-10k) into `model` (a MUSIQModel), in place. This covers every
    parameter except model.head: pyiqa's koniq10k checkpoint ends in a 1-d regression head, which
    has no matching shape in our 5-way MeanOpinionScoreHead (see head.py), so — matching the
    vit/cnn/hybrid pretrained loaders — model.head is left as whatever MUSIQModel.__init__ already
    initialized it to.
    """
    src = _fetch_musiq_koniq_state_dict()

    with torch.no_grad():
        model.tokenizer.stem_conv.weight.copy_(src["conv_root.weight"])
        model.tokenizer.stem_norm.weight.copy_(src["gn_root.weight"])
        model.tokenizer.stem_norm.bias.copy_(src["gn_root.bias"])
        _copy_bottleneck(model.tokenizer.stem_block, src, "block1")
        model.tokenizer.patch_embedding.weight.copy_(src["embedding.weight"])
        model.tokenizer.patch_embedding.bias.copy_(src["embedding.bias"])

        model.encoder.cls_token.copy_(src["transformer_encoder.cls"])
        model.encoder.spatial_position_embedding.position_emb.copy_(
            src["transformer_encoder.posembed_input.position_emb"]
        )
        model.encoder.scale_embedding.scale_emb.copy_(src["transformer_encoder.scaleembed_input.scale_emb"])
        model.encoder.final_layernorm.weight.copy_(src["transformer_encoder.encoder_norm.weight"])
        model.encoder.final_layernorm.bias.copy_(src["transformer_encoder.encoder_norm.bias"])

        for i, block in enumerate(model.encoder.blocks):
            _copy_block(block, src, f"transformer_encoder.transformer.encoderblock_{i}")

    if verbose:
        transferred = (
            sum(p.numel() for p in model.tokenizer.parameters())
            + sum(p.numel() for p in model.encoder.parameters())
        )
        total = sum(p.numel() for p in model.parameters())
        print(f"Loaded pyiqa MUSIQ (koniq10k) pretrained weights into tokenizer+encoder "
              f"({transferred}/{total} params, {100 * transferred / total:.1f}%); "
              f"model.head left randomly initialized")
