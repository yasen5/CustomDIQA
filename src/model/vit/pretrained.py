import torch

from . import constants

DINOV2_HUB_REPO = "facebookresearch/dinov2"
DINOV2_HUB_MODEL = "dinov2_vitb14"


def _fetch_dinov2_state_dict():
    # One-time network fetch (github.com + dl.fbaipublicfiles.com), cached
    # under ~/.cache/torch/hub afterward. Only invoked when --pretrained dinov2
    # is passed.
    model = torch.hub.load(DINOV2_HUB_REPO, DINOV2_HUB_MODEL)
    return model.state_dict()


def _copy_block(dst_layer, src_sd, src_prefix, has_layerscale):
    dst_layer.embedding_layernorm.weight.copy_(src_sd[f"{src_prefix}.norm1.weight"])
    dst_layer.embedding_layernorm.bias.copy_(src_sd[f"{src_prefix}.norm1.bias"])
    dst_layer.attention_layernorm.weight.copy_(src_sd[f"{src_prefix}.norm2.weight"])
    dst_layer.attention_layernorm.bias.copy_(src_sd[f"{src_prefix}.norm2.bias"])

    dst_layer.attention.query_key_value.weight.copy_(src_sd[f"{src_prefix}.attn.qkv.weight"])
    dst_layer.attention.query_key_value.bias.copy_(src_sd[f"{src_prefix}.attn.qkv.bias"])
    dst_layer.attention.dense.weight.copy_(src_sd[f"{src_prefix}.attn.proj.weight"])
    dst_layer.attention.dense.bias.copy_(src_sd[f"{src_prefix}.attn.proj.bias"])

    # GELU (source) vs QuickGELU (ours) — weights transfer fine, brief
    # re-adaptation happens during fine-tuning.
    dst_layer.feed_forward.dense1.weight.copy_(src_sd[f"{src_prefix}.mlp.fc1.weight"])
    dst_layer.feed_forward.dense1.bias.copy_(src_sd[f"{src_prefix}.mlp.fc1.bias"])
    dst_layer.feed_forward.dense2.weight.copy_(src_sd[f"{src_prefix}.mlp.fc2.weight"])
    dst_layer.feed_forward.dense2.bias.copy_(src_sd[f"{src_prefix}.mlp.fc2.bias"])

    if has_layerscale:
        dst_layer.layerscale1.gamma.copy_(src_sd[f"{src_prefix}.ls1.gamma"])
        dst_layer.layerscale2.gamma.copy_(src_sd[f"{src_prefix}.ls2.gamma"])


def load_dinov2_partial(model, num_layers=None, verbose=True):
    """Copies DINOv2 ViT-B/14's patch-embed conv and first `num_layers`
    transformer blocks into `model` (an EncoderModel), in place.

    Not transferred: DINOv2's remaining blocks, its cls_token, and its
    absolute position embeddings (this model has no slot for them — it uses
    RoPE instead, applied unchanged inside EncoderAttention). model.head and
    model.final_encoder_layernorm are left as whatever EncoderModel.__init__
    already initialized them to.
    """
    num_layers = num_layers or constants.num_vit_layers
    src_sd = _fetch_dinov2_state_dict()
    has_layerscale = any(k.startswith("blocks.0.ls1") for k in src_sd)

    with torch.no_grad():
        model.embedding.weight.copy_(src_sd["patch_embed.proj.weight"])
        model.embedding.bias.copy_(src_sd["patch_embed.proj.bias"])
        for i in range(num_layers):
            _copy_block(model.layers[i], src_sd, f"blocks.{i}", has_layerscale)

    if verbose:
        transferred = sum(p.numel() for p in model.layers[:num_layers].parameters())
        transferred += model.embedding.weight.numel() + model.embedding.bias.numel()
        print(f"Loaded DINOv2 ViT-B/14 patch-embed + first {num_layers}/12 blocks "
              f"({transferred} params transferred)")
