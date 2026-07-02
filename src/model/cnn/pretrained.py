import torch

IMAGENET_HUB_MODEL = "efficientnet_b0"


def _copy_conv_bn(dst_seq, src_seq):
    # Both are nn.Sequential(Conv2d, BatchNorm2d, [SiLU]) — the head_conv, stem,
    # expand, depthwise and project sub-blocks all share this shape.
    dst_seq[0].weight.copy_(src_seq[0].weight)
    dst_seq[1].weight.copy_(src_seq[1].weight)
    dst_seq[1].bias.copy_(src_seq[1].bias)
    dst_seq[1].running_mean.copy_(src_seq[1].running_mean)
    dst_seq[1].running_var.copy_(src_seq[1].running_var)


def _copy_se(dst_se, src_se):
    dst_se.reduce.weight.copy_(src_se.fc1.weight)
    dst_se.reduce.bias.copy_(src_se.fc1.bias)
    dst_se.expand.weight.copy_(src_se.fc2.weight)
    dst_se.expand.bias.copy_(src_se.fc2.bias)


def _fetch_efficientnet_b0_state():
    # One-time network fetch (download.pytorch.org), cached under
    # ~/.cache/torch/hub/checkpoints/ afterward. Only invoked when
    # --pretrained imagenet is passed.
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    return efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)


def load_imagenet_efficientnet_b0(model, verbose=True):
    """Copies torchvision's ImageNet-pretrained EfficientNet-B0 weights into
    `model` (an EfficientNet), in place. `model`'s stem/blocks/head_conv are
    architecturally identical to torchvision's efficientnet_b0 (same MBConv+SE
    stage config), so this transfers the entire backbone — everything except
    model.head (the 5-way MeanOpinionScoreHead), which is left as whatever
    EfficientNet.__init__ already initialized it to.
    """
    src = _fetch_efficientnet_b0_state()

    with torch.no_grad():
        _copy_conv_bn(model.stem, src.features[0])
        block_idx = 0
        for stage_idx in range(1, 8):
            for src_block in src.features[stage_idx]:
                dst_block = model.blocks[block_idx]
                layers = src_block.block
                idx = 0
                if dst_block.expand is not None:
                    _copy_conv_bn(dst_block.expand, layers[idx])
                    idx += 1
                _copy_conv_bn(dst_block.depthwise, layers[idx])
                idx += 1
                _copy_se(dst_block.squeeze_excite, layers[idx])
                idx += 1
                _copy_conv_bn(dst_block.project, layers[idx])
                block_idx += 1
        _copy_conv_bn(model.head_conv, src.features[8])

    if verbose:
        transferred = (
            sum(p.numel() for p in model.stem.parameters())
            + sum(p.numel() for p in model.blocks.parameters())
            + sum(p.numel() for p in model.head_conv.parameters())
        )
        total = sum(p.numel() for p in model.parameters())
        print(f"Loaded torchvision EfficientNet-B0 ImageNet weights into stem+{block_idx} "
              f"blocks+head_conv ({transferred}/{total} params, "
              f"{100 * transferred / total:.1f}%); model.head left randomly initialized")
