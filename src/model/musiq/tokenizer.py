import torch.nn as nn

from . import constants
from . import stdconv
from .bottleneck import Bottleneck


class PatchTokenizer(nn.Module):
    """Turns each raw `patch_size` x `patch_size` RGB patch into a `hidden_size`-dim token: a small
    BiT-style ResNet stem (conv + GroupNorm + maxpool + one Bottleneck block) reduces each patch to
    an 8x8 feature map (32 / 4, from the stem's two stride-2 ops), which a linear projection then
    embeds into the transformer's token space."""

    def __init__(self):
        super().__init__()
        self.stem_conv = stdconv.StdConv(
            3, constants.resnet_token_dim,
            kernel_size=constants.stem_conv_kernel_size, stride=constants.stem_conv_stride, bias=False,
        )
        self.stem_norm = nn.GroupNorm(constants.group_norm_groups, constants.resnet_token_dim, eps=constants.root_group_norm_eps)
        self.stem_relu = nn.ReLU(inplace=True)
        self.stem_pad = stdconv.SamePad2d(kernel_size=constants.stem_pool_kernel_size, stride=constants.stem_pool_stride)
        self.stem_pool = nn.MaxPool2d(kernel_size=constants.stem_pool_kernel_size, stride=constants.stem_pool_stride)
        self.stem_block = Bottleneck(constants.resnet_token_dim, constants.resnet_token_dim * constants.bottleneck_expansion)

        token_side = constants.patch_size // constants.stem_spatial_reduction
        token_feature_dim = constants.resnet_token_dim * constants.bottleneck_expansion * token_side * token_side
        self.patch_embedding = nn.Linear(token_feature_dim, constants.hidden_size)

    def forward(self, patches):
        # patches: (num_patches, 3, patch_size, patch_size)
        out = self.stem_conv(patches)  # (num_patches, resnet_token_dim, patch_size / stem_conv_stride, patch_size / stem_conv_stride)
        out = self.stem_norm(out)
        out = self.stem_relu(out)
        out = self.stem_pad(out)  # zero-padded so the stride-2 maxpool below lands on patch_size / 4
        out = self.stem_pool(out)  # (num_patches, resnet_token_dim, patch_size / 4, patch_size / 4)
        out = self.stem_block(out)  # (num_patches, resnet_token_dim * 4, patch_size / 4, patch_size / 4)
        out = out.permute(0, 2, 3, 1)  # match TF's NHWC channel order before flattening
        out = out.flatten(1)  # (num_patches, token_feature_dim)
        return self.patch_embedding(out)  # (num_patches, hidden_size)
