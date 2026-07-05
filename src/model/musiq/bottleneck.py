import torch.nn as nn

from . import constants
from . import stdconv


class Bottleneck(nn.Module):
    """A single BiT-style (weight-standardized conv + GroupNorm) ResNet bottleneck block — MUSIQ
    runs one of these over each raw image patch as its per-patch tokenizer, before the linear patch
    embedding. Always used at stride 1 in MUSIQ (there's only ever this one block), so unlike a
    general-purpose ResNet bottleneck this has no stride parameter."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = stdconv.StdConv(in_channels, in_channels, kernel_size=1, bias=False)
        self.norm1 = nn.GroupNorm(constants.group_norm_groups, in_channels, eps=constants.block_group_norm_eps)
        self.conv2 = stdconv.StdConv(in_channels, in_channels, kernel_size=3, bias=False)
        self.norm2 = nn.GroupNorm(constants.group_norm_groups, in_channels, eps=constants.block_group_norm_eps)
        self.conv3 = stdconv.StdConv(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm3 = nn.GroupNorm(constants.group_norm_groups, out_channels, eps=constants.block_group_norm_eps)
        self.relu = nn.ReLU(inplace=True)

        # MUSIQ's stem block always goes in_channels=resnet_token_dim -> out_channels=resnet_token_dim
        # * bottleneck_expansion, so the identity shortcut always needs this projection to match shapes.
        self.needs_projection = in_channels != out_channels
        if self.needs_projection:
            self.proj_conv = stdconv.StdConv(in_channels, out_channels, kernel_size=1, bias=False)
            self.proj_norm = nn.GroupNorm(constants.group_norm_groups, out_channels, eps=constants.block_group_norm_eps)

    def forward(self, x):
        identity = x
        if self.needs_projection:
            identity = self.proj_norm(self.proj_conv(identity))

        out = self.relu(self.norm1(self.conv1(x)))  # (N, width, H, W)
        out = self.relu(self.norm2(self.conv2(out)))  # (N, width, H, W)
        out = self.norm3(self.conv3(out))  # (N, out_channels, H, W)
        return self.relu(out + identity)
