import torch
import torch.nn as nn
from . import constants
from .squeeze_excite import SqueezeExcite

class MBConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, expand_ratio: int, stride: int, kernel_size: int, drop_connect_rate: float):
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        self.drop_connect_rate = drop_connect_rate

        expanded_channels = in_channels * expand_ratio
        self.expand = None
        if expand_ratio != 1:
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(expanded_channels, momentum=constants.batch_norm_momentum, eps=constants.batch_norm_epsilon),
                nn.SiLU(inplace=True),
            )

        self.depthwise = nn.Sequential(
            nn.Conv2d(
                expanded_channels,
                expanded_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=expanded_channels,
                bias=False,
            ),
            nn.BatchNorm2d(expanded_channels, momentum=constants.batch_norm_momentum, eps=constants.batch_norm_epsilon),
            nn.SiLU(inplace=True),
        )

        self.squeeze_excite = SqueezeExcite(in_channels, expanded_channels)

        self.project = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=constants.batch_norm_momentum, eps=constants.batch_norm_epsilon),
        )

    def _drop_connect(self, feature_map: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_connect_rate == 0.0:
            return feature_map
        keep_prob = 1.0 - self.drop_connect_rate
        random_tensor = keep_prob + torch.rand(feature_map.size(0), 1, 1, 1, device=feature_map.device, dtype=feature_map.dtype) # (batch_size, 1, 1, 1)
        return feature_map / keep_prob * random_tensor.floor() # (batch_size, out_channels, H, W)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        out = feature_map # (batch_size, in_channels, H, W)
        if self.expand is not None:
            out = self.expand(out) # (batch_size, expanded_channels, H, W)
        out = self.depthwise(out) # (batch_size, expanded_channels, H / stride, W / stride)
        out = self.squeeze_excite(out) # (batch_size, expanded_channels, H / stride, W / stride)
        out = self.project(out) # (batch_size, out_channels, H / stride, W / stride)
        if self.use_residual:
            out = feature_map + self._drop_connect(out) # (batch_size, out_channels, H, W)
        return out
