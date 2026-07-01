import torch.nn as nn
import torch
from . import constants

class SqueezeExcite(nn.Module):
    def __init__(self, in_channels: int, expanded_channels: int):
        super().__init__()
        squeeze_channels = max(1, int(in_channels * constants.se_ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Conv2d(expanded_channels, squeeze_channels, kernel_size=1)
        self.activation = nn.SiLU(inplace=True)
        self.expand = nn.Conv2d(squeeze_channels, expanded_channels, kernel_size=1)
        self.gate = nn.Sigmoid()

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        squeezed = self.pool(feature_map) # (batch_size, expanded_channels, 1, 1)
        squeezed = self.activation(self.reduce(squeezed)) # (batch_size, squeeze_channels, 1, 1)
        squeezed = self.gate(self.expand(squeezed)) # (batch_size, expanded_channels, 1, 1)
        return feature_map * squeezed # (batch_size, expanded_channels, H, W)
