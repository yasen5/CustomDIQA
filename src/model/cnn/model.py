import torch.nn as nn
import torch
from . import constants
from . import head
from .mbconv import MBConvBlock

class EfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        stem_channels = constants.round_filters(constants.stem_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels, momentum=constants.batch_norm_momentum, eps=constants.batch_norm_epsilon),
            nn.SiLU(inplace=True),
        )

        stage_settings = constants.scaled_stage_settings()
        total_blocks = sum(num_repeats for _, _, num_repeats, _, _ in stage_settings)

        blocks = []
        in_channels = stem_channels
        block_index = 0
        for expand_ratio, out_channels, num_repeats, stride, kernel_size in stage_settings:
            for repeat_index in range(num_repeats):
                drop_connect_rate = constants.drop_connect_rate * block_index / total_blocks
                blocks.append(MBConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    expand_ratio=expand_ratio,
                    stride=stride if repeat_index == 0 else 1,
                    kernel_size=kernel_size,
                    drop_connect_rate=drop_connect_rate,
                ))
                in_channels = out_channels
                block_index += 1
        self.blocks = nn.Sequential(*blocks)

        head_channels = constants.round_filters(constants.head_channels)
        self.head_conv = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(head_channels, momentum=constants.batch_norm_momentum, eps=constants.batch_norm_epsilon),
            nn.SiLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = head.MeanOpinionScoreHead()

        self.apply(self._init_weights)

    def forward(self, rgb_image: torch.FloatTensor) -> torch.Tensor:
        out = self.stem(rgb_image) # (batch_size, stem_channels, img_size / 2, img_size / 2)
        out = self.blocks(out) # (batch_size, last_stage_channels, img_size / 32, img_size / 32)
        out = self.head_conv(out) # (batch_size, head_channels, img_size / 32, img_size / 32)
        out = self.pool(out).flatten(1) # (batch_size, head_channels)
        out = self.head(out) # (batch_size, 5)
        return out

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
