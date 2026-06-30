import torch.nn as nn
import torch
from . import constants

class MeanOpinionScoreHead(nn.Module):
    def __init__(self):
        super().__init__()
        intermediate_dim = constants.head_channels // constants.abs_value_head_scaledown
        self.dropout = nn.Dropout(constants.dropout)
        self.dense1 = nn.Linear(constants.head_channels, intermediate_dim)
        self.activation = nn.SiLU(inplace=True)
        self.dense2 = nn.Linear(intermediate_dim, 5)

    def forward(self, pooled_features: torch.Tensor) -> torch.Tensor:
        out = self.dropout(pooled_features)
        out = self.dense1(out)
        out = self.activation(out)
        out = self.dense2(out)
        return out
