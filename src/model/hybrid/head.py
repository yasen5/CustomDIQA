import torch.nn as nn
import torch
from . import constants

class MeanOpinionScoreHead(nn.Module):
    def __init__(self):
        super().__init__()
        intermediate_dim = constants.inter_dim // constants.abs_value_head_scaledown
        self.dropout = nn.Dropout(constants.dropout)
        self.dense1 = nn.Linear(constants.inter_dim, intermediate_dim)
        self.activation = nn.GELU()
        self.dense2 = nn.Linear(intermediate_dim, 5)

    def forward(self, pooled_features: torch.Tensor) -> torch.Tensor:
        out = self.dropout(pooled_features) # (batch_size, inter_dim)
        out = self.dense1(out) # (batch_size, intermediate_dim)
        out = self.activation(out) # (batch_size, intermediate_dim)
        out = self.dense2(out) # (batch_size, 5)
        return out
