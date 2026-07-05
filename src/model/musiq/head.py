import torch.nn as nn
import torch
from . import constants


class MeanOpinionScoreHead(nn.Module):
    def __init__(self):
        super().__init__()
        intermediate_dim = constants.hidden_size // constants.abs_value_head_scaledown
        self.dense1 = nn.Linear(constants.hidden_size, intermediate_dim)
        self.activation = nn.GELU()
        self.dense2 = nn.Linear(intermediate_dim, 5)

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        out = self.dense1(cls_token)  # (batch_size, intermediate_dim)
        out = self.activation(out)
        out = self.dense2(out)  # (batch_size, 5)
        return out
