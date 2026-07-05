import torch.nn as nn

from . import constants


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(constants.hidden_size, constants.mlp_dim)
        self.activation = nn.GELU()
        self.dense2 = nn.Linear(constants.mlp_dim, constants.hidden_size)
        self.dropout = nn.Dropout(constants.dropout_rate)

    def forward(self, x):
        # x: (..., hidden_size)
        x = self.dense1(x)  # (..., mlp_dim)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.dense2(x)  # (..., hidden_size)
        x = self.dropout(x)
        return x
