import torch.nn as nn
import constants
import feedforward

class MeanOpinionScoreHead(nn.Module):
    def __init__(self):
        super().__init__()
        intermediate_dim = constants.embedding_channels // constants.abs_value_head_scaledown
        self.dense1 = nn.Linear(constants.embedding_channels, intermediate_dim)
        self.activation = feedforward.QuickGELU()
        self.dense2 = nn.Linear(intermediate_dim, 5)

    def forward(self, patch_mean):
        out = self.dense1(patch_mean)
        out = self.activation(out)
        out = self.dense2(out)
        return out

