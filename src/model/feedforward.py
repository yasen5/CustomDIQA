import torch.nn as nn
import torch
from . import constants

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = constants.embedding_channels * constants.feedforward_hidden_dim_scalar
        self.dense1 = nn.Linear(constants.embedding_channels, self.hidden_dim)
        self.activation = QuickGELU()
        self.dense2 = nn.Linear(self.hidden_dim, constants.embedding_channels)
        self.dropout = nn.Dropout(constants.dropout)

    def forward(self, attended_features):
        out = self.dense1(attended_features) # (batch_size, num_patches, embedding_dims) @ (batch_size, embedding_dims, upscale)
        out = self.activation(out)
        out = self.dense2(out) # (batch_size, num_patches, upscale) @ (batch_size, upscale, embedding_dims)
        out = self.dropout(out)
        return out

