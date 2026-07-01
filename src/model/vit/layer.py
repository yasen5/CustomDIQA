import torch
import torch.nn as nn
import torch.utils.checkpoint
from . import constants
from . import attention
from . import feedforward

class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_layernorm = nn.LayerNorm(constants.embedding_channels, eps=constants.layer_norm_epsilon)
        self.attention = attention.EncoderAttention()
        self.attention_layernorm = nn.LayerNorm(constants.embedding_channels, eps=constants.layer_norm_epsilon)
        self.feed_forward = feedforward.FeedForward()
        self.layerscale1 = LayerScale(constants.embedding_channels)
        self.layerscale2 = LayerScale(constants.embedding_channels)

    def _forward(self, batched_inputs):
        normed_inputs = self.embedding_layernorm(batched_inputs)
        out = batched_inputs + self.layerscale1(self.attention(normed_inputs))
        out = out + self.layerscale2(self.feed_forward(self.attention_layernorm(out)))
        return out

    def forward(self, batched_inputs):
        if self.training:
            return torch.utils.checkpoint.checkpoint(self._forward, batched_inputs, use_reentrant=False)
        return self._forward(batched_inputs)
        
