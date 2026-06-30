import torch
import torch.nn as nn
import torch.utils.checkpoint
from . import constants
from . import attention
from . import feedforward

class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_layernorm = nn.LayerNorm(constants.embedding_channels, eps=constants.layer_norm_epsilon)
        self.attention = attention.EncoderAttention()
        self.attention_layernorm = nn.LayerNorm(constants.embedding_channels, eps=constants.layer_norm_epsilon)
        self.feed_forward = feedforward.FeedForward()

    def _forward(self, batched_inputs):
        normed_inputs = self.embedding_layernorm(batched_inputs)
        out = batched_inputs + self.attention(normed_inputs)
        out = out + self.feed_forward(self.attention_layernorm(out))
        return out

    def forward(self, batched_inputs):
        if self.training:
            return torch.utils.checkpoint.checkpoint(self._forward, batched_inputs, use_reentrant=False)
        return self._forward(batched_inputs)
        
