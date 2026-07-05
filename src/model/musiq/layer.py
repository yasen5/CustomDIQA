import torch.nn as nn

from . import constants
from .attention import MultiHeadAttention
from .feedforward import FeedForward


class TransformerBlock(nn.Module):
    """Pre-norm transformer block. MUSIQ's original also supports stochastic-depth drop_path, but
    it's only ever instantiated with drop_path=0 (TransformerEncoder never passes a nonzero rate),
    so that's omitted here rather than kept as dead config."""

    def __init__(self):
        super().__init__()
        self.attention_layernorm = nn.LayerNorm(constants.hidden_size, eps=constants.layer_norm_eps)
        self.attention = MultiHeadAttention()
        self.feedforward_layernorm = nn.LayerNorm(constants.hidden_size, eps=constants.layer_norm_eps)
        self.feed_forward = FeedForward()

    def forward(self, x, mask):
        x = x + self.attention(self.attention_layernorm(x), mask)
        x = x + self.feed_forward(self.feedforward_layernorm(x))
        return x
