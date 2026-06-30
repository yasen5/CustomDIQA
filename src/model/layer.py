import torch.nn as nn
import constants
import attention
import feedforward

class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_layernorm = nn.LayerNorm(constants.embedding_channels, eps=constants.layer_norm_epsilon)
        self.attention = attention.EncoderAttention()
        self.attention_layernorm = nn.LayerNorm(constants.embedding_channels, eps=constants.layer_norm_epsilon)
        self.feed_forward = feedforward.FeedForward()

    def forward(self, batched_inputs):
        normed_inputs = self.embedding_layernorm(batched_inputs) # (batch_size, num_patches, embedding_dims)
        out = batched_inputs + self.attention(normed_inputs) # (batch_size, num_patches, embedding_dims)
        out = out + self.feed_forward(self.attention_layernorm(out)) # (batch_size, num_patches, embedding_dims)
        return out
        
