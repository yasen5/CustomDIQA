import torch.nn as nn
import torch
import constants

class EncoderAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_key_value = nn.Linear(constants.embedding_channels, 3 * constants.embedding_channels)
        self.dense = nn.Linear(constants.embedding_channels, constants.embedding_channels)
    
    def forward(self, patch_embeddings: torch.FloatTensor) -> torch.FloatTensor:
        batch_size, _, _ = patch_embeddings.size()
        query_key_value_mats = self.query_key_value(patch_embeddings) # (batch_size, num_patches, 3 * embedding_dims)
        query_key_value_mats = query_key_value_mats.view(batch_size, constants.num_patches, 3, constants.num_heads, constants.attention_head_dim)
        query_key_value_mats = query_key_value_mats.permute(2, 0, 3, 1, 4) # (3, batch_size, num_heads, num_patches, head_dim)
        queries, keys, value_mat = query_key_value_mats.unbind(0) # (batch-size, num_heads, num_patches, head_dim)
        attention_weighted_adjustments = nn.functional.scaled_dot_product_attention(queries, keys, value_mat, attn_mask=None, dropout_p=constants.dropout if self.training else 0, is_causal=False) # (batch_size, num_heads, num_patches, head_dims)
        attention_weighted_adjustments = attention_weighted_adjustments.transpose(1, 2).view(batch_size, constants.num_patches, -1)
        mixed_attention = self.dense(attention_weighted_adjustments) # (batch_size, num_patches, embedding_dims) @ (batch_size, embedding_dims, embedding_dims) = no change
        return mixed_attention

