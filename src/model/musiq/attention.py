import torch.nn as nn

from . import constants


class MultiHeadAttention(nn.Module):
    """Standard multi-head self-attention, except the mask (marking padding patches — see
    patch_extraction.py) is applied as a pairwise row*col product rather than a simple broadcast,
    so padded patches never attend to, or are attended from."""

    def __init__(self):
        super().__init__()
        self.query_key_value = nn.Linear(constants.hidden_size, 3 * constants.hidden_size, bias=True)
        self.out_proj = nn.Linear(constants.hidden_size, constants.hidden_size)
        self.out_dropout = nn.Dropout(constants.dropout_rate)

    def forward(self, x, mask):
        # x: (batch_size, seq_len, hidden), mask: (batch_size, seq_len)
        batch_size, seq_len, hidden = x.shape

        qkv = self.query_key_value(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, constants.num_heads, constants.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv.unbind(0)

        mask_row = mask.reshape(batch_size, 1, seq_len, 1)
        mask_col = mask.reshape(batch_size, 1, 1, seq_len)
        attn_mask = mask_row * mask_col != 0

        out = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=constants.attention_dropout_rate if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).reshape(batch_size, seq_len, hidden)  # (batch_size, seq_len, hidden)
        out = self.out_proj(out)
        return self.out_dropout(out)
