import torch
import torch.nn as nn

from . import constants
from .embeddings import HashedSpatialPositionEmbedding, ScaleEmbedding
from .layer import TransformerBlock


class TransformerEncoder(nn.Module):
    """Adds spatial/scale position info and a CLS token to the tokenized patch sequence, then runs
    it through `num_layers` pre-norm transformer blocks. The CLS token's output embedding (read by
    model.py) is what the regression head scores."""

    def __init__(self):
        super().__init__()
        self.spatial_position_embedding = HashedSpatialPositionEmbedding()
        self.scale_embedding = ScaleEmbedding()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, constants.hidden_size))
        self.dropout = nn.Dropout(constants.dropout_rate)
        self.blocks = nn.ModuleList(TransformerBlock() for _ in range(constants.num_layers))
        self.final_layernorm = nn.LayerNorm(constants.hidden_size, eps=constants.layer_norm_eps)

    def forward(self, patch_embeddings, spatial_positions, scale_positions, patch_masks):
        # patch_embeddings: (batch, seq, hidden_size), patch_masks: (batch, seq)
        batch_size = patch_embeddings.shape[0]

        x = self.spatial_position_embedding(patch_embeddings, spatial_positions)
        x = self.scale_embedding(x, scale_positions)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, hidden_size)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, seq + 1, hidden_size)

        cls_mask = torch.ones((batch_size, 1), dtype=patch_masks.dtype, device=patch_masks.device)
        full_mask = torch.cat([cls_mask, patch_masks], dim=1)  # (batch, seq + 1)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x, full_mask)  # (batch, seq + 1, hidden_size)
        return self.final_layernorm(x)
