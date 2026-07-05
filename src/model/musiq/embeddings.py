import torch
import torch.nn as nn

from . import constants


class HashedSpatialPositionEmbedding(nn.Module):
    """Learned position embedding indexed by each patch's hashed (row, col) grid cell — MUSIQ
    supports variable input resolutions (different patch counts per image), so it can't use a fixed
    per-index position table the way a standard ViT does; instead every patch's row/col is hashed
    down onto a fixed `spatial_pos_grid_size` x `spatial_pos_grid_size` grid (see
    patch_extraction.py) and that grid cell's embedding is looked up here."""

    def __init__(self):
        super().__init__()
        # initialized with an extra dimension to be compatible with MUSIQ
        self.position_emb = nn.Parameter(
            torch.randn(1, constants.spatial_pos_grid_size ** 2, constants.hidden_size)
        )
        nn.init.normal_(self.position_emb, std=constants.embedding_init_std)

    def forward(self, patch_embeddings, spatial_positions):
        # patch_embeddings: (batch, seq, hidden_size), spatial_positions: (batch, seq)
        return patch_embeddings + self.position_emb.squeeze(0)[spatial_positions.long()]  # (batch, seq, hidden_size)


class ScaleEmbedding(nn.Module):
    """Learned embedding indexed by which multi-scale representation (see patch_extraction.py) a
    patch came from — one of the resized scales, or the native-resolution scale."""

    def __init__(self):
        super().__init__()
        self.scale_emb = nn.Parameter(
            torch.randn(constants.num_scales, constants.hidden_size) * constants.embedding_init_std
        )

    def forward(self, patch_embeddings, scale_positions):
        # patch_embeddings: (batch, seq, hidden_size), scale_positions: (batch, seq)
        return patch_embeddings + self.scale_emb[scale_positions.long()]  # (batch, seq, hidden_size)
