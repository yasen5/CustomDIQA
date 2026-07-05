import torch.nn as nn

from . import constants
from . import patch_extraction
from . import head
from .tokenizer import PatchTokenizer
from .encoder import TransformerEncoder


class MUSIQModel(nn.Module):
    """MUSIQ (Multi-scale Image Quality Transformer): a per-patch ResNet tokenizer feeding a
    transformer encoder over a multi-scale patch sequence, scored from its CLS token via our own
    5-way MeanOpinionScoreHead (in place of pyiqa's 1-d regression head).

    Only implements MUSIQ's eval-mode forward (raw RGB image in, scalar score out) — pyiqa's
    version also accepts pre-extracted patch tensors directly for training, which this repo has no
    use for. `rgb_image` values are expected in [0, 1], matching pyiqa's convention (this model
    rescales to [-1, 1] internally, as MUSIQ's original TF implementation does).
    """

    def __init__(self):
        super().__init__()
        self.tokenizer = PatchTokenizer()
        self.encoder = TransformerEncoder()
        self.head = head.MeanOpinionScoreHead()

    def forward(self, rgb_image):
        x = (rgb_image - constants.rgb_rescale_mean) * constants.rgb_rescale_scale

        x = patch_extraction.get_multiscale_patches(
            x,
            patch_size=constants.patch_size,
            patch_stride=constants.patch_stride,
            hse_grid_size=constants.spatial_pos_grid_size,
            longer_side_lengths=constants.longer_side_lengths,
            max_seq_len_from_original_res=constants.max_seq_len_from_original_res,
        )  # (batch_size, seq_len, 3 * patch_size**2 + 3)

        batch_size, seq_len, _ = x.shape
        spatial_positions = x[:, :, -3]  # (batch_size, seq_len)
        scale_positions = x[:, :, -2]  # (batch_size, seq_len)
        patch_masks = x[:, :, -1].bool()  # (batch_size, seq_len)
        patches = x[:, :, :-3]  # (batch_size, seq_len, 3 * patch_size**2)

        patches = patches.reshape(batch_size * seq_len, 3, constants.patch_size, constants.patch_size)
        patch_embeddings = self.tokenizer(patches)  # (batch_size * seq_len, hidden_size)
        patch_embeddings = patch_embeddings.reshape(batch_size, seq_len, constants.hidden_size)

        encoded = self.encoder(patch_embeddings, spatial_positions, scale_positions, patch_masks)
        return self.head(encoded[:, 0])  # (batch_size, 5), scored from the CLS token
