import torch.nn as nn
import torch
from . import constants
from . import layer
from . import head

class EncoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Conv2d(in_channels=3, out_channels=constants.embedding_channels, kernel_size=constants.patch_size, stride=constants.patch_size)
        self.layers = nn.Sequential(*(layer.EncoderLayer() for _ in range(constants.num_vit_layers)))
        self.final_encoder_layernorm = nn.LayerNorm(constants.embedding_channels, eps=constants.layer_norm_epsilon)
        self.head = head.MeanOpinionScoreHead()
        self.apply(self._init_weights)

    def forward(self, rgb_image: torch.FloatTensor) -> torch.Tensor:
        embeddings = self.embedding(rgb_image) # (batch_size, out_ch, 3, 14, 14) over (batch_size, 3, 488, 488) = (batch_size, out_ch, 32, 32)
        embeddings = embeddings.flatten(2) # (batch_size, out_ch, num_patches)
        embeddings = embeddings.transpose(1, 2) # (batch_size, num_patches, out_ch)
        patch_outputs = self.layers(embeddings) # (batch_size, num_patches, out_ch)
        patch_outputs = self.final_encoder_layernorm(patch_outputs)
        patch_mean = patch_outputs.mean(dim=1)
        out = self.head(patch_mean)
        return out

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
