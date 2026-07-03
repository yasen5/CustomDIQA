import torch.nn as nn
import torch
from pyiqa.archs.topiq_arch import CFANet

from . import constants
from . import head


class HybridModel(nn.Module):
    """CNN+Transformer hybrid: pyiqa's CFANet (TOPIQ) backbone — a ResNet50 semantic
    feature extractor fused across scales by cross-attention transformer blocks — feeding
    our own 5-way head in place of CFANet's own 1-d regression head."""

    def __init__(self):
        super().__init__()
        self.backbone = CFANet(
            semantic_model_name="resnet50",
            model_name="cfanet_nr_koniq_res50",
            use_ref=False,
            inter_dim=constants.inter_dim,
            backbone_pretrain=False,
            pretrained=False,
        )
        self.backbone.score_linear = nn.Identity()
        self.head = head.MeanOpinionScoreHead()

    def forward(self, rgb_image: torch.FloatTensor) -> torch.Tensor:
        pooled = self.backbone.forward_cross_attention(rgb_image) # (batch_size, inter_dim)
        out = self.head(pooled) # (batch_size, 5)
        return out
