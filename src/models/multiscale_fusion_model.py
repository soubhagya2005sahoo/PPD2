"""End-to-End Multi-Scale Feature Fusion Architecture for Person Re-ID.

Integrates:
1. ResNet-50 Multi-Scale Backbone (layer1, layer2, layer4)
2. Spatial Global Average Pooling (GAP)
3. Multi-Scale Feature Projection into common dimension d
4. Multi-Scale Feature Fusion (Concatenation or Learned Weighted)
5. BNNeck for stable metric learning
6. Identity Classifier for Cross-Entropy Loss

Returns both the unified fused representation and all individual scale features.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet50_multiscale import ResNet50MultiScale
from ..features.feature_pooling import MultiScaleFeaturePooler
from .feature_projection import MultiScaleFeatureProjector
from .feature_fusion import MultiScaleFusion


class MultiScaleFusionModel(nn.Module):
    """Unified Multi-Scale Person Re-ID Model with Feature Fusion."""

    def __init__(
        self,
        num_classes: int = 590,
        fusion_strategy: str = "weighted",
        common_dim: int = 512,
        fused_dim: int = 512,
        pretrained: bool = True,
        checkpoint_path: Optional[str] = None,
        last_stride: int = 1,
    ):
        """Initialize the Multi-Scale Fusion Model.

        Args:
            num_classes: Number of unique training person identities.
            fusion_strategy: 'concat' or 'weighted'.
            common_dim: Common projection dimension for each scale (default: 512).
            fused_dim: Output dimension of unified fused feature (default: 512).
            pretrained: Whether to initialize ResNet-50 with pretrained weights.
            checkpoint_path: Path to existing checkpoint.
            last_stride: Stride of layer4 (1 preserves spatial detail).
        """
        super().__init__()
        self.num_classes = num_classes
        self.fusion_strategy = fusion_strategy
        self.common_dim = common_dim
        self.fused_dim = fused_dim

        # 1. Multi-scale feature extractor
        self.backbone = ResNet50MultiScale(
            pretrained=pretrained,
            checkpoint_path=checkpoint_path,
            last_stride=last_stride,
            medium_stage="layer2",
        )

        # 2. GAP Feature Pooler
        self.pooler = MultiScaleFeaturePooler(l2_normalize=False)

        # 3. Projection layers
        in_channels = {"fine": 256, "medium": 512, "coarse": 2048}
        self.projector = MultiScaleFeatureProjector(
            in_channels=in_channels,
            common_dim=common_dim,
            l2_normalize=False,
        )

        # 4. Multi-scale fusion module
        self.fusion = MultiScaleFusion(
            strategy=fusion_strategy,
            common_dim=common_dim,
            fused_dim=fused_dim,
        )

        # 5. BNNeck
        self.bnneck = nn.BatchNorm1d(fused_dim)
        self.bnneck.bias.requires_grad_(False)
        nn.init.constant_(self.bnneck.weight, 1.0)
        nn.init.constant_(self.bnneck.bias, 0.0)

        # 6. Classifier
        self.classifier = nn.Linear(fused_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, Dict]]:
        """Forward pass.

        During Training:
            Returns:
                cls_score: (B, num_classes) classification logits
                fused_feat: (B, fused_dim) unified feature for Triplet Loss
                aux_info: Dict containing original and projected scale representations and weights

        During Evaluation:
            Returns:
                norm_feat: (B, fused_dim) L2-normalized fused feature for retrieval
        """
        # 1. Extract 2D feature maps
        feat_maps = self.backbone(x)

        # 2. Global Average Pooling -> 1D vectors
        pooled_dict = self.pooler(feat_maps)

        # 3. Project each scale to common_dim
        proj_dict = self.projector(pooled_dict)

        # 4. Feature Fusion
        fused_feat, scale_weights = self.fusion(proj_dict)

        # 5. BNNeck
        bn_feat = self.bnneck(fused_feat)

        if self.training:
            cls_score = self.classifier(bn_feat)
            aux_info = {
                "original_vectors": pooled_dict,
                "projected_vectors": proj_dict,
                "scale_weights": scale_weights,
            }
            return cls_score, fused_feat, aux_info
        else:
            # Normalized fused representation for cosine retrieval
            return F.normalize(bn_feat, p=2, dim=1)

