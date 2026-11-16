"""Feature Pooling Module for Multi-Scale Representations.

Applies Global Average Pooling (GAP) across spatial dimensions for
fine, medium, and coarse feature maps independently.

Converts:
- Fine map   (B, 256, H1, W1)  -> Fine vector   (B, 256)
- Medium map (B, 512, H2, W2)  -> Medium vector (B, 512)
- Coarse map (B, 2048, H3, W3) -> Coarse vector (B, 2048)

Strictly preserves individual representation boundaries (no fusion in Step 5).
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleFeaturePooler(nn.Module):
    """Pools 2D multi-scale feature maps into 1D embedding vectors."""

    def __init__(self, l2_normalize: bool = False):
        """Initialize the feature pooler.

        Args:
            l2_normalize: Whether to apply L2 normalization to each pooled vector individually.
        """
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.l2_normalize = l2_normalize

    def forward(self, feature_maps: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Pool each scale's feature map into a 1D vector.

        Args:
            feature_maps: Dictionary containing 'fine', 'medium', and 'coarse' tensors.

        Returns:
            Dictionary containing 'fine_vec', 'medium_vec', 'coarse_vec'.
        """
        pooled_vectors = {}

        for scale_name in ["fine", "medium", "coarse"]:
            feat_map = feature_maps[scale_name]
            # Spatial average pooling: (B, C, H, W) -> (B, C, 1, 1) -> (B, C)
            vec = self.gap(feat_map)
            vec = vec.view(vec.size(0), -1)

            if self.l2_normalize:
                vec = F.normalize(vec, p=2, dim=1)

            pooled_vectors[f"{scale_name}_vec"] = vec

        return pooled_vectors

