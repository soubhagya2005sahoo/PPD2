"""Feature Dimension Alignment and Projection Module for Multi-Scale Re-ID.

Converts unaligned multi-scale pooled vectors:
- Fine Vector   : (B, 256)
- Medium Vector : (B, 512)
- Coarse Vector : (B, 2048)

into a shared common embedding dimension (default: d = 512):
- proj_fine   : (B, 512)
- proj_medium : (B, 512)
- proj_coarse : (B, 512)

Each scale utilizes an independent Linear -> BatchNorm1d -> ReLU projection block.
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleFeatureProjector(nn.Module):
    """Projects multi-scale vectors with varying dimensions into a shared subspace."""

    def __init__(
        self,
        in_channels: Dict[str, int] = None,
        common_dim: int = 512,
        l2_normalize: bool = False,
        use_bn: bool = True,
        use_relu: bool = True,
    ):
        """Initialize the projection module.

        Args:
            in_channels: Dict mapping scale names ('fine', 'medium', 'coarse') to input dimensions.
            common_dim: Shared projection dimension (default: 512).
            l2_normalize: Whether to apply L2 normalization to each projected vector.
            use_bn: Whether to include 1D Batch Normalization.
            use_relu: Whether to include non-linear ReLU activation.
        """
        super().__init__()
        if in_channels is None:
            in_channels = {"fine": 256, "medium": 512, "coarse": 2048}

        self.common_dim = common_dim
        self.l2_normalize = l2_normalize

        def make_proj_block(in_dim: int, out_dim: int) -> nn.Sequential:
            layers = [nn.Linear(in_dim, out_dim, bias=not use_bn)]
            if use_bn:
                layers.append(nn.BatchNorm1d(out_dim))
            if use_relu:
                layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        self.proj_fine = make_proj_block(in_channels["fine"], common_dim)
        self.proj_medium = make_proj_block(in_channels["medium"], common_dim)
        self.proj_coarse = make_proj_block(in_channels["coarse"], common_dim)

    def forward(self, vectors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Project each scale into the common embedding dimension.

        Args:
            vectors: Dict containing 'fine_vec', 'medium_vec', 'coarse_vec'.

        Returns:
            Dict containing:
            - 'proj_fine'   : (B, common_dim)
            - 'proj_medium' : (B, common_dim)
            - 'proj_coarse' : (B, common_dim)
        """
        p_fine = self.proj_fine(vectors["fine_vec"])
        p_med = self.proj_medium(vectors["medium_vec"])
        p_coarse = self.proj_coarse(vectors["coarse_vec"])

        if self.l2_normalize:
            p_fine = F.normalize(p_fine, p=2, dim=1)
            p_med = F.normalize(p_med, p=2, dim=1)
            p_coarse = F.normalize(p_coarse, p=2, dim=1)

        return {
            "proj_fine": p_fine,
            "proj_medium": p_med,
            "proj_coarse": p_coarse,
        }

