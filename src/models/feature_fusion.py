"""Multi-Scale Feature Fusion Strategies for Person Re-ID.

Implements two complementary fusion methodologies:
1. Concatenation-Based Fusion:
   - Concatenates projected vectors: [p_fine, p_medium, p_coarse] -> 3 * d
   - Reduces dimension via a fusion projection layer -> d_fused
2. Learned Weighted Fusion:
   - Learns three trainable importance weights w = [w_fine, w_medium, w_coarse]
   - Applies Softmax to guarantee non-negativity and unit sum:
     alpha = Softmax(w) = [alpha_fine, alpha_medium, alpha_coarse], sum(alpha) = 1.0
   - Computes convex combination:
     fused = alpha_fine * p_fine + alpha_medium * p_medium + alpha_coarse * p_coarse
"""

from typing import Dict, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatFusion(nn.Module):
    """Concatenation-based multi-scale feature fusion."""

    def __init__(self, common_dim: int = 512, fused_dim: int = 512):
        """Initialize Concatenation Fusion.

        Args:
            common_dim: Dimension of each input projected feature (d).
            fused_dim: Target dimension for the unified output representation.
        """
        super().__init__()
        self.common_dim = common_dim
        self.fused_dim = fused_dim

        # Fuses 3 * common_dim down to fused_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(3 * common_dim, fused_dim, bias=False),
            nn.BatchNorm1d(fused_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, proj_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Concatenate and project features.

        Returns:
            fused_feature: (B, fused_dim)
        """
        cat_features = torch.cat([
            proj_dict["proj_fine"],
            proj_dict["proj_medium"],
            proj_dict["proj_coarse"],
        ], dim=1)

        fused = self.fusion_layer(cat_features)
        return fused


class LearnedWeightedFusion(nn.Module):
    """Learned weighted multi-scale feature fusion via Softmax gating."""

    def __init__(self, common_dim: int = 512, fused_dim: int = 512):
        """Initialize Learned Weighted Fusion.

        Args:
            common_dim: Dimension of each input projected feature (d).
            fused_dim: Target dimension for the unified output representation.
        """
        super().__init__()
        self.common_dim = common_dim
        self.fused_dim = fused_dim

        # 3 learnable scalar weights initialized equally to 0.0 (uniform 1/3 after softmax)
        self.scale_weights = nn.Parameter(torch.zeros(3))

        # Optional linear transformation if fused_dim differs from common_dim
        if fused_dim != common_dim:
            self.proj_out = nn.Sequential(
                nn.Linear(common_dim, fused_dim, bias=False),
                nn.BatchNorm1d(fused_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.proj_out = nn.Identity()

    def get_normalized_weights(self) -> torch.Tensor:
        """Return the normalized softmax weights [alpha_fine, alpha_med, alpha_coarse]."""
        return F.softmax(self.scale_weights, dim=0)

    def forward(self, proj_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute convex combination of projected features.

        Returns:
            fused_feature: (B, fused_dim)
            normalized_weights: (3,) tensor summing to 1.0
        """
        weights = self.get_normalized_weights()

        fused = (
            weights[0] * proj_dict["proj_fine"]
            + weights[1] * proj_dict["proj_medium"]
            + weights[2] * proj_dict["proj_coarse"]
        )

        fused = self.proj_out(fused)
        return fused, weights


class MultiScaleFusion(nn.Module):
    """Unified interface supporting both concatenation and learned weighted fusion."""

    def __init__(self, strategy: str = "weighted", common_dim: int = 512, fused_dim: int = 512):
        super().__init__()
        self.strategy = strategy.lower()
        if self.strategy in ["concat", "concatenation"]:
            self.fusion_module = ConcatFusion(common_dim=common_dim, fused_dim=fused_dim)
        elif self.strategy in ["weighted", "learned_weighted"]:
            self.fusion_module = LearnedWeightedFusion(common_dim=common_dim, fused_dim=fused_dim)
        else:
            raise ValueError(f"Unknown fusion strategy '{strategy}'. Choose 'concat' or 'weighted'.")

    def forward(self, proj_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """Execute the configured fusion strategy.

        Returns:
            fused_feature: (B, fused_dim)
            scale_weights: (3,) tensor if weighted fusion, else None
        """
        if self.strategy in ["concat", "concatenation"]:
            fused = self.fusion_module(proj_dict)
            return fused, None
        else:
            fused, weights = self.fusion_module(proj_dict)
            return fused, weights

