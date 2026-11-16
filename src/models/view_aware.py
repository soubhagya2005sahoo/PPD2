"""View-Aware Conditioning and Scale Attention Module for Aerial-Ground Re-ID.

Handles the viewpoint domain gap between:
- View 0 (C0): Aerial UAV Drone (top-down, small scale, atmospheric haze)
- View 1 (C2): Wearable Smart Glasses (egocentric ground level)
- View 2 (C3): Ground CCTV Surveillance (elevated static ground level)

Components:
1. Camera ID -> View Index mapping + learnable View Embedding table.
2. Auxiliary View Classifier (enables view prediction even if cam_id is absent).
3. View-Aware Dynamic Scale Attention (adapts fine/medium/coarse fusion weights
   based on both visual content and camera viewpoint).
4. Zero-initialized FiLM View Modulation (normalizes viewpoint-specific style shifts).
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# Mapping from AG-ReID.v2 raw camera IDs {0, 2, 3} to contiguous view indices {0, 1, 2}
CAM_TO_VIEW = {0: 0, 2: 1, 3: 2}
VIEW_NAMES = {0: "Aerial_UAV (C0)", 1: "Wearable (C2)", 2: "CCTV (C3)"}


def map_cam_to_view(cam_ids: torch.Tensor) -> torch.Tensor:
    """Map raw camera IDs (0, 2, 3) to contiguous view indices (0, 1, 2)."""
    view_ids = torch.zeros_like(cam_ids, dtype=torch.long)
    view_ids = torch.where(cam_ids == 2, torch.ones_like(view_ids), view_ids)
    view_ids = torch.where(cam_ids == 3, torch.full_like(view_ids, 2), view_ids)
    return view_ids


class ViewAwareModule(nn.Module):
    """View-Aware Scale Attention and Feature Modulation for Cross-View Re-ID."""

    def __init__(
        self,
        common_dim: int = 512,
        num_views: int = 3,
        view_dim: int = 128,
    ):
        super().__init__()
        self.common_dim = common_dim
        self.num_views = num_views
        self.view_dim = view_dim

        # 1. Learnable View Embeddings
        self.view_embedding = nn.Embedding(num_views, view_dim)
        nn.init.normal_(self.view_embedding.weight, std=0.02)

        # 2. Auxiliary View Classifier (from coarse semantics)
        self.view_classifier = nn.Sequential(
            nn.Linear(common_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_views),
        )

        # 3. View-Aware Dynamic Scale Attention Gate
        # Inputs: concatenated [proj_coarse (d), view_emb (view_dim)] -> 3 scale logits
        self.scale_gate = nn.Sequential(
            nn.Linear(common_dim + view_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
        )
        # Global learnable base prior so weights start balanced (1/3, 1/3, 1/3)
        self.base_scale_logits = nn.Parameter(torch.zeros(3))

        # 4. View-Conditional FiLM Normalization (Residual channel calibration)
        self.film_generator = nn.Linear(view_dim, 2 * common_dim)
        nn.init.zeros_(self.film_generator.weight)
        nn.init.zeros_(self.film_generator.bias)

    def forward(
        self,
        proj_dict: Dict[str, torch.Tensor],
        cam_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute view-aware fused features and view classification logits.

        Args:
            proj_dict: Dict with 'proj_fine', 'proj_medium', 'proj_coarse' of shape (B, d).
            cam_ids: Optional tensor of raw camera IDs (B,) with values in {0, 2, 3}.

        Returns:
            fused_feat: (B, d) view-aware fused & calibrated feature tensor.
            scale_weights: (B, 3) per-sample dynamic scale weights [fine, med, coarse].
            view_logits: (B, num_views) predicted view logits for auxiliary loss.
            view_ids: (B,) contiguous view indices used.
        """
        p_fine = proj_dict["proj_fine"]
        p_med = proj_dict["proj_medium"]
        p_coarse = proj_dict["proj_coarse"]

        # Predict view logits from coarse visual features
        view_logits = self.view_classifier(p_coarse)

        if cam_ids is not None:
            view_ids = map_cam_to_view(cam_ids.to(p_coarse.device))
        else:
            view_ids = torch.argmax(view_logits, dim=1)

        view_emb = self.view_embedding(view_ids)  # (B, view_dim)

        # Compute per-sample view-aware scale weights
        gate_input = torch.cat([p_coarse, view_emb], dim=1)
        dynamic_logits = self.scale_gate(gate_input) + self.base_scale_logits.unsqueeze(0)
        scale_weights = F.softmax(dynamic_logits, dim=1)  # (B, 3)

        # Weighted fusion per sample
        fused = (
            scale_weights[:, 0:1] * p_fine
            + scale_weights[:, 1:2] * p_med
            + scale_weights[:, 2:3] * p_coarse
        )

        # Apply residual FiLM view calibration: (1 + gamma) * fused + beta
        film_params = self.film_generator(view_emb)
        delta_gamma, beta = torch.chunk(film_params, 2, dim=1)
        calibrated_fused = (1.0 + delta_gamma) * fused + beta

        return calibrated_fused, scale_weights, view_logits, view_ids

