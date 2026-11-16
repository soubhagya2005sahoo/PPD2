"""Unified View-Aware Multi-Scale Feature Fusion & Latent Diffusion Re-ID Model.

Integrates all proposed components from the research methodology:
1. ResNet-50 Multi-Scale Backbone (fine=layer1, medium=layer2, coarse=layer4, last_stride=1)
2. Global Average Pooling (GAP)
3. Multi-Scale Projection into common dimension d=512
4. View-Aware Conditioning & Dynamic Scale Attention (adapts scale weights per camera view)
5. Efficient Latent Feature Diffusion Refinement (2-step compact 1D latent denoising)
6. BNNeck + Identity Classification + Batch-Hard Triplet Learning
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet50_multiscale import ResNet50MultiScale
from ..features.feature_pooling import MultiScaleFeaturePooler
from .feature_projection import MultiScaleFeatureProjector
from .feature_fusion import MultiScaleFusion
from .view_aware import ViewAwareModule, map_cam_to_view
from .latent_diffusion import LatentFeatureDiffusion


class ViewAwareMultiScaleReIDModel(nn.Module):
    """Full Proposed Framework: View-Aware Multi-Scale Fusion + Latent Diffusion."""

    def __init__(
        self,
        num_classes: int = 590,
        common_dim: int = 512,
        fused_dim: int = 512,
        pretrained: bool = True,
        checkpoint_path: Optional[str] = None,
        last_stride: int = 1,
        use_view_aware: bool = True,
        use_diffusion: bool = True,
        fusion_strategy: str = "weighted",
        num_views: int = 3,
    ):
        """Initialize the View-Aware Multi-Scale Re-ID Model."""
        super().__init__()
        self.num_classes = num_classes
        self.common_dim = common_dim
        self.fused_dim = fused_dim
        self.use_view_aware = use_view_aware
        self.use_diffusion = use_diffusion

        # 1. Multi-Scale Backbone
        self.backbone = ResNet50MultiScale(
            pretrained=pretrained,
            checkpoint_path=checkpoint_path,
            last_stride=last_stride,
            medium_stage="layer2",
        )

        # 2. GAP Feature Pooler
        self.pooler = MultiScaleFeaturePooler(l2_normalize=False)

        # 3. Projection Layers (256, 512, 2048 -> common_dim)
        in_channels = {"fine": 256, "medium": 512, "coarse": 2048}
        self.projector = MultiScaleFeatureProjector(
            in_channels=in_channels,
            common_dim=common_dim,
            l2_normalize=False,
        )

        # 4. View-Aware Fusion OR Standard Fusion (for clean ablation)
        if self.use_view_aware:
            self.view_aware = ViewAwareModule(
                common_dim=common_dim,
                num_views=num_views,
                view_dim=128,
            )
            self.fusion = None
        else:
            self.view_aware = None
            self.fusion = MultiScaleFusion(
                strategy=fusion_strategy,
                common_dim=common_dim,
                fused_dim=fused_dim,
            )

        # 5. Optional Efficient Latent Feature Diffusion Refinement
        if self.use_diffusion:
            self.latent_diffusion = LatentFeatureDiffusion(
                feat_dim=fused_dim,
                num_views=num_views,
                view_dim=128,
                time_dim=64,
                num_timesteps=100,
                inference_steps=2,
                refinement_alpha=0.25,
            )
        else:
            self.latent_diffusion = None

        # 6. BNNeck
        self.bnneck = nn.BatchNorm1d(fused_dim)
        self.bnneck.bias.requires_grad_(False)
        nn.init.constant_(self.bnneck.weight, 1.0)
        nn.init.constant_(self.bnneck.bias, 0.0)

        # 7. Identity Classifier
        self.classifier = nn.Linear(fused_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(
        self,
        x: torch.Tensor,
        cam_ids: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, Dict]]:
        """Forward pass supporting optional camera viewpoint IDs.

        During Training:
            Returns (cls_score, refined_feat, aux_info) where aux_info contains:
            - scale_weights: (B, 3) or (3,)
            - view_logits: (B, 3) or None
            - view_ids: (B,) or None
            - diff_loss: scalar tensor
        During Evaluation:
            Returns L2-normalized embedding of shape (B, fused_dim).
        """
        feat_maps = self.backbone(x)
        pooled_dict = self.pooler(feat_maps)
        proj_dict = self.projector(pooled_dict)

        view_logits = None
        view_ids = None

        if self.use_view_aware and self.view_aware is not None:
            fused_feat, scale_weights, view_logits, view_ids = self.view_aware(proj_dict, cam_ids=cam_ids)
        else:
            fused_feat, scale_weights = self.fusion(proj_dict)
            if cam_ids is not None:
                view_ids = map_cam_to_view(cam_ids.to(x.device))

        diff_loss = torch.tensor(0.0, device=x.device)
        if self.use_diffusion and self.latent_diffusion is not None:
            refined_feat, diff_loss = self.latent_diffusion(fused_feat, view_ids=view_ids)
        else:
            refined_feat = fused_feat

        bn_feat = self.bnneck(refined_feat)

        if self.training:
            cls_score = self.classifier(bn_feat)
            aux_info = {
                "scale_weights": scale_weights,
                "view_logits": view_logits,
                "view_ids": view_ids,
                "diff_loss": diff_loss,
            }
            return cls_score, refined_feat, aux_info
        else:
            return F.normalize(bn_feat, p=2, dim=1)

