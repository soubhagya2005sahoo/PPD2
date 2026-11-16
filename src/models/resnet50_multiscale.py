"""Multi-Scale ResNet-50 Feature Extractor for Aerial-Ground Person Re-ID.

Extracts three distinct visual scale representations from intermediate CNN stages:
1. Fine-Scale Features (layer1):
   - Channels: 256
   - Spatial Size: 64 x 32 (for 256 x 128 input)
   - Semantics: Edges, clothing texture, color patterns, localized details.
2. Medium-Scale Features (layer2 or layer3):
   - Channels: 512 (layer2) or 1024 (layer3)
   - Spatial Size: 32 x 16 (layer2) or 16 x 8 (layer3)
   - Semantics: Regional body anatomy (torso, limbs, accessories).
3. Coarse-Scale Features (layer4):
   - Channels: 2048
   - Spatial Size: 16 x 8
   - Semantics: Global human posture, holistic viewpoint, high-level abstraction.

All three scales remain separate (no fusion in Step 5).
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50MultiScale(nn.Module):
    """Multi-Scale Feature Extractor leveraging intermediate ResNet-50 stages."""

    def __init__(
        self,
        pretrained: bool = True,
        checkpoint_path: Optional[str] = None,
        last_stride: int = 1,
        medium_stage: str = "layer2",
    ):
        """Initialize the multi-scale extractor.

        Args:
            pretrained: Whether to load ImageNet pretrained weights if checkpoint_path is None.
            checkpoint_path: Optional path to a trained person Re-ID model checkpoint.
            last_stride: Stride of the final conv block (layer4). Setting to 1
                         preserves 16x8 spatial resolution instead of 8x4.
            medium_stage: Which stage to use for medium scale ('layer2' or 'layer3').
        """
        super().__init__()
        self.medium_stage = medium_stage.lower()
        if self.medium_stage not in ["layer2", "layer3"]:
            raise ValueError(f"Invalid medium_stage '{medium_stage}'. Choose 'layer2' or 'layer3'.")

        # Load ResNet-50 base
        if pretrained and checkpoint_path is None:
            base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            base = models.resnet50(weights=None)

        if last_stride == 1:
            base.layer4[0].conv2.stride = (1, 1)
            base.layer4[0].downsample[0].stride = (1, 1)

        # Deconstruct layers
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        self.layer1 = base.layer1  # Fine-scale stage (256 channels)
        self.layer2 = base.layer2  # Medium stage option A (512 channels)
        self.layer3 = base.layer3  # Medium stage option B (1024 channels)
        self.layer4 = base.layer4  # Coarse-scale stage (2048 channels)

        # Load custom weights if provided
        if checkpoint_path is not None:
            self.load_custom_checkpoint(checkpoint_path)

        # Channel metadata
        self.channels = {
            "fine": 256,
            "medium": 512 if self.medium_stage == "layer2" else 1024,
            "coarse": 2048,
        }

    def load_custom_checkpoint(self, path: str) -> None:
        """Load weights from a trained Step 3 checkpoint."""
        ckpt = torch.load(path, map_location="cpu")
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        # Filter matching keys
        model_dict = self.state_dict()
        filtered = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(filtered)
        self.load_state_dict(model_dict)
        print(f"Loaded {len(filtered)} matching weight tensors from checkpoint: {path}")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract multi-scale 2D feature maps from the network.

        Args:
            x: Input image tensor (B, 3, H, W).

        Returns:
            Dictionary containing 3 separate feature maps:
            - 'fine': (B, 256, H/4, W/4)
            - 'medium': (B, 512 or 1024, H/8 or H/16, W/8 or W/16)
            - 'coarse': (B, 2048, H/16, W/16)
        """
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Stage 1: Fine-Scale
        feat_fine = self.layer1(x)

        # Stage 2: Intermediate
        feat_layer2 = self.layer2(feat_fine)

        # Stage 3: Higher Intermediate
        feat_layer3 = self.layer3(feat_layer2)

        # Stage 4: Coarse-Scale
        feat_coarse = self.layer4(feat_layer3)

        feat_medium = feat_layer2 if self.medium_stage == "layer2" else feat_layer3

        return {
            "fine": feat_fine,
            "medium": feat_medium,
            "coarse": feat_coarse,
        }

