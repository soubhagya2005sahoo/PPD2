"""ResNet-50 Baseline Architecture with BNNeck for Person Re-ID.

Architecture Components:
1. ResNet-50 Backbone: Extracts deep semantic visual feature representations.
   - Initialized with ImageNet-1k pretrained weights.
   - The final stride in layer4 is conventionally set to 1 in Person Re-ID to preserve
     spatial resolution (output feature map: 2048 x 16 x 8 instead of 2048 x 8 x 4).
2. Global Average Pooling (GAP): Condenses the 2D feature map into a 2048-dim embedding vector.
3. Batch Normalization Neck (BNNeck): Normalizes features before the linear classifier,
   enabling simultaneous convergence of Triplet Loss (metric space) and Cross-Entropy Loss.
4. Linear Identity Classifier: Predicts person identity across the training set classes.
"""

from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNet50Baseline(nn.Module):
    """ResNet-50 Person Re-Identification Baseline with BNNeck."""

    def __init__(
        self,
        num_classes: int = 590,
        pretrained: bool = True,
        last_stride: int = 1,
    ):
        """Initialize the model.

        Args:
            num_classes: Number of unique training person identities.
            pretrained: Whether to load ImageNet pretrained weights.
            last_stride: Stride of the final conv block (layer4). Setting to 1
                         preserves richer spatial granularity for pedestrian features.
        """
        super().__init__()
        self.num_classes = num_classes

        # 1. ResNet-50 Backbone
        if pretrained:
            weights = models.ResNet50_Weights.DEFAULT
            base = models.resnet50(weights=weights)
        else:
            base = models.resnet50(weights=None)

        # Optionally modify the last stride of layer4 to retain spatial detail
        if last_stride == 1:
            base.layer4[0].conv2.stride = (1, 1)
            base.layer4[0].downsample[0].stride = (1, 1)

        # Deconstruct ResNet-50 into feature extractor stages
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        # 2. Global Average Pooling (GAP)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # 3. BNNeck (Batch Normalization Neck)
        self.in_features = 2048
        self.bnneck = nn.BatchNorm1d(self.in_features)
        self.bnneck.bias.requires_grad_(False)  # Standard practice: do not shift center
        nn.init.constant_(self.bnneck.weight, 1.0)
        nn.init.constant_(self.bnneck.bias, 0.0)

        # 4. Identity Classification Layer
        self.classifier = nn.Linear(self.in_features, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def extract_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Extract intermediate 2D feature maps from the backbone."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.

        During Training:
            Returns:
                cls_score: (B, num_classes) classification logits for Cross-Entropy Loss.
                global_feat: (B, 2048) embedding vector for Triplet Loss.

        During Evaluation / Inference:
            Returns:
                norm_feat: (B, 2048) L2-normalized embedding vector for cosine ranking.
        """
        # Backbone forward
        feat_map = self.extract_feature_maps(x)

        # Global average pooling -> flatten to 2048-dim
        global_feat = self.gap(feat_map)
        global_feat = global_feat.view(global_feat.size(0), -1)

        # Pass through BNNeck
        bn_feat = self.bnneck(global_feat)

        if self.training:
            cls_score = self.classifier(bn_feat)
            return cls_score, global_feat
        else:
            # During evaluation, return L2-normalized feature representation
            return F.normalize(bn_feat, p=2, dim=1)

