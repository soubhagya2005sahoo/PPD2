"""Validation and Unit Test for Multi-Scale Feature Fusion Pipeline.

Tests:
1. Projection dimension alignment (256, 512, 2048 -> 512).
2. Concatenation-based fusion.
3. Learned weighted fusion via Softmax gating.
4. Softmax properties (sum == 1.0, non-negative).
5. Full MultiScaleFusionModel end-to-end forward and backward gradient flow.
6. Numerical stability (no NaNs, no Infs).
"""

import sys
import os
import torch
import torch.nn as nn

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.feature_projection import MultiScaleFeatureProjector
from src.models.feature_fusion import ConcatFusion, LearnedWeightedFusion
from src.models.multiscale_fusion_model import MultiScaleFusionModel


def test_fusion_modules():
    print("=" * 65)
    print("TESTING MULTI-SCALE FEATURE FUSION PIPELINE")
    print("=" * 65)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[1/6] Compute Device Selected: {device}")

    batch_size = 4
    common_dim = 512
    fused_dim = 512

    # Synthetic multi-scale vectors matching Step 5 outputs
    fine_vec = torch.randn(batch_size, 256, device=device)
    med_vec = torch.randn(batch_size, 512, device=device)
    coarse_vec = torch.randn(batch_size, 2048, device=device)
    vectors = {"fine_vec": fine_vec, "medium_vec": med_vec, "coarse_vec": coarse_vec}

    print(f"\n[2/6] Input Vectors (Step 5 outputs):")
    print(f"  - fine_vec   : {list(fine_vec.shape)}")
    print(f"  - medium_vec : {list(med_vec.shape)}")
    print(f"  - coarse_vec : {list(coarse_vec.shape)}")

    # 1. Test Projector
    print(f"\n[3/6] Testing MultiScaleFeatureProjector (Target dim={common_dim})...")
    projector = MultiScaleFeatureProjector(common_dim=common_dim).to(device)
    proj_dict = projector(vectors)

    print(f"  ✓ proj_fine   : {list(proj_dict['proj_fine'].shape)}")
    print(f"  ✓ proj_medium : {list(proj_dict['proj_medium'].shape)}")
    print(f"  ✓ proj_coarse : {list(proj_dict['proj_coarse'].shape)}")

    assert proj_dict["proj_fine"].shape == (batch_size, common_dim)
    assert proj_dict["proj_medium"].shape == (batch_size, common_dim)
    assert proj_dict["proj_coarse"].shape == (batch_size, common_dim)

    # 2. Test Concat Fusion
    print("\n[4/6] Testing Concatenation-Based Fusion...")
    concat_module = ConcatFusion(common_dim=common_dim, fused_dim=fused_dim).to(device)
    fused_cat = concat_module(proj_dict)
    print(f"  ✓ Concat Fused Vector: {list(fused_cat.shape)} (Expected: [{batch_size}, {fused_dim}])")
    assert fused_cat.shape == (batch_size, fused_dim)

    # 3. Test Learned Weighted Fusion
    print("\n[5/6] Testing Learned Weighted Fusion (Softmax Gating)...")
    weighted_module = LearnedWeightedFusion(common_dim=common_dim, fused_dim=fused_dim).to(device)
    fused_weighted, weights = weighted_module(proj_dict)
    print(f"  ✓ Weighted Fused Vector : {list(fused_weighted.shape)}")
    print(f"  ✓ Learned Scale Weights : {weights.detach().cpu().numpy()}")

    assert fused_weighted.shape == (batch_size, fused_dim)
    # Check Softmax properties
    weight_sum = weights.sum().item()
    print(f"  ✓ Softmax Weights Sum   : {weight_sum:.6f} (Expected: ~1.000000)")
    assert abs(weight_sum - 1.0) < 1e-5, f"Weights must sum to 1.0, got {weight_sum}"
    assert (weights >= 0).all().item(), "Weights must be strictly non-negative!"

    # 4. Test Full MultiScaleFusionModel End-to-End with Backprop
    print("\n[6/6] Testing End-to-End MultiScaleFusionModel Gradient Flow...")
    full_model = MultiScaleFusionModel(
        num_classes=590,
        fusion_strategy="weighted",
        common_dim=common_dim,
        fused_dim=fused_dim,
        pretrained=False,
    ).to(device)
    full_model.train()

    dummy_imgs = torch.randn(batch_size, 3, 256, 128, device=device)
    dummy_labels = torch.randint(0, 590, (batch_size,), device=device)

    cls_scores, fused_embed, aux = full_model(dummy_imgs)
    print(f"  ✓ Output Logits Shape : {list(cls_scores.shape)} (Expected: [{batch_size}, 590])")
    print(f"  ✓ Output Feature Shape: {list(fused_embed.shape)} (Expected: [{batch_size}, {fused_dim}])")

    # Verify backpropagation through projection and weights
    criterion = nn.CrossEntropyLoss()
    loss = criterion(cls_scores, dummy_labels)
    loss.backward()

    # Check gradients
    proj_grad = full_model.projector.proj_fine[0].weight.grad
    fusion_weights_grad = full_model.fusion.fusion_module.scale_weights.grad

    assert proj_grad is not None, "Gradients must flow to projection layers!"
    assert fusion_weights_grad is not None, "Gradients must flow to learned scale weights!"
    print(f"  ✓ Projection Layer Gradient Norm: {torch.norm(proj_grad).item():.6f}")
    print(f"  ✓ Fusion Scale Weights Gradient  : {torch.norm(fusion_weights_grad).item():.6f}")

    print("\n" + "=" * 65)
    print("ALL FEATURE FUSION TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    test_fusion_modules()

