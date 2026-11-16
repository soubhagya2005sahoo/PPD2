"""Validation and Unit Test for Multi-Scale Feature Extraction Pipeline.

Verifies:
1. Feature map extraction across fine (layer1), medium (layer2), and coarse (layer4) stages.
2. Exact spatial dimensions and channel counts.
3. Global Average Pooling (GAP) into 1D vectors.
4. Numerical stability (checks for NaN / Inf).
5. Device execution on Apple Silicon (MPS) or CPU.
"""

import sys
import os
import torch

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.resnet50_multiscale import ResNet50MultiScale
from src.features.feature_pooling import MultiScaleFeaturePooler


def test_multiscale_extraction():
    print("=" * 65)
    print("TESTING MULTI-SCALE FEATURE EXTRACTION PIPELINE")
    print("=" * 65)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[1/5] Compute Device Selected: {device}")

    # Initialize model (without downloading weights for fast unit testing)
    print("\n[2/5] Initializing ResNet50MultiScale model...")
    model = ResNet50MultiScale(pretrained=False, medium_stage="layer2").to(device)
    pooler = MultiScaleFeaturePooler(l2_normalize=True).to(device)
    model.eval()
    print("  ✓ Model & Pooler initialized successfully")

    # Synthetic batch of 2 images of size 256 x 128
    batch_size = 2
    x = torch.randn(batch_size, 3, 256, 128, device=device)
    print(f"\n[3/5] Testing 2D Feature Map Extraction on input: {list(x.shape)}...")

    with torch.no_grad():
        feat_maps = model(x)

    fine_map = feat_maps["fine"]
    med_map = feat_maps["medium"]
    coarse_map = feat_maps["coarse"]

    print(f"  ✓ Fine Feature Map (layer1)   : {list(fine_map.shape)}   (Expected: [2, 256, 64, 32])")
    print(f"  ✓ Medium Feature Map (layer2) : {list(med_map.shape)}   (Expected: [2, 512, 32, 16])")
    print(f"  ✓ Coarse Feature Map (layer4) : {list(coarse_map.shape)}  (Expected: [2, 2048, 16, 8])")

    assert fine_map.shape == (batch_size, 256, 64, 32), f"Unexpected fine shape: {fine_map.shape}"
    assert med_map.shape == (batch_size, 512, 32, 16), f"Unexpected medium shape: {med_map.shape}"
    assert coarse_map.shape == (batch_size, 2048, 16, 8), f"Unexpected coarse shape: {coarse_map.shape}"

    # Pool into 1D vectors
    print("\n[4/5] Testing Global Average Pooling (GAP)...")
    pooled = pooler(feat_maps)

    fine_vec = pooled["fine_vec"]
    med_vec = pooled["medium_vec"]
    coarse_vec = pooled["coarse_vec"]

    print(f"  ✓ Fine Pooled Vector   : {list(fine_vec.shape)}    (Expected: [2, 256])")
    print(f"  ✓ Medium Pooled Vector : {list(med_vec.shape)}    (Expected: [2, 512])")
    print(f"  ✓ Coarse Pooled Vector : {list(coarse_vec.shape)}   (Expected: [2, 2048])")

    assert fine_vec.shape == (batch_size, 256), f"Unexpected fine vector shape: {fine_vec.shape}"
    assert med_vec.shape == (batch_size, 512), f"Unexpected medium vector shape: {med_vec.shape}"
    assert coarse_vec.shape == (batch_size, 2048), f"Unexpected coarse vector shape: {coarse_vec.shape}"

    # Numerical validity checks
    print("\n[5/5] Checking Numerical Validity (NaN / Inf checks)...")
    for name, tensor in [("fine_map", fine_map), ("med_map", med_map), ("coarse_map", coarse_map),
                         ("fine_vec", fine_vec), ("med_vec", med_vec), ("coarse_vec", coarse_vec)]:
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        assert not has_nan, f"NaN detected in {name}!"
        assert not has_inf, f"Inf detected in {name}!"
        print(f"  ✓ {name:12s}: Clean (NaN={has_nan}, Inf={has_inf})")

    # L2 norm check
    fine_norm = torch.norm(fine_vec[0], p=2).item()
    coarse_norm = torch.norm(coarse_vec[0], p=2).item()
    print(f"  ✓ L2 Norm of fine vector   : {fine_norm:.4f} (Expected: ~1.0)")
    print(f"  ✓ L2 Norm of coarse vector : {coarse_norm:.4f} (Expected: ~1.0)")
    assert abs(fine_norm - 1.0) < 1e-4, "L2 normalization check failed"

    print("\n" + "=" * 65)
    print("ALL MULTI-SCALE FEATURE TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    test_multiscale_extraction()

