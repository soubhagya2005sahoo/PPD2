"""Unit Tests for Step 7: View-Aware Conditioning, Latent Diffusion & Transforms."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from PIL import Image

from src.models.view_aware import ViewAwareModule, map_cam_to_view
from src.models.latent_diffusion import LatentFeatureDiffusion
from src.models.view_aware_fusion_model import ViewAwareMultiScaleReIDModel
from src.data.transforms import build_transforms


def test_proposed_pipeline():
    print("Testing Step 7: View-Aware Multi-Scale Fusion & Latent Diffusion...")
    torch.manual_seed(42)

    # 1. Test camera ID mapping
    cams = torch.tensor([0, 2, 3, 0])
    views = map_cam_to_view(cams)
    assert torch.equal(views, torch.tensor([0, 1, 2, 0])), f"Unexpected view mapping: {views}"
    print("  ✓ Camera-to-View mapping verified: {0, 2, 3} -> {0, 1, 2}")

    # 2. Test ViewAwareModule directly
    va = ViewAwareModule(common_dim=512, num_views=3, view_dim=128)
    proj_dict = {
        "proj_fine": torch.randn(4, 512),
        "proj_medium": torch.randn(4, 512),
        "proj_coarse": torch.randn(4, 512),
    }
    fused, scale_w, view_logits, view_ids = va(proj_dict, cam_ids=cams)
    assert fused.shape == (4, 512)
    assert scale_w.shape == (4, 3)
    assert torch.allclose(scale_w.sum(dim=1), torch.ones(4), atol=1e-5)
    assert view_logits.shape == (4, 3)
    print(f"  ✓ ViewAwareModule output verified: fused={fused.shape}, scale_weights sum=1.0")

    # Test ViewAwareModule without cam_ids (predicted fallback)
    fused_nocam, _, _, view_ids_pred = va(proj_dict, cam_ids=None)
    assert fused_nocam.shape == (4, 512)
    assert view_ids_pred.shape == (4,)
    print("  ✓ ViewAwareModule fallback (cam_ids=None) verified")

    # 3. Test LatentFeatureDiffusion (train + eval)
    ld = LatentFeatureDiffusion(feat_dim=512, num_views=3)
    ld_params = sum(p.numel() for p in ld.parameters())
    ld.train()
    z_train, diff_loss = ld(fused, view_ids=view_ids)
    assert z_train.shape == (4, 512)
    assert diff_loss.item() >= 0.0
    ld.eval()
    z_eval, zero_loss = ld(fused, view_ids=view_ids)
    assert z_eval.shape == (4, 512)
    assert zero_loss.item() == 0.0
    print(f"  ✓ LatentFeatureDiffusion verified: {ld_params:,} params ({ld_params/1e6:.2f}M)")

    # 4. Test full ViewAwareMultiScaleReIDModel end-to-end
    model = ViewAwareMultiScaleReIDModel(
        num_classes=590,
        pretrained=False,
        use_view_aware=True,
        use_diffusion=True,
    )
    total_params = sum(p.numel() for p in model.parameters())
    model.train()
    x = torch.randn(4, 3, 256, 128)
    pids = torch.tensor([0, 1, 2, 3])

    cls_score, feat, aux = model(x, cam_ids=cams)
    assert cls_score.shape == (4, 590)
    assert feat.shape == (4, 512)
    assert aux["diff_loss"].requires_grad
    assert aux["view_logits"].shape == (4, 3)

    # Check gradient flow across all modules
    loss = (
        nn.CrossEntropyLoss()(cls_score, pids)
        + nn.CrossEntropyLoss()(aux["view_logits"], aux["view_ids"])
        + aux["diff_loss"]
    )
    loss.backward()
    assert model.view_aware.view_embedding.weight.grad is not None
    assert model.latent_diffusion.denoiser.net[0].weight.grad is not None
    print(f"  ✓ Full model training & gradient flow verified ({total_params/1e6:.2f}M total params)")

    # Eval mode check
    model.eval()
    with torch.no_grad():
        emb = model(x, cam_ids=cams)
        norms = torch.norm(emb, p=2, dim=1)
        assert emb.shape == (4, 512)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)
    print("  ✓ Eval mode L2-normalized embedding verified")

    # 5. Test enhanced transform pipeline
    tfm = build_transforms(is_train=True, use_enhancement=True, use_random_erasing=True)
    dummy_img = Image.new("RGB", (100, 200), (80, 100, 120))
    t_img = tfm(dummy_img)
    assert t_img.shape == (3, 256, 128)
    print("  ✓ On-the-fly CLAHE + Gamma + RandomErasing transform verified")

    print("\nALL STEP 7 UNIT TESTS PASSED!")


if __name__ == "__main__":
    test_proposed_pipeline()

