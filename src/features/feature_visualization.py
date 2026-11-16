"""Multi-Scale Feature Map Activation Visualization.

Renders spatial activation heatmaps across:
1. Original pedestrian image.
2. Fine-Scale activation map (layer1, 256 channels -> 64x32 spatial grid).
3. Medium-Scale activation map (layer2, 512 channels -> 32x16 spatial grid).
4. Coarse-Scale activation map (layer4, 2048 channels -> 16x8 spatial grid).

Activation heatmaps represent the mean spatial intensity response across channels:
    A(h, w) = (1 / C) * sum_{c=1}^C |F(c, h, w)|
Normalized strictly for visual inspection.
"""

import os
import sys
from typing import List, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.resnet50_multiscale import ResNet50MultiScale
from src.data.transforms import build_transforms
from src.enhancement.visualize_enhancement import find_representative_images, CAM_LABELS


def compute_activation_map(feat_tensor: torch.Tensor, target_size: Tuple[int, int] = (256, 128)) -> np.ndarray:
    """Compute normalized spatial activation heatmap from a (1, C, H, W) tensor."""
    # Mean absolute activation across channels: (1, 1, H, W)
    activation = torch.mean(torch.abs(feat_tensor), dim=1, keepdim=True)
    # Upsample to input image size for visual overlay comparison
    activation = F.interpolate(activation, size=target_size, mode="bilinear", align_corners=False)
    act_np = activation.squeeze().cpu().numpy()

    # Min-max normalization for display
    norm_act = (act_np - act_np.min()) / (act_np.max() - act_np.min() + 1e-8)
    return norm_act


def visualize_multiscale_features(
    model: ResNet50MultiScale,
    sample_paths: List[Tuple[str, str, str]],
    device: torch.device,
    output_dir: str = "results/multiscale_visualizations",
) -> List[str]:
    """Generate multi-scale feature activation panels for sample images."""
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    transform = build_transforms(is_train=False)

    saved_figures = []

    for idx, (img_path, cam_code, fname) in enumerate(sample_paths):
        with Image.open(img_path) as img:
            orig_pil = img.convert("RGB")

        input_tensor = transform(orig_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            feat_maps = model(input_tensor)

        fine_map = feat_maps["fine"]      # (1, 256, 64, 32)
        med_map = feat_maps["medium"]     # (1, 512, 32, 16)
        coarse_map = feat_maps["coarse"]  # (1, 2048, 16, 8)

        act_fine = compute_activation_map(fine_map)
        act_med = compute_activation_map(med_map)
        act_coarse = compute_activation_map(coarse_map)

        cam_name = CAM_LABELS.get(cam_code, f"Cam {cam_code}")

        # Render 4-column figure: Original | Fine | Medium | Coarse
        fig, axes = plt.subplots(1, 4, figsize=(14, 5.0))

        # Column 1: Original Image
        axes[0].imshow(orig_pil)
        axes[0].set_title(f"Input Pedestrian\n{cam_name}\n({orig_pil.size[0]}x{orig_pil.size[1]} px)", fontsize=10, fontweight="bold")
        axes[0].axis("off")

        # Column 2: Fine-Scale Map
        axes[1].imshow(act_fine, cmap="viridis")
        axes[1].set_title(f"Fine Scale (layer1)\nShape: [256, 64, 32]\nEdges & Local Textures", fontsize=10, color="navy")
        axes[1].axis("off")

        # Column 3: Medium-Scale Map
        axes[2].imshow(act_med, cmap="inferno")
        axes[2].set_title(f"Medium Scale (layer2)\nShape: [512, 32, 16]\nBody Regional Semantics", fontsize=10, color="darkred")
        axes[2].axis("off")

        # Column 4: Coarse-Scale Map
        axes[3].imshow(act_coarse, cmap="magma")
        axes[3].set_title(f"Coarse Scale (layer4)\nShape: [2048, 16, 8]\nGlobal Semantic Context", fontsize=10, color="purple")
        axes[3].axis("off")

        plt.suptitle(f"Multi-Scale Feature Responses: {fname}", fontsize=13, fontweight="bold")
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"multiscale_feat_{idx}_{fname.replace('.jpg', '')}.png")
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close()
        saved_figures.append(out_path)
        print(f"Saved feature visualization: {out_path}")

    return saved_figures


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running multi-scale visualization on: {device}")

    # Check for latest trained checkpoint, fallback to ImageNet pretrained
    ckpt_path = None
    if os.path.exists("results/checkpoints/best_model.pth"):
        ckpt_path = "results/checkpoints/best_model.pth"
    elif os.path.exists("results/checkpoints/checkpoint_batch_1200.pth"):
        ckpt_path = "results/checkpoints/checkpoint_batch_1200.pth"

    model = ResNet50MultiScale(pretrained=True, checkpoint_path=ckpt_path)
    model.to(device)

    samples = find_representative_images("datasets/AG-ReID.v2", count=6)
    visualize_multiscale_features(model, samples, device)


if __name__ == "__main__":
    main()

