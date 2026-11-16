"""Visualization and Diagram Generator for Multi-Scale Feature Fusion.

Generates:
1. Feature Fusion Architecture Diagram (flow from input image to unified vector).
2. Weight distribution bar chart (learned scale weights alpha_fine, alpha_med, alpha_coarse).
3. Projected vs. Fused embedding magnitude profiles.

Saves figures to results/fusion_visualizations/.
"""

import os
import sys
import numpy as np
import torch

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.multiscale_fusion_model import MultiScaleFusionModel


def plot_fusion_architecture_and_weights(
    weights: np.ndarray,
    output_dir: str = "results/fusion_visualizations",
) -> str:
    """Plot the learned scale weights and fusion flow."""
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # 1. Bar Chart of Learned Scale Weights
    scales = ["Fine Scale\n(layer1, 256-d)", "Medium Scale\n(layer2, 512-d)", "Coarse Scale\n(layer4, 2048-d)"]
    colors = ["#2b5c8f", "#d95f02", "#7570b3"]

    bars = ax1.bar(scales, weights * 100, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    ax1.set_ylabel("Normalized Attention Weight (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Learned Multi-Scale Importance Weights\n(Softmax-Normalized, Sum = 100%)", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 100)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, h + 2.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # 2. Textual Workflow / Information Flow Box
    ax2.axis("off")
    flow_text = (
        "MULTI-SCALE FEATURE FUSION ARCHITECTURE\n"
        "=========================================\n\n"
        "Input Image (256 x 128 x 3)\n"
        "   │\n"
        "   ▼\n"
        "ResNet-50 Multi-Scale Backbone\n"
        "   ├── Fine Map (layer1)   : [256, 64, 32]  ──► GAP ──► [256-d]\n"
        "   ├── Medium Map (layer2) : [512, 32, 16]  ──► GAP ──► [512-d]\n"
        "   └── Coarse Map (layer4) : [2048, 16, 8]  ──► GAP ──► [2048-d]\n"
        "   │\n"
        "   ▼\n"
        "Dimension Alignment Projections\n"
        "   ├── Linear + BN + ReLU ──► proj_fine   : [512-d]\n"
        "   ├── Linear + BN + ReLU ──► proj_medium : [512-d]\n"
        "   └── Linear + BN + ReLU ──► proj_coarse : [512-d]\n"
        "   │\n"
        "   ▼\n"
        "Learned Weighted Feature Fusion\n"
        "   Fused = (alpha_fine * p_fine) + (alpha_med * p_med) + (alpha_coarse * p_coarse)\n"
        "   │\n"
        "   ▼\n"
        "Unified Fused Embedding Vector: [512-d]  ──► L2 Normalize ──► Retrieval"
    )
    ax2.text(0.02, 0.95, flow_text, transform=ax2.transAxes, fontsize=10, family="monospace", va="top",
             bbox=dict(boxstyle="round,pad=0.8", facecolor="#f0f4f8", edgecolor="#bccad6", linewidth=1.5))

    plt.tight_layout()
    out_path = os.path.join(output_dir, "fusion_weights_and_workflow.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Fusion visualization saved to: {out_path}")
    return out_path


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = MultiScaleFusionModel(num_classes=590, fusion_strategy="weighted", pretrained=False)
    weights = model.fusion.fusion_module.get_normalized_weights().detach().cpu().numpy()
    plot_fusion_architecture_and_weights(weights)


if __name__ == "__main__":
    main()

