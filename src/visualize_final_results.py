"""Final Research Results Visualization Generator.

Reads JSON evaluation summaries from `results/eval_*.json` and generates
publication-ready figures in `results/final_visualizations/`:
1. Overall Rank-1, mAP, mINP comparison across models.
2. Cross-View (Aerial -> Ground and Ground -> Aerial) comparison.
3. Learned View-Aware Scale Attention weights across camera platforms.
4. Computational Efficiency comparison vs. heavy pixel-space diffusion models.
"""

import os
import sys
import json
import glob

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_eval_results(json_paths):
    results = []
    for path in json_paths:
        if os.path.exists(path):
            with open(path, "r") as f:
                results.append(json.load(f))
    return results


def plot_overall_comparison(results, out_dir):
    names = [r["experiment"] for r in results]
    r1 = [r["overall"]["Rank-1"] for r in results]
    m_ap = [r["overall"]["mAP"] for r in results]
    m_inp = [r["overall"]["mINP"] for r in results]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=180)
    b1 = ax.bar(x - width, r1, width, label="Rank-1 (%)", color="#2563EB")
    b2 = ax.bar(x, m_ap, width, label="mAP (%)", color="#10B981")
    b3 = ax.bar(x + width, m_inp, width, label="mINP (%)", color="#F59E0B")

    ax.set_ylabel("Score (%)")
    ax.set_title("AG-ReID.v2 Overall Re-ID Performance Comparison", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=10, ha="right")
    ax.set_ylim(0, max(max(r1), max(m_ap), max(m_inp), 10) * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "model_comparison_metrics.png")
    plt.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_cross_view_comparison(results, out_dir):
    names = [r["experiment"] for r in results]
    a2g_r1 = [r["aerial_to_ground"]["Rank-1"] for r in results]
    g2a_r1 = [r["ground_to_aerial"]["Rank-1"] for r in results]
    a2g_map = [r["aerial_to_ground"]["mAP"] for r in results]
    g2a_map = [r["ground_to_aerial"]["mAP"] for r in results]

    x = np.arange(len(names))
    width = 0.32

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=180)

    # Rank-1 subplot
    b1 = axes[0].bar(x - width / 2, a2g_r1, width, label="Aerial -> Ground", color="#3B82F6")
    b2 = axes[0].bar(x + width / 2, g2a_r1, width, label="Ground -> Aerial", color="#8B5CF6")
    axes[0].set_title("Cross-View Rank-1 Accuracy (%)", fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=12, ha="right")
    axes[0].set_ylim(0, max(max(a2g_r1), max(g2a_r1), 10) * 1.25)
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)
    axes[0].legend()
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            axes[0].annotate(f"{h:.2f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                             xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    # mAP subplot
    b3 = axes[1].bar(x - width / 2, a2g_map, width, label="Aerial -> Ground", color="#059669")
    b4 = axes[1].bar(x + width / 2, g2a_map, width, label="Ground -> Aerial", color="#D97706")
    axes[1].set_title("Cross-View mAP (%)", fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=12, ha="right")
    axes[1].set_ylim(0, max(max(a2g_map), max(g2a_map), 10) * 1.25)
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)
    axes[1].legend()
    for bars in (b3, b4):
        for bar in bars:
            h = bar.get_height()
            axes[1].annotate(f"{h:.2f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                             xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "cross_view_comparison.png")
    plt.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_view_scale_weights(proposed_result, out_dir):
    sw = proposed_result.get("view_scale_weights")
    if not sw:
        return None

    views = list(sw.keys())
    fine = [sw[v]["fine"] for v in views]
    med = [sw[v]["medium"] for v in views]
    coarse = [sw[v]["coarse"] for v in views]

    x = np.arange(len(views))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=180)
    b1 = ax.bar(x - width, fine, width, label="Fine (layer1)", color="#EF4444")
    b2 = ax.bar(x, med, width, label="Medium (layer2)", color="#F59E0B")
    b3 = ax.bar(x + width, coarse, width, label="Coarse (layer4)", color="#3B82F6")

    ax.set_ylabel("Learned Attention Weight")
    ax.set_title("View-Aware Dynamic Multi-Scale Fusion Weights per Camera Platform", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(views)
    ax.set_ylim(0, max(max(fine), max(med), max(coarse)) * 1.3)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "view_scale_weights.png")
    plt.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_efficiency_comparison(proposed_result, out_dir):
    # Reference published Stable Diffusion v1.5 U-Net parameter count (~860M) + typical image encoder
    # vs our proposed compact 1D latent diffusion model
    models = ["SD-ReID / Pixel Diffusion\n(Published Reference)", proposed_result["experiment"]]
    params = [860.0, proposed_result["efficiency"]["params_millions"]]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=180)
    bars = ax.barh(models, params, color=["#94A3B8", "#10B981"], height=0.45)
    ax.set_xlabel("Model Parameters (Millions, Lower = More Accessible)")
    ax.set_title("Parameter Efficiency: Proposed Latent Diffusion vs. Pixel Diffusion", fontweight="bold")
    ax.set_xlim(0, max(params) * 1.18)
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:.2f}M", xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "efficiency_vs_diffusion.png")
    plt.savefig(out_path)
    plt.close(fig)
    return out_path


def main():
    out_dir = "results/final_visualizations"
    os.makedirs(out_dir, exist_ok=True)

    # Order JSON files logically if present
    preferred = [
        "results/eval_baseline.json",
        "results/eval_enhanced_baseline.json",
        "results/eval_proposed_full.json",
    ]
    json_files = [p for p in preferred if os.path.exists(p)]
    if not json_files:
        json_files = sorted(glob.glob("results/eval_*.json"))

    results = load_eval_results(json_files)
    if not results:
        print("No eval JSON files found in results/.")
        sys.exit(1)

    p1 = plot_overall_comparison(results, out_dir)
    p2 = plot_cross_view_comparison(results, out_dir)
    print(f"Generated: {p1}")
    print(f"Generated: {p2}")

    # Plot view-aware scale weights & efficiency from latest proposed model
    for r in reversed(results):
        if "view_scale_weights" in r:
            p3 = plot_view_scale_weights(r, out_dir)
            if p3:
                print(f"Generated: {p3}")
            p4 = plot_efficiency_comparison(r, out_dir)
            print(f"Generated: {p4}")
            break


if __name__ == "__main__":
    main()

