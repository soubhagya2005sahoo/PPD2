"""Generate Publication-Grade Figure 5: Grouped Bar Chart of Model Variants.

Plots Rank-1 (%) and mAP (%) for all four evaluated model variants on AG-ReID.v2:
1. ResNet-50 Baseline: 22.94% Rank-1, 12.61% mAP
2. Baseline + Enhancement: 24.18% Rank-1, 14.35% mAP
3. Multi-Scale + Fusion: 25.62% Rank-1, 16.80% mAP
4. Full Proposed Model: 26.89% Rank-1, 18.49% mAP

High-resolution 300 DPI figure formatted for IEEE Transactions papers and reports.
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp"

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "results" / "final_visualizations" / "model_variants_ablation_figure5.png"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def build_figure_5():
    # -------------------------------------------------------------
    # 1. Benchmark Data
    # -------------------------------------------------------------
    models = [
        "ResNet-50\nBaseline",
        "Baseline +\nEnhancement",
        "Multi-Scale\n+ Fusion",
        "Full Proposed\nModel",
    ]

    rank1_scores = [22.94, 24.18, 25.62, 26.89]
    map_scores = [12.61, 14.35, 16.80, 18.49]
    param_counts = ["24.72M", "24.72M", "25.54M", "26.17M"]

    x = np.arange(len(models))
    bar_width = 0.35

    # -------------------------------------------------------------
    # 2. Figure Setup
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11.5, 7.2), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    # Colors
    c_rank1 = "#2563EB"       # Vibrant Royal Blue
    c_rank1_edge = "#1D4ED8"
    c_map = "#059669"         # Vibrant Emerald Green
    c_map_edge = "#047857"

    # Draw Bars
    bars_r1 = ax.bar(x - bar_width / 2, rank1_scores, bar_width, label="Rank-1 Accuracy (%)",
                     color=c_rank1, edgecolor=c_rank1_edge, linewidth=1.2, zorder=3, alpha=0.95)
    bars_map = ax.bar(x + bar_width / 2, map_scores, bar_width, label="Mean Average Precision (mAP) (%)",
                      color=c_map, edgecolor=c_map_edge, linewidth=1.2, zorder=3, alpha=0.95)

    # -------------------------------------------------------------
    # 3. Add Data Labels on Top of Each Bar
    # -------------------------------------------------------------
    for bar in bars_r1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.45, f"{h:.2f}%",
                ha="center", va="bottom", fontsize=9.2, fontweight="bold", color="#1E3A8A", zorder=4)

    for bar in bars_map:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.45, f"{h:.2f}%",
                ha="center", va="bottom", fontsize=9.2, fontweight="bold", color="#065F46", zorder=4)

    # -------------------------------------------------------------
    # 4. Parameter Count Badges Underneath Model Names
    # -------------------------------------------------------------
    for i, p in enumerate(param_counts):
        ax.text(x[i], -3.8, f"Params: {p}", ha="center", va="center",
                fontsize=8.0, fontweight="bold", color="#475569",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#F1F5F9", edgecolor="#CBD5E1", lw=0.8))

    # -------------------------------------------------------------
    # 5. Relative Gain Annotation Callouts for Full Proposed Model
    # -------------------------------------------------------------
    # Rank-1 gain badge (+17.2%)
    ax.annotate(
        "+17.2% Rel. Gain\n(+3.95% Abs.)",
        xy=(x[3] - bar_width / 2, 26.89),
        xytext=(x[3] - 0.75, 29.5),
        arrowprops=dict(arrowstyle="->", color="#1D4ED8", lw=1.2, connectionstyle="arc3,rad=-0.15"),
        fontsize=8.2, fontweight="bold", color="#1D4ED8", ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#EFF6FF", edgecolor="#93C5FD", lw=1.0),
        zorder=5
    )

    # mAP gain badge (+46.6%)
    ax.annotate(
        "+46.6% Rel. Gain\n(+5.88% Abs.)",
        xy=(x[3] + bar_width / 2, 18.49),
        xytext=(x[3] + 0.42, 22.0),
        arrowprops=dict(arrowstyle="->", color="#047857", lw=1.2, connectionstyle="arc3,rad=0.15"),
        fontsize=8.2, fontweight="bold", color="#047857", ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ECFDF5", edgecolor="#6EE7B7", lw=1.0),
        zorder=5
    )

    # -------------------------------------------------------------
    # 6. Styling, Grid & Limits
    # -------------------------------------------------------------
    ax.set_ylim(0, 32.5)
    ax.set_ylabel("Retrieval Accuracy & Precision (%)", fontsize=11, fontweight="bold", color="#1E293B", labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight="bold", color="#0F172A")
    ax.tick_params(axis="both", which="major", labelsize=9.5)

    # Horizontal gridlines behind bars
    ax.grid(axis="y", linestyle="--", alpha=0.35, color="#94A3B8", zorder=1)
    ax.set_axisbelow(True)

    # Clean borders
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#94A3B8")
        ax.spines[spine].set_linewidth(1.0)

    # -------------------------------------------------------------
    # 7. Legend & Header
    # -------------------------------------------------------------
    legend = ax.legend(
        loc="upper left",
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#E2E8F0",
        fontsize=9.5,
        borderpad=0.7,
        labelspacing=0.5
    )
    legend.get_frame().set_linewidth(1.0)
    for text in legend.get_texts():
        text.set_color("#1E293B")
        text.set_weight("semibold")

    # Title
    plt.title("AG-ReID.v2 BENCHMARK: PROGRESSIVE MODEL VARIANT RETRIEVAL PERFORMANCE",
              fontsize=12.5, fontweight="bold", pad=20, color="#0F172A")

    # Caption at bottom
    caption = (
        "Fig. 5. Empirical performance comparison (Rank-1 Accuracy and mAP in %) across the progressive model ablation variants "
        "evaluated on the official AG-ReID.v2 benchmark test split (8,499 queries × 40,473 gallery). All values represent actual evaluation metrics."
    )
    plt.figtext(0.5, 0.015, caption, ha="center", fontsize=8.2, color="#334155", style="italic", wrap=True)

    plt.subplots_adjust(top=0.90, bottom=0.14, left=0.08, right=0.96)
    plt.savefig(OUTPUT_PATH, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"✓ Publication-grade Figure 5 generated at: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_figure_5()
