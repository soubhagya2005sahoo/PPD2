"""Retrieval Ranking Visualization for Person Re-ID.

Renders visual ranking grids showing:
- Probe Query image (blue frame).
- Top-10 retrieved gallery images.
- Green frame if the retrieved image belongs to the true identity (Match).
- Red frame if the retrieved image belongs to an incorrect identity (Distractor).
- Pairwise cosine similarity score and camera tags.

Saves results to results/retrieval_examples/.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.data.dataset import AGReIDDataset
from src.data.transforms import build_transforms
from src.models.resnet50_baseline import ResNet50Baseline
from src.evaluate_baseline import extract_features

CAM_NAMES = {0: "Aerial (C0)", 2: "Wearable (C2)", 3: "CCTV (C3)"}


def visualize_topk(
    query_idx: int,
    q_feats: np.ndarray,
    q_pids: np.ndarray,
    q_cams: np.ndarray,
    q_paths: list,
    g_feats: np.ndarray,
    g_pids: np.ndarray,
    g_cams: np.ndarray,
    g_paths: list,
    topk: int = 10,
    out_dir: str = "results/retrieval_examples",
) -> str:
    """Plot a single query against its top-k retrieved gallery images."""
    os.makedirs(out_dir, exist_ok=True)

    q_feat = q_feats[query_idx]
    q_pid = q_pids[query_idx]
    q_cam = q_cams[query_idx]
    q_path = q_paths[query_idx]

    # Compute cosine distances: 1 - dot product
    sims = np.matmul(q_feat, g_feats.T)
    dists = 1.0 - sims

    # Sort gallery by increasing distance
    ranking = np.argsort(dists)

    # Filter junk matches (same identity and same camera)
    valid_gallery_indices = []
    for g_idx in ranking:
        if g_pids[g_idx] == q_pid and g_cams[g_idx] == q_cam:
            continue  # junk
        valid_gallery_indices.append(g_idx)
        if len(valid_gallery_indices) >= topk:
            break

    # Setup plot: 1 row with 1 (query) + topk (gallery) subplots
    total_cols = 1 + len(valid_gallery_indices)
    fig, axes = plt.subplots(1, total_cols, figsize=(2.2 * total_cols, 4.0))

    # Query image
    ax_q = axes[0]
    with Image.open(q_path) as img:
        ax_q.imshow(img)
    ax_q.set_title(f"QUERY\nID: P{q_pid:04d}\n{CAM_NAMES.get(q_cam, f'C{q_cam}')}", fontsize=9, fontweight="bold", color="blue")
    ax_q.axis("off")
    # Draw blue border
    rect = Rectangle((0, 0), 1, 1, transform=ax_q.transAxes, linewidth=4, edgecolor="blue", facecolor="none")
    ax_q.add_patch(rect)

    # Retrieved gallery images
    for rank_idx, g_idx in enumerate(valid_gallery_indices):
        ax_g = axes[1 + rank_idx]
        with Image.open(g_paths[g_idx]) as img:
            ax_g.imshow(img)

        is_match = (g_pids[g_idx] == q_pid)
        border_color = "green" if is_match else "red"
        status_text = "TRUE" if is_match else "FALSE"
        sim_val = sims[g_idx]

        title = f"Rank {rank_idx+1}: {status_text}\nID: P{g_pids[g_idx]:04d}\nSim: {sim_val:.2f}\n{CAM_NAMES.get(g_cams[g_idx], f'C{g_cams[g_idx]}')}"
        ax_g.set_title(title, fontsize=8, color=border_color, fontweight="bold")
        ax_g.axis("off")
        # Draw border
        rect = Rectangle((0, 0), 1, 1, transform=ax_g.transAxes, linewidth=3, edgecolor=border_color, facecolor="none")
        ax_g.add_patch(rect)

    plt.tight_layout()
    out_file = os.path.join(out_dir, f"retrieval_query_idx_{query_idx}_P{q_pid:04d}.png")
    plt.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close()
    return out_file


def run_visualizations(
    model: torch.nn.Module,
    query_loader: DataLoader,
    gallery_loader: DataLoader,
    device: torch.device,
    num_examples: int = 4,
    out_dir: str = "results/retrieval_examples",
) -> list:
    """Run retrieval visualizations for diverse sample queries."""
    q_feats, q_pids, q_cams, q_paths = extract_features(model, query_loader, device)
    g_feats, g_pids, g_cams, g_paths = extract_features(model, gallery_loader, device)

    # Pick queries from different cameras
    sample_indices = []
    cams_found = set()
    for idx, c in enumerate(q_cams):
        if c not in cams_found:
            sample_indices.append(idx)
            cams_found.add(c)
        if len(sample_indices) >= num_examples:
            break

    # If still need more, pick randomly
    if len(sample_indices) < num_examples:
        for idx in range(len(q_cams)):
            if idx not in sample_indices:
                sample_indices.append(idx)
            if len(sample_indices) >= num_examples:
                break

    saved_files = []
    for q_idx in sample_indices:
        out_f = visualize_topk(
            q_idx, q_feats, q_pids, q_cams, q_paths,
            g_feats, g_pids, g_cams, g_paths,
            topk=10, out_dir=out_dir,
        )
        saved_files.append(out_f)
        print(f"Saved visualization: {out_f}")

    return saved_files
