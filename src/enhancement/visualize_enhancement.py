"""Visualization and Metric Evaluation for Image Enhancement.

Generates:
1. 4-way visual comparison: Original vs. CLAHE vs. Gamma vs. CLAHE+Gamma.
2. Cross-camera comparative grids across UAV (C0), Wearable (C2), and CCTV (C3).
3. Image quality evaluation metrics saved to results/enhancement_metrics.csv:
   - Mean Brightness (Luminance)
   - Standard Deviation of Pixel Intensity
   - RMS Contrast
   - Shannon Information Entropy
"""

import os
import sys
import csv
import glob
import math
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.enhancement.clahe import apply_clahe
from src.enhancement.gamma import apply_gamma_correction
from src.enhancement.image_enhancer import ImageEnhancer

CAM_LABELS = {"0": "UAV Aerial (C0)", "2": "Wearable (C2)", "3": "Ground CCTV (C3)"}


def calculate_image_metrics(img_array: np.ndarray) -> Dict[str, float]:
    """Compute standard statistical and informational image quality metrics.

    Args:
        img_array: RGB uint8 numpy array (H, W, 3).

    Returns:
        Dictionary containing:
        - 'mean_brightness': Average grayscale intensity [0, 255].
        - 'std_contrast': Standard deviation of pixel intensities.
        - 'rms_contrast': Root-mean-square contrast.
        - 'shannon_entropy': Information entropy in bits.
    """
    # Convert to grayscale luminance: Y = 0.299 R + 0.587 G + 0.114 B
    gray = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))

    # RMS Contrast = sqrt(1/(N-1) * sum((I - mean)^2))
    rms_contrast = std_val / 255.0

    # Shannon Information Entropy: H = - sum(p * log2(p))
    hist, _ = np.histogram(gray, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]  # Avoid log2(0)
    entropy = -float(np.sum(hist * np.log2(hist)))

    return {
        "mean_brightness": round(mean_val, 2),
        "std_intensity": round(std_val, 2),
        "rms_contrast": round(rms_contrast, 4),
        "shannon_entropy": round(entropy, 4),
    }


def find_representative_images(dataset_root: str, count: int = 6) -> List[Tuple[str, str, str]]:
    """Select representative sample images across different cameras (C0, C2, C3)."""
    candidates = []
    # Search in gallery and train_all
    for split in ["gallery", "train_all"]:
        split_dir = os.path.join(dataset_root, split)
        if not os.path.isdir(split_dir):
            split_dir = os.path.join(dataset_root, "AG-ReID.v2", split)
        if not os.path.isdir(split_dir):
            continue

        for root, _, files in os.walk(split_dir):
            for f in files:
                if f.endswith(".jpg"):
                    for cam_code in ["0", "2", "3"]:
                        if f"C{cam_code}" in f:
                            candidates.append((os.path.join(root, f), cam_code, f))
                            break
            if len(candidates) >= 100:
                break
        if len(candidates) >= 100:
            break

    # Pick balanced samples across C0, C2, C3
    selected = []
    by_cam = {"0": [], "2": [], "3": []}
    for item in candidates:
        by_cam[item[1]].append(item)

    per_cam = max(1, count // 3)
    for c in ["0", "2", "3"]:
        selected.extend(by_cam[c][:per_cam])

    return selected[:count]


def run_visual_and_metric_evaluation(
    dataset_root: str = "datasets/AG-ReID.v2",
    output_dir: str = "results/enhancement_visualizations",
    metrics_csv: str = "results/enhancement_metrics.csv",
    clip_limit: float = 2.0,
    gamma: float = 1.2,
) -> None:
    """Generate visual comparisons and metric evaluations."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(metrics_csv), exist_ok=True)

    samples = find_representative_images(dataset_root, count=6)
    if not samples:
        print(f"Warning: No sample images found in {dataset_root}")
        return

    enhancer_clahe = ImageEnhancer(mode="clahe", clip_limit=clip_limit)
    enhancer_gamma = ImageEnhancer(mode="gamma", gamma=gamma)
    enhancer_both = ImageEnhancer(mode="clahe_gamma", clip_limit=clip_limit, gamma=gamma, order="clahe_first")

    csv_rows = []

    # 1. Process individual samples and generate multi-panel grid
    num_samples = len(samples)
    fig, axes = plt.subplots(num_samples, 4, figsize=(14, 3.5 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)

    for idx, (img_path, cam_code, fname) in enumerate(samples):
        with Image.open(img_path) as img:
            orig_rgb = np.array(img.convert("RGB"))

        clahe_rgb = enhancer_clahe.enhance(orig_rgb)
        gamma_rgb = enhancer_gamma.enhance(orig_rgb)
        both_rgb = enhancer_both.enhance(orig_rgb)

        versions = [
            ("Original", orig_rgb),
            (f"CLAHE (clip={clip_limit})", clahe_rgb),
            (f"Gamma (γ={gamma})", gamma_rgb),
            ("CLAHE + Gamma", both_rgb),
        ]

        cam_name = CAM_LABELS.get(cam_code, f"Cam {cam_code}")

        for col_idx, (title, v_img) in enumerate(versions):
            ax = axes[idx, col_idx]
            ax.imshow(v_img)
            m = calculate_image_metrics(v_img)
            header = f"{title}\n{cam_name}" if col_idx == 0 else title
            ax.set_title(f"{header}\nB:{m['mean_brightness']} | E:{m['shannon_entropy']}", fontsize=9)
            ax.axis("off")

            # Record metrics
            csv_rows.append({
                "sample_filename": fname,
                "camera": cam_name,
                "method": title.split()[0],
                "mean_brightness": m["mean_brightness"],
                "std_intensity": m["std_intensity"],
                "rms_contrast": m["rms_contrast"],
                "shannon_entropy": m["shannon_entropy"],
            })

    plt.suptitle("AG-ReID.v2: Image Enhancement Pipeline Comparison", fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout()
    grid_path = os.path.join(output_dir, "enhancement_comparison_grid.png")
    plt.savefig(grid_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Visual comparison grid saved to: {grid_path}")

    # 2. Write CSV report
    fieldnames = ["sample_filename", "camera", "method", "mean_brightness", "std_intensity", "rms_contrast", "shannon_entropy"]
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Metrics table saved to: {metrics_csv}")


if __name__ == "__main__":
    run_visual_and_metric_evaluation()

