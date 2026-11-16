"""Visualization Script for AG-ReID.v2 Sample Images.

Demonstrates cross-view and multi-scale variations by selecting
representative samples across:
- Camera platforms: C0 (UAV Aerial), C2 (Wearable Glasses), C3 (Ground CCTV)
- Altitudes: A0 (Low), A1 (Medium), A2 (High)

Saves the visual grid to results/sample_cross_view_visualization.png.
"""

import os
import re
import sys
from PIL import Image

# Use headless backend and /tmp cache for non-interactive / remote environments
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAMERA_NAMES = {
    "0": "Aerial UAV (C0)",
    "2": "Wearable (C2)",
    "3": "Ground CCTV (C3)",
}

ALTITUDE_NAMES = {
    "0": "Low Alt (A0)",
    "1": "Med Alt (A1)",
    "2": "High Alt (A2)",
}


def find_dataset_root() -> str:
    candidates = [
        "datasets/AG-ReID.v2/AG-ReID.v2",
        "datasets/AG-ReID.v2",
        "/Users/soubhagyaranjansahoo/person re identification /datasets/AG-ReID.v2/AG-ReID.v2",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "gallery")) and os.path.exists(os.path.join(c, "train_all")):
            return c
    raise FileNotFoundError("Could not find AG-ReID.v2 dataset splits.")


def collect_diverse_samples(dataset_root: str, target_count: int = 6):
    """Find images for a single identity captured across different cameras/altitudes."""
    filename_regex = re.compile(
        r"^P(?P<pid>\d+)T(?P<tid>\d+)A(?P<alt>\d+)C(?P<cam>\d+)F(?P<frame>\d+[A-Z]?)\.jpg$",
        re.IGNORECASE,
    )

    # Search in train_all or gallery for identities that appear in multiple cameras
    samples_by_pid = {}
    for split in ["gallery", "train_all"]:
        split_path = os.path.join(dataset_root, split)
        if not os.path.isdir(split_path):
            continue
        for root, _, files in os.walk(split_path):
            for f in files:
                m = filename_regex.match(f)
                if m:
                    pid = m.group("pid")
                    cam = m.group("cam")
                    alt = m.group("alt")
                    if pid not in samples_by_pid:
                        samples_by_pid[pid] = {}
                    key = (cam, alt)
                    if key not in samples_by_pid[pid]:
                        samples_by_pid[pid][key] = os.path.join(root, f)

    # Find a person with the highest variety of camera/altitude views
    best_pid = max(samples_by_pid.keys(), key=lambda k: len(samples_by_pid[k]))
    selected = list(samples_by_pid[best_pid].values())[:target_count]

    # If best_pid has fewer than target_count, supplement with diverse samples from other PIDs
    if len(selected) < target_count:
        for pid, view_dict in samples_by_pid.items():
            if pid == best_pid:
                continue
            for img_path in view_dict.values():
                selected.append(img_path)
                if len(selected) >= target_count:
                    break
            if len(selected) >= target_count:
                break

    return selected[:target_count], filename_regex


def main():
    print("=" * 60)
    print("Generating AG-ReID.v2 Sample Cross-View Visualization")
    print("=" * 60)

    dataset_root = find_dataset_root()
    print(f"Dataset root: {dataset_root}")

    sample_paths, regex = collect_diverse_samples(dataset_root, target_count=6)
    print(f"Selected {len(sample_paths)} representative samples.")

    # Create visualization grid (2 rows, 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(12, 9))
    axes = axes.flatten()

    for idx, img_path in enumerate(sample_paths):
        ax = axes[idx]
        fname = os.path.basename(img_path)
        m = regex.match(fname)

        with Image.open(img_path) as img:
            w, h = img.size
            ax.imshow(img)

        if m:
            pid = m.group("pid")
            cam = m.group("cam")
            alt = m.group("alt")
            cam_str = CAMERA_NAMES.get(cam, f"Cam {cam}")
            alt_str = ALTITUDE_NAMES.get(alt, f"Alt {alt}")
            title = f"ID: P{pid}\n{cam_str} | {alt_str}\nSize: {w}x{h} px"
        else:
            title = f"{fname}\nSize: {w}x{h} px"

        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")

    plt.suptitle("AG-ReID.v2: Cross-View & Multi-Scale Sample Representations", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "sample_cross_view_visualization.png")
    plt.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close()

    print(f"Visualization successfully saved to: {out_file}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
