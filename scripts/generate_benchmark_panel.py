"""Generate Authorised Benchmark Camera Views Panel for AG-ReID.v2.

Creates a publication-grade, labelled comparison figure showing authentic pedestrian
examples across the three heterogeneous camera platforms:
1. Aerial UAV Drone View (C0)
2. Wearable Smart Glasses View (C2)
3. Ground CCTV Surveillance View (C3)

Preserves exact source file labels from the official dataset release.
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp"

from pathlib import Path
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_BASE = PROJECT_ROOT / "datasets" / "AG-ReID.v2" / "AG-ReID.v2" / "gallery"
OUTPUT_PATH = PROJECT_ROOT / "results" / "final_visualizations" / "benchmark_camera_views_panel.png"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Selected authentic identities with high-quality cross-view presence
IDENTITIES = [
    {
        "id_label": "Identity 1 (P0132)",
        "tracklet": "Tracklet T02210 / Altitude Tier A1",
        "c0": ("P0132T02210A1/P0132T02210A1C0F10501.jpg", "P0132T02210A1C0F10501.jpg"),
        "c2": ("P0132T02210A1/P0132T02210A1C2F8341.jpg",  "P0132T02210A1C2F8341.jpg"),
        "c3": ("P0132T02210A1/P0132T02210A1C3F4801.jpg",  "P0132T02210A1C3F4801.jpg"),
    },
    {
        "id_label": "Identity 2 (P0297)",
        "tracklet": "Tracklet T04041 / Altitude Tier A2",
        "c0": ("P0297T04041A2/P0297T04041A2C0F15691.jpg", "P0297T04041A2C0F15691.jpg"),
        "c2": ("P0297T04041A2/P0297T04041A2C2F20131.jpg", "P0297T04041A2C2F20131.jpg"),
        "c3": ("P0297T04041A2/P0297T04041A2C3F8761.jpg",  "P0297T04041A2C3F8761.jpg"),
    },
    {
        "id_label": "Identity 3 (P2207)",
        "tracklet": "Tracklet T03241 / Altitude Tier A0",
        "c0": ("P2207T03241A0/P2207T03241A0C0F26761.jpg", "P2207T03241A0C0F26761.jpg"),
        "c2": ("P2207T03241A0/P2207T03241A0C2F24871.jpg", "P2207T03241A0C2F24871.jpg"),
        "c3": ("P2207T03241A0/P2207T03241A0C3F12181.jpg", "P2207T03241A0C3F12181.jpg"),
    },
]

CAMERA_CONFIGS = [
    {
        "key": "c0",
        "name": "Aerial UAV View (C0)",
        "desc": "Top-Down Pitch (60°–90°) | 15–50m Altitude | Foreshortened Stature",
        "header_bg": "#1E3A8A",
        "card_bg": "#EFF6FF",
        "border": "#3B82F6",
        "badge_text": "#93C5FD",
    },
    {
        "key": "c2",
        "name": "Wearable-Camera View (C2)",
        "desc": "Smart Glasses | Egocentric Eye-Level (1.6m) | Walking Motion Dynamic",
        "header_bg": "#581C87",
        "card_bg": "#FAF5FF",
        "border": "#A855F7",
        "badge_text": "#D8B4FE",
    },
    {
        "key": "c3",
        "name": "Ground CCTV View (C3)",
        "desc": "Fixed Infrastructure Surveillance | Elevated (3–5m) | Full-Body Profile",
        "header_bg": "#065F46",
        "card_bg": "#ECFDF5",
        "border": "#10B981",
        "badge_text": "#A7F3D0",
    },
]


def load_and_pad_image(rel_path: str, target_size=(256, 128)) -> Image.Image:
    """Load authentic dataset image and letterbox-pad to standard 2:1 aspect ratio."""
    full_path = DATASET_BASE / rel_path
    if not full_path.is_file():
        raise FileNotFoundError(f"Dataset image not found: {full_path}")
    
    img = Image.open(full_path).convert("RGB")
    target_h, target_w = target_size
    img_w, img_h = img.size

    # Maintain original aspect ratio and center on dark neutral background
    scale = min(target_w / img_w, target_h / img_h)
    new_w = max(1, int(img_w * scale))
    new_h = max(1, int(img_h * scale))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (target_w, target_h), (24, 28, 36))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))
    return canvas


def build_figure():
    fig = plt.figure(figsize=(15.5, 12.0), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    # Title & Subtitle
    plt.suptitle("AG-ReID.v2 BENCHMARK: CROSS-VIEW SURVEILLANCE PLATFORM EXAMPLES",
                 fontsize=14, fontweight="bold", y=0.975, color="#0F172A")
    plt.figtext(0.5, 0.952, "Authorised Person Re-Identification Bounding Crops across Heterogeneous Sensor Modalities (Preserving Original Release Labels)",
               ha="center", fontsize=9.5, color="#64748B")

    # Layout dimensions
    num_rows = len(IDENTITIES)
    num_cols = len(CAMERA_CONFIGS)

    # Top banner headers for the 3 camera views
    col_x_starts = [0.18, 0.46, 0.74]
    col_width = 0.245

    for c_idx, cfg in enumerate(CAMERA_CONFIGS):
        x = col_x_starts[c_idx]
        # Header box
        hdr_ax = fig.add_axes([x, 0.885, col_width, 0.052])
        hdr_ax.axis("off")
        hdr_box = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.03,rounding_size=0.15",
                                 facecolor=cfg["header_bg"], edgecolor=cfg["border"], lw=1.2)
        hdr_ax.add_patch(hdr_box)
        hdr_ax.text(0.5, 0.62, cfg["name"], ha="center", va="center", fontsize=10.5, fontweight="bold", color="#FFFFFF")
        hdr_ax.text(0.5, 0.25, cfg["desc"], ha="center", va="center", fontsize=6.8, color=cfg["badge_text"])

    # Rows for each identity
    row_y_starts = [0.615, 0.355, 0.095]
    row_height = 0.240

    for r_idx, ident in enumerate(IDENTITIES):
        y = row_y_starts[r_idx]

        # Left label panel for Identity
        id_ax = fig.add_axes([0.02, y, 0.135, row_height])
        id_ax.axis("off")
        id_box = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.04,rounding_size=0.1",
                                facecolor="#F8FAFC", edgecolor="#CBD5E1", lw=1.2)
        id_ax.add_patch(id_box)
        id_ax.text(0.5, 0.65, ident["id_label"], ha="center", va="center", fontsize=11, fontweight="bold", color="#1E293B")
        id_ax.text(0.5, 0.40, ident["tracklet"], ha="center", va="center", fontsize=7.2, color="#64748B", multialignment="center")
        id_ax.text(0.5, 0.18, "Same Identity Probe", ha="center", va="center", fontsize=7.5, fontweight="semibold", color="#059669")

        # 3 Camera images for this identity
        for c_idx, cfg in enumerate(CAMERA_CONFIGS):
            x = col_x_starts[c_idx]
            rel_path, orig_filename = ident[cfg["key"]]

            # Card background
            card_ax = fig.add_axes([x, y, col_width, row_height])
            card_ax.axis("off")
            card_box = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.03,rounding_size=0.1",
                                      facecolor=cfg["card_bg"], edgecolor=cfg["border"], lw=1.2)
            card_ax.add_patch(card_box)

            # Image axis inside card
            img_w_norm = 0.48
            img_h_norm = 0.72
            img_x_norm = (1.0 - img_w_norm) / 2
            img_y_norm = 0.22

            img_ax = fig.add_axes([x + img_x_norm * col_width, y + img_y_norm * row_height,
                                   img_w_norm * col_width, img_h_norm * row_height])
            padded_img = load_and_pad_image(rel_path)
            img_ax.imshow(padded_img)
            img_ax.axis("off")
            for spine in img_ax.spines.values():
                spine.set_visible(True)
                spine.set_color(cfg["border"])
                spine.set_linewidth(1.0)

            # Original filename label below image
            card_ax.text(0.5, 0.12, orig_filename, ha="center", va="center",
                         fontsize=7.2, fontfamily="monospace", fontweight="bold", color="#1E293B")
            card_ax.text(0.5, 0.045, f"Camera: {cfg['name'].split()[0]} ({cfg['key'].upper()})",
                         ha="center", va="center", fontsize=6.8, color="#475569")

    # Bottom caption bar
    cap_text = (
        "Fig. 3. Authorised benchmark examples from the official AG-ReID.v2 dataset release across three heterogeneous surveillance platforms: "
        "Aerial UAV (C0, top-down pitch 60°–90°), Wearable Smart Glasses (C2, egocentric eye-level walking perspective), and Ground CCTV (C3, static surveillance profile). "
        "Each horizontal row illustrates the severe cross-view domain gap for the exact same pedestrian identity, while original source release labels are preserved."
    )
    plt.figtext(0.5, 0.025, cap_text, ha="center", fontsize=8.2, color="#334155",
                wrap=True, style="italic")

    plt.savefig(OUTPUT_PATH, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"✓ Publication-grade Figure 3 generated at: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_figure()
