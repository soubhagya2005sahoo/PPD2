"""Interactive Inference Script for Aerial-Ground Person Re-Identification.

Supports two intuitive modes:
1. Pair Comparison Mode (--img1 <path> --img2 <path>):
   Compares two person images, computes cosine similarity, predicts whether
   they show the SAME PERSON or DIFFERENT PERSONS, and saves a visual card.

2. Query Search Mode (--query <path>):
   Takes a single person image and finds the Top-K closest matches from
   the gallery (or any folder specified via --gallery_dir), saving a ranked
   visual output image. Uses disk caching for instant repeat searches.
"""

import os
import sys
import re
import argparse
from typing import Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import AGReIDDataset
from src.data.transforms import build_transforms
from src.enhancement.image_enhancer import ImageEnhancer
from src.models.view_aware_fusion_model import ViewAwareMultiScaleReIDModel


VIEW_ARG_TO_CAM = {
    "auto": None,
    "aerial": 0,
    "uav": 0,
    "c0": 0,
    "wearable": 2,
    "c2": 2,
    "cctv": 3,
    "ground": 3,
    "c3": 3,
}
CAM_NAMES = {0: "Aerial (C0)", 2: "Wearable (C2)", 3: "CCTV (C3)"}
CAM_REGEX = re.compile(r"C(?P<cam>\d+)F", re.IGNORECASE)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def infer_cam_id(img_path: str, view_arg: str = "auto") -> Optional[int]:
    if view_arg.lower() != "auto":
        return VIEW_ARG_TO_CAM.get(view_arg.lower(), None)
    match = CAM_REGEX.search(os.path.basename(img_path))
    if match:
        cam_val = int(match.group("cam"))
        if cam_val in (0, 2, 3):
            return cam_val
    return None


def load_proposed_model(checkpoint_path: str, device: torch.device) -> ViewAwareMultiScaleReIDModel:
    model = ViewAwareMultiScaleReIDModel(
        num_classes=590,
        pretrained=False,
        use_view_aware=True,
        use_diffusion=True,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_single_embedding(
    model: ViewAwareMultiScaleReIDModel,
    img_path: str,
    transform,
    device: torch.device,
    cam_id: Optional[int] = None,
) -> Tuple[np.ndarray, str]:
    with Image.open(img_path) as img:
        img_rgb = img.convert("RGB")
    tensor = transform(img_rgb).unsqueeze(0).to(device)

    if cam_id is not None:
        cam_tensor = torch.tensor([cam_id], dtype=torch.long, device=device)
        detected_view = CAM_NAMES.get(cam_id, f"C{cam_id}")
    else:
        cam_tensor = None
        # Also detect view name from internal view classifier for transparency
        feat_maps = model.backbone(tensor)
        pooled = model.pooler(feat_maps)
        proj = model.projector(pooled)
        view_logits = model.view_aware.view_classifier(proj["proj_coarse"])
        pred_v = int(torch.argmax(view_logits, dim=1).item())
        idx_to_name = {0: "Predicted: Aerial", 1: "Predicted: Wearable", 2: "Predicted: CCTV"}
        detected_view = idx_to_name.get(pred_v, "Auto")

    emb = model(tensor, cam_ids=cam_tensor).squeeze(0).cpu().numpy()
    return emb, detected_view


def run_pair_comparison(
    model: ViewAwareMultiScaleReIDModel,
    img1_path: str,
    img2_path: str,
    view1: str,
    view2: str,
    threshold: float,
    out_path: str,
    device: torch.device,
):
    eval_tf = build_transforms(img_size=(256, 128), is_train=False, use_enhancement=True)
    enhancer = ImageEnhancer(mode="clahe_gamma", clip_limit=2.0, gamma=1.2)

    cam1 = infer_cam_id(img1_path, view1)
    cam2 = infer_cam_id(img2_path, view2)

    emb1, view_label1 = extract_single_embedding(model, img1_path, eval_tf, device, cam_id=cam1)
    emb2, view_label2 = extract_single_embedding(model, img2_path, eval_tf, device, cam_id=cam2)

    similarity = float(np.dot(emb1, emb2))
    is_same = similarity >= threshold
    verdict = "SAME PERSON" if is_same else "DIFFERENT PERSONS"
    color = "#16A34A" if is_same else "#DC2626"

    print("\n" + "=" * 60)
    print("PERSON RE-ID PAIR COMPARISON RESULT")
    print("=" * 60)
    print(f"  Image 1          : {img1_path} ({view_label1})")
    print(f"  Image 2          : {img2_path} ({view_label2})")
    print(f"  Cosine Similarity: {similarity:.4f} ({similarity * 100:.1f}%)")
    print(f"  Threshold        : {threshold:.2f}")
    print(f"  Prediction       : {verdict}")
    print("=" * 60)

    # Save visual comparison
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(11, 4.2), dpi=160)
    with Image.open(img1_path) as im1, Image.open(img2_path) as im2:
        im1_rgb = im1.convert("RGB")
        im2_rgb = im2.convert("RGB")
        im1_enh = enhancer.enhance_pil(im1_rgb)
        im2_enh = enhancer.enhance_pil(im2_rgb)

    axes[0].imshow(im1_rgb)
    axes[0].set_title(f"Input 1 (Raw)\n{view_label1}", fontsize=9)
    axes[1].imshow(im1_enh)
    axes[1].set_title("Input 1 (Enhanced)", fontsize=9)
    axes[2].imshow(im2_rgb)
    axes[2].set_title(f"Input 2 (Raw)\n{view_label2}", fontsize=9)
    axes[3].imshow(im2_enh)
    axes[3].set_title("Input 2 (Enhanced)", fontsize=9)

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"Prediction: {verdict}  |  Cosine Similarity: {similarity:.4f} ({similarity * 100:.1f}%)",
        fontsize=12,
        fontweight="bold",
        color=color,
    )
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Visual output saved to: {out_path}")


def load_or_build_gallery(
    model: ViewAwareMultiScaleReIDModel,
    dataset_root: str,
    gallery_dir: Optional[str],
    cache_path: str,
    device: torch.device,
) -> Tuple[np.ndarray, list, np.ndarray]:
    eval_tf = build_transforms(img_size=(256, 128), is_train=False, use_enhancement=True)

    # Custom user gallery folder
    if gallery_dir is not None:
        exts = (".jpg", ".jpeg", ".png")
        paths = []
        for root, _, files in os.walk(gallery_dir):
            for f in sorted(files):
                if f.lower().endswith(exts) and not f.startswith("."):
                    paths.append(os.path.join(root, f))
        if not paths:
            raise FileNotFoundError(f"No images found in gallery_dir: {gallery_dir}")

        feats = []
        cams = []
        print(f"Scanning {len(paths)} images from '{gallery_dir}'...")
        for p in tqdm(paths, desc="Encoding Gallery"):
            cam = infer_cam_id(p, "auto")
            emb, _ = extract_single_embedding(model, p, eval_tf, device, cam_id=cam)
            feats.append(emb)
            cams.append(-1 if cam is None else cam)
        return np.stack(feats, axis=0), paths, np.asarray(cams)

    # Default AG-ReID.v2 gallery with disk cache for instant repeat queries
    if os.path.exists(cache_path):
        print(f"Loading cached gallery embeddings from {cache_path}...")
        data = np.load(cache_path, allow_pickle=False)
        return data["feats"], data["paths"].tolist(), data["cams"]

    print("Building AG-ReID.v2 gallery embedding cache (one-time ~3 min step)...")
    gallery_set = AGReIDDataset(dataset_root, split="gallery", transform=eval_tf)
    loader = DataLoader(gallery_set, batch_size=64, shuffle=False, num_workers=0)
    feats, cams, paths = [], [], []
    with torch.no_grad():
        for imgs, _, cam_ids, _, img_paths in tqdm(loader, desc="Caching Gallery"):
            emb = model(imgs.to(device), cam_ids=cam_ids.to(device)).cpu().numpy()
            feats.append(emb)
            cams.extend(cam_ids.numpy())
            paths.extend(img_paths)
    feats_arr = np.concatenate(feats, axis=0)
    cams_arr = np.asarray(cams)

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.savez(cache_path, feats=feats_arr, paths=np.asarray(paths), cams=cams_arr)
    print(f"Cached {len(paths)} gallery embeddings to {cache_path}")
    return feats_arr, paths, cams_arr


def run_query_search(
    model: ViewAwareMultiScaleReIDModel,
    query_path: str,
    view: str,
    dataset_root: str,
    gallery_dir: Optional[str],
    cache_path: str,
    topk: int,
    out_path: str,
    device: torch.device,
):
    eval_tf = build_transforms(img_size=(256, 128), is_train=False, use_enhancement=True)
    cam_id = infer_cam_id(query_path, view)
    q_emb, q_view_label = extract_single_embedding(model, query_path, eval_tf, device, cam_id=cam_id)

    g_feats, g_paths, g_cams = load_or_build_gallery(
        model, dataset_root, gallery_dir, cache_path, device
    )

    sims = np.matmul(q_emb, g_feats.T)
    order = np.argsort(-sims)[:topk]

    print("\n" + "=" * 65)
    print(f"TOP-{topk} RETRIEVAL RESULTS FOR QUERY: {query_path} ({q_view_label})")
    print("=" * 65)
    for rank, idx in enumerate(order, start=1):
        c_val = int(g_cams[idx])
        c_str = CAM_NAMES.get(c_val, "Unknown") if c_val >= 0 else "Custom"
        print(f"  Rank {rank:2d} | Sim: {sims[idx]:.4f} | {c_str:14s} | {os.path.basename(g_paths[idx])}")
    print("=" * 65)

    # Save ranked visual strip
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, 1 + len(order), figsize=(2.3 * (1 + len(order)), 4.0), dpi=170)

    with Image.open(query_path) as qim:
        axes[0].imshow(qim.convert("RGB"))
    axes[0].set_title(f"QUERY\n{q_view_label}", fontsize=9, fontweight="bold", color="blue")
    axes[0].axis("off")
    axes[0].add_patch(Rectangle((0, 0), 1, 1, transform=axes[0].transAxes, linewidth=4, edgecolor="blue", facecolor="none"))

    for rank, idx in enumerate(order):
        ax = axes[1 + rank]
        with Image.open(g_paths[idx]) as gim:
            ax.imshow(gim.convert("RGB"))
        c_val = int(g_cams[idx])
        c_str = CAM_NAMES.get(c_val, "Custom") if c_val >= 0 else "Custom"
        ax.set_title(
            f"Rank {rank+1}\nSim: {sims[idx]:.3f}\n{c_str}\n{os.path.basename(g_paths[idx])[:14]}",
            fontsize=8,
            fontweight="bold",
        )
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Ranked visual output saved to: {out_path}")


def clean_user_path(raw: str) -> str:
    """Clean paths pasted or dragged-and-dropped into macOS Terminal."""
    p = raw.strip()
    if (p.startswith("'") and p.endswith("'")) or (p.startswith('"') and p.endswith('"')):
        p = p[1:-1]
    else:
        p = p.replace("\\ ", " ")
    return os.path.expanduser(p)


def prompt_for_image_path(prompt_text: str) -> Optional[str]:
    """Repeatedly prompt user until a valid image file is given, or user cancels."""
    while True:
        raw = input(prompt_text).strip()
        if not raw:
            print("  ⚠️  No path entered! Drag an image file from Finder into this window, or paste a valid path (or 'q' to cancel).")
            continue
        if raw.lower() in ("q", "quit", "exit"):
            return None
        cleaned = clean_user_path(raw)
        if not os.path.exists(cleaned):
            print(f"  ❌ File not found: '{cleaned}'\n      Please check the path or drag & drop again (or 'q' to cancel).")
            continue
        if os.path.isdir(cleaned):
            print(f"  ❌ '{cleaned}' is a folder, not an image file. Please provide an image file.")
            continue
        return cleaned


def run_interactive_mode(model: ViewAwareMultiScaleReIDModel, args, device: torch.device):
    """Beginner-friendly interactive prompt supporting macOS drag-and-drop."""
    print("\n" + "=" * 60)
    print("INTERACTIVE PERSON RE-IDENTIFICATION DEMO")
    print("=" * 60)
    print("Tip: You can DRAG AND DROP any image file from Finder/Desktop")
    print("     directly into this Terminal window, then press Enter!")
    print("-" * 60)
    print("  1) Compare TWO person images (Are they the same person?)")
    print("  2) Search ONE person image against a folder or AG-ReID gallery")
    choice = input("\nEnter 1 or 2 (or 'q' to exit): ").strip()
    if choice.lower() in ("q", "quit", "exit"):
        return

    if choice == "1":
        img1 = prompt_for_image_path("\nDrag & drop Image 1 (or type path): ")
        if not img1:
            print("Operation cancelled.")
            return
        v1 = input("Image 1 camera view [auto / aerial / wearable / cctv] (default auto): ").strip() or "auto"

        img2 = prompt_for_image_path("Drag & drop Image 2 (or type path): ")
        if not img2:
            print("Operation cancelled.")
            return
        v2 = input("Image 2 camera view [auto / aerial / wearable / cctv] (default auto): ").strip() or "auto"

        out_path = args.out or "results/custom_pair_result.png"
        run_pair_comparison(model, img1, img2, v1, v2, args.threshold, out_path, device)
    elif choice == "2":
        q_img = prompt_for_image_path("\nDrag & drop Query Image (or type path): ")
        if not q_img:
            print("Operation cancelled.")
            return
        v = input("Query camera view [auto / aerial / wearable / cctv] (default auto): ").strip() or "auto"

        g_raw = input("Drag & drop a Gallery Folder (or press Enter to search full AG-ReID.v2 gallery): ").strip()
        gallery_dir = clean_user_path(g_raw) if g_raw else None
        out_path = args.out or "results/custom_query_result.png"
        run_query_search(
            model, q_img, v, args.dataset_root, gallery_dir, args.cache_path, args.topk, out_path, device
        )
    else:
        print("Invalid choice. Please enter 1 or 2.")


def main():
    parser = argparse.ArgumentParser(description="Single-Image / Pair Inference for Person Re-ID")
    parser.add_argument("--checkpoint", type=str, default="results/checkpoints/proposed_full_best.pth")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive drag-and-drop prompt")

    # Mode 1: Pair comparison
    parser.add_argument("--img1", type=str, default=None, help="First person image path")
    parser.add_argument("--img2", type=str, default=None, help="Second person image path")
    parser.add_argument("--view1", type=str, default="auto", help="auto|aerial|wearable|cctv")
    parser.add_argument("--view2", type=str, default="auto", help="auto|aerial|wearable|cctv")
    parser.add_argument("--threshold", type=float, default=0.40, help="Cosine similarity threshold for Same Person")

    # Mode 2: Query vs Gallery search
    parser.add_argument("--query", type=str, default=None, help="Query image path to search in gallery")
    parser.add_argument("--view", type=str, default="auto", help="auto|aerial|wearable|cctv")
    parser.add_argument("--dataset_root", type=str, default="datasets/AG-ReID.v2")
    parser.add_argument("--gallery_dir", type=str, default=None, help="Optional custom folder of gallery images")
    parser.add_argument("--cache_path", type=str, default="features/gallery_cache_proposed.npz")
    parser.add_argument("--topk", type=int, default=5)

    parser.add_argument("--out", type=str, default=None, help="Output visualization PNG path")
    args = parser.parse_args()

    device = get_device()
    model = load_proposed_model(args.checkpoint, device)

    if args.img1 and args.img2:
        out_path = args.out or "results/custom_pair_result.png"
        run_pair_comparison(
            model, clean_user_path(args.img1), clean_user_path(args.img2),
            args.view1, args.view2, args.threshold, out_path, device
        )
    elif args.query:
        out_path = args.out or "results/custom_query_result.png"
        g_dir = clean_user_path(args.gallery_dir) if args.gallery_dir else None
        run_query_search(
            model, clean_user_path(args.query), args.view, args.dataset_root,
            g_dir, args.cache_path, args.topk, out_path, device
        )
    else:
        run_interactive_mode(model, args, device)


if __name__ == "__main__":
    main()
