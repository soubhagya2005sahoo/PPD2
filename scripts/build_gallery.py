"""Extract and Precompute Gallery Feature Embeddings for AG-ReID.v2.

Uses the trained Proposed Model (View-Aware Multi-Scale + Latent Diffusion)
to extract 512-dim L2-normalized embeddings for all 40,473 gallery images.
Saves cache to: features/gallery_cache_proposed.npz
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import AGReIDDataset
from src.data.transforms import build_transforms
from src.models.view_aware_fusion_model import ViewAwareMultiScaleReIDModel


def build_gallery_cache(
    dataset_root: str = "datasets/AG-ReID.v2",
    checkpoint_path: str = "results/checkpoints/proposed_full_best.pth",
    output_cache: str = "features/gallery_cache_proposed.npz",
    batch_size: int = 64,
):
    print("=" * 70)
    print("AG-ReID.v2 Gallery Feature Cache Generator")
    print("=" * 70)

    # Determine hardware device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Compute Device: Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Compute Device: NVIDIA CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("Compute Device: CPU")

    checkpoint_file = PROJECT_ROOT / checkpoint_path
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_file}")

    # Build evaluation transforms with on-the-fly CLAHE + Gamma enhancement
    eval_transform = build_transforms(img_size=(256, 128), is_train=False, use_enhancement=True)
    gallery_dataset = AGReIDDataset(str(PROJECT_ROOT / dataset_root), split="gallery", transform=eval_transform)
    gallery_loader = DataLoader(gallery_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Loaded Gallery Dataset: {len(gallery_dataset):,} images")

    # Load proposed model
    model = ViewAwareMultiScaleReIDModel(
        num_classes=590,
        pretrained=False,
        use_view_aware=True,
        use_diffusion=True,
    )
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("✓ Model weights loaded successfully.")

    features = []
    pids = []
    cams = []
    paths = []
    rel_paths = []

    t0 = time.time()
    with torch.no_grad():
        for imgs, pid_labels, cam_ids, _, img_paths in tqdm(gallery_loader, desc="Extracting Gallery Embeddings"):
            imgs = imgs.to(device)
            cam_tensor = cam_ids.to(device)

            # Extract 512-dim L2-normalized embeddings
            feats = model(imgs, cam_ids=cam_tensor)
            features.append(feats.cpu().numpy())
            pids.append(pid_labels.numpy())
            cams.append(cam_ids.numpy())

            for p in img_paths:
                rel = os.path.relpath(p, str(PROJECT_ROOT))
                paths.append(rel)
                # AG-ReID relative path for clean retrieval
                rel_parts = rel.split("datasets/")
                rel_paths.append(rel_parts[-1] if len(rel_parts) > 1 else rel)

    features = np.concatenate(features, axis=0).astype(np.float32)
    pids = np.concatenate(pids, axis=0).astype(np.int64)
    cams = np.concatenate(cams, axis=0).astype(np.int64)
    paths = np.array(paths)
    rel_paths = np.array(rel_paths)

    elapsed = time.time() - t0
    fps = len(gallery_dataset) / elapsed
    print(f"\nExtracted {len(features):,} features in {elapsed:.1f}s ({fps:.1f} FPS)")

    # Validation
    assert not np.isnan(features).any(), "NaN found in gallery features!"
    assert not np.isinf(features).any(), "Inf found in gallery features!"
    norms = np.linalg.norm(features, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), "Embeddings are not unit L2 normalized!"

    # Save to disk
    out_file = PROJECT_ROOT / output_cache
    os.makedirs(out_file.parent, exist_ok=True)
    np.savez_compressed(
        out_file,
        feats=features,
        pids=pids,
        cams=cams,
        paths=paths,
        rel_paths=rel_paths,
    )
    print(f"✓ Gallery cache saved to: {out_file} ({out_file.stat().st_size / (1024*1024):.1f} MB)")
    print("=" * 70)


if __name__ == "__main__":
    build_gallery_cache()

