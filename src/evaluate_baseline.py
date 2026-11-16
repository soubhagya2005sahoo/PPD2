"""Evaluation Script for Person Re-ID on AG-ReID.v2.

Extracts normalized 2048-dim feature representations from query and gallery
sets, computes pairwise cosine distances, and calculates standard Re-ID metrics:
- Rank-1, Rank-5, Rank-10 CMC accuracies.
- Mean Average Precision (mAP).
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from typing import Dict, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import AGReIDDataset
from src.data.transforms import build_transforms
from src.models.resnet50_baseline import ResNet50Baseline
from tests.test_metrics import eval_reid_metrics


def extract_features(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Extract L2-normalized feature embeddings from a dataloader."""
    model.eval()
    features = []
    pids = []
    cams = []
    img_paths = []

    with torch.no_grad():
        for imgs, pid_labels, cam_ids, _, paths in tqdm(dataloader, desc="Extracting Features", leave=False):
            imgs = imgs.to(device)
            feats = model(imgs)  # During eval, returns normalized 2048-dim tensor
            features.append(feats.cpu().numpy())
            pids.extend(pid_labels.numpy())
            cams.extend(cam_ids.numpy())
            img_paths.extend(paths)

    features = np.concatenate(features, axis=0)
    pids = np.asarray(pids)
    cams = np.asarray(cams)
    return features, pids, cams, img_paths


def evaluate(
    model: torch.nn.Module,
    query_loader: DataLoader,
    gallery_loader: DataLoader,
    device: torch.device,
    max_rank: int = 10,
) -> Dict[str, float]:
    """Run full query-to-gallery retrieval evaluation."""
    print("\nExtracting Query Features...")
    q_feats, q_pids, q_cams, _ = extract_features(model, query_loader, device)

    print("Extracting Gallery Features...")
    g_feats, g_pids, g_cams, _ = extract_features(model, gallery_loader, device)

    print(f"Features extracted: Query={q_feats.shape}, Gallery={g_feats.shape}")
    print("Computing Cosine Distance Matrix...")

    # For L2-normalized features, cosine similarity is dot product;
    # cosine distance is 1.0 - dot_product.
    # We compute in chunks if gallery is large to preserve memory.
    num_q = q_feats.shape[0]
    num_g = g_feats.shape[0]
    dist_matrix = np.zeros((num_q, num_g), dtype=np.float32)

    chunk_size = 500
    for i in range(0, num_q, chunk_size):
        end = min(i + chunk_size, num_q)
        # (chunk, dim) @ (dim, num_g) -> (chunk, num_g)
        sim = np.matmul(q_feats[i:end], g_feats.T)
        dist_matrix[i:end] = 1.0 - sim

    print("Calculating CMC, mAP, and mINP metrics...")
    cmc, m_ap, m_inp = eval_reid_metrics(
        dist_matrix, q_pids, g_pids, q_cams, g_cams, max_rank=max_rank, return_minp=True
    )

    results = {
        "mAP": m_ap * 100.0,
        "mINP": m_inp * 100.0,
        "Rank-1": float(cmc[0] * 100.0),
        "Rank-5": float(cmc[min(4, len(cmc) - 1)] * 100.0),
        "Rank-10": float(cmc[min(9, len(cmc) - 1)] * 100.0),
    }

    print("\n" + "=" * 50)
    print("AG-ReID.v2 BASELINE EVALUATION RESULTS")
    print("=" * 50)
    print(f"  mAP     : {results['mAP']:.2f}%")
    print(f"  mINP    : {results['mINP']:.2f}%")
    print(f"  Rank-1  : {results['Rank-1']:.2f}%")
    print(f"  Rank-5  : {results['Rank-5']:.2f}%")
    print(f"  Rank-10 : {results['Rank-10']:.2f}%")
    print("=" * 50)
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate ResNet-50 Baseline on AG-ReID.v2")
    parser.add_argument("--dataset_root", type=str, default="datasets/AG-ReID.v2")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model weights checkpoint")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Evaluating on device: {device}")

    # Build transforms & datasets
    eval_transform = build_transforms(is_train=False)
    query_set = AGReIDDataset(args.dataset_root, split="query", transform=eval_transform)
    gallery_set = AGReIDDataset(args.dataset_root, split="gallery", transform=eval_transform)

    query_loader = DataLoader(query_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    gallery_loader = DataLoader(gallery_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Initialize model
    model = ResNet50Baseline(num_classes=590, pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    evaluate(model, query_loader, gallery_loader, device)


if __name__ == "__main__":
    main()
