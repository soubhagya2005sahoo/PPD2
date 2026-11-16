"""Comprehensive Cross-View Person Re-ID Evaluator for AG-ReID.v2.

Computes:
1. Overall Re-ID Metrics: Rank-1, Rank-5, Rank-10, mAP, and mINP.
2. Cross-View Breakdown:
   - Aerial (C0) -> Ground (C2, C3)
   - Ground (C2, C3) -> Aerial (C0)
3. Model Efficiency Metrics:
   - Parameter count (Millions), inference throughput (img/sec), latency (ms/img).
"""

import os
import sys
import time
import inspect
import json
from typing import Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import AGReIDDataset
from src.data.transforms import build_transforms
from src.models.resnet50_baseline import ResNet50Baseline
from src.models.view_aware_fusion_model import ViewAwareMultiScaleReIDModel
from tests.test_metrics import eval_reid_metrics


def _accepts_cam_ids(model: torch.nn.Module) -> bool:
    sig = inspect.signature(model.forward)
    return "cam_ids" in sig.parameters


def extract_features(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list, float]:
    """Extract L2-normalized embeddings and measure inference time."""
    model.eval()
    pass_cam = _accepts_cam_ids(model)
    features = []
    pids = []
    cams = []
    img_paths = []

    total_forward_time = 0.0
    total_images = 0

    with torch.no_grad():
        for imgs, pid_labels, cam_ids, _, paths in tqdm(dataloader, desc="Extracting Features", leave=False):
            imgs = imgs.to(device)
            cam_tensor = cam_ids.to(device) if pass_cam else None

            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            feats = model(imgs, cam_ids=cam_tensor) if pass_cam else model(imgs)

            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
            total_forward_time += time.perf_counter() - t0
            total_images += imgs.shape[0]

            features.append(feats.cpu().numpy())
            pids.extend(pid_labels.numpy())
            cams.extend(cam_ids.numpy())
            img_paths.extend(paths)

    features = np.concatenate(features, axis=0)
    pids = np.asarray(pids)
    cams = np.asarray(cams)
    ms_per_img = (total_forward_time / max(1, total_images)) * 1000.0
    return features, pids, cams, img_paths, ms_per_img


def compute_cosine_dist_matrix(q_feats: np.ndarray, g_feats: np.ndarray, chunk_size: int = 500) -> np.ndarray:
    """Compute pairwise cosine distance matrix in memory-safe chunks."""
    num_q = q_feats.shape[0]
    num_g = g_feats.shape[0]
    dist_matrix = np.zeros((num_q, num_g), dtype=np.float32)
    for i in range(0, num_q, chunk_size):
        end = min(i + chunk_size, num_q)
        sim = np.matmul(q_feats[i:end], g_feats.T)
        dist_matrix[i:end] = 1.0 - sim
    return dist_matrix


def _eval_subset(
    dist_matrix: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_cams: np.ndarray,
    g_cams: np.ndarray,
    q_mask: np.ndarray,
    g_mask: np.ndarray,
    max_rank: int = 10,
) -> Dict[str, float]:
    sub_dist = dist_matrix[np.ix_(q_mask, g_mask)]
    sub_q_pids = q_pids[q_mask]
    sub_g_pids = g_pids[g_mask]
    sub_q_cams = q_cams[q_mask]
    sub_g_cams = g_cams[g_mask]
    cmc, m_ap, m_inp = eval_reid_metrics(
        sub_dist, sub_q_pids, sub_g_pids, sub_q_cams, sub_g_cams, max_rank=max_rank, return_minp=True
    )
    return {
        "Rank-1": float(cmc[0] * 100.0),
        "Rank-5": float(cmc[min(4, len(cmc) - 1)] * 100.0),
        "Rank-10": float(cmc[min(9, len(cmc) - 1)] * 100.0),
        "mAP": float(m_ap * 100.0),
        "mINP": float(m_inp * 100.0),
    }


def evaluate_comprehensive(
    model: torch.nn.Module,
    query_loader: DataLoader,
    gallery_loader: DataLoader,
    device: torch.device,
    max_rank: int = 10,
    exp_name: str = "Model",
) -> Dict[str, object]:
    """Run complete Overall + Aerial<->Ground + Efficiency evaluation."""
    print(f"\n[{exp_name}] Extracting Query Features...")
    q_feats, q_pids, q_cams, _, q_ms = extract_features(model, query_loader, device)

    print(f"[{exp_name}] Extracting Gallery Features...")
    g_feats, g_pids, g_cams, _, g_ms = extract_features(model, gallery_loader, device)

    avg_ms_per_img = 0.5 * (q_ms + g_ms)
    fps = 1000.0 / max(1e-6, avg_ms_per_img)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"[{exp_name}] Computing Cosine Distance Matrix ({q_feats.shape[0]} x {g_feats.shape[0]})...")
    dist_matrix = compute_cosine_dist_matrix(q_feats, g_feats)

    # 1. Overall
    cmc, m_ap, m_inp = eval_reid_metrics(
        dist_matrix, q_pids, g_pids, q_cams, g_cams, max_rank=max_rank, return_minp=True
    )
    overall = {
        "Rank-1": float(cmc[0] * 100.0),
        "Rank-5": float(cmc[min(4, len(cmc) - 1)] * 100.0),
        "Rank-10": float(cmc[min(9, len(cmc) - 1)] * 100.0),
        "mAP": float(m_ap * 100.0),
        "mINP": float(m_inp * 100.0),
    }

    # 2. Aerial (C0) -> Ground (C2, C3)
    q_aerial = q_cams == 0
    g_ground = (g_cams == 2) | (g_cams == 3)
    aerial_to_ground = _eval_subset(
        dist_matrix, q_pids, g_pids, q_cams, g_cams, q_aerial, g_ground, max_rank=max_rank
    )

    # 3. Ground (C2, C3) -> Aerial (C0)
    q_ground = (q_cams == 2) | (q_cams == 3)
    g_aerial = g_cams == 0
    ground_to_aerial = _eval_subset(
        dist_matrix, q_pids, g_pids, q_cams, g_cams, q_ground, g_aerial, max_rank=max_rank
    )

    summary = {
        "experiment": exp_name,
        "overall": overall,
        "aerial_to_ground": aerial_to_ground,
        "ground_to_aerial": ground_to_aerial,
        "efficiency": {
            "params_millions": round(total_params / 1e6, 2),
            "ms_per_image": round(avg_ms_per_img, 3),
            "fps": round(fps, 1),
            "embedding_dim": int(q_feats.shape[1]),
        },
    }

    print("\n" + "=" * 68)
    print(f"EVALUATION REPORT: {exp_name}")
    print("=" * 68)
    print(
        f"  Overall           | Rank-1: {overall['Rank-1']:5.2f}% | Rank-5: {overall['Rank-5']:5.2f}% | "
        f"mAP: {overall['mAP']:5.2f}% | mINP: {overall['mINP']:5.2f}%"
    )
    print(
        f"  Aerial -> Ground  | Rank-1: {aerial_to_ground['Rank-1']:5.2f}% | "
        f"mAP: {aerial_to_ground['mAP']:5.2f}% | mINP: {aerial_to_ground['mINP']:5.2f}%"
    )
    print(
        f"  Ground -> Aerial  | Rank-1: {ground_to_aerial['Rank-1']:5.2f}% | "
        f"mAP: {ground_to_aerial['mAP']:5.2f}% | mINP: {ground_to_aerial['mINP']:5.2f}%"
    )
    print(
        f"  Efficiency        | Params: {summary['efficiency']['params_millions']}M | "
        f"Speed: {summary['efficiency']['ms_per_image']:.2f} ms/img ({summary['efficiency']['fps']:.1f} FPS)"
    )
    print("=" * 68)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Evaluator on AG-ReID.v2")
    parser.add_argument("--dataset_root", type=str, default="datasets/AG-ReID.v2")
    parser.add_argument("--model_type", type=str, default="baseline", choices=["baseline", "proposed"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="Baseline_ResNet50")
    parser.add_argument("--use_enhancement", action="store_true")
    parser.add_argument("--use_view_aware", action="store_true")
    parser.add_argument("--use_diffusion", action="store_true")
    parser.add_argument("--fusion_strategy", type=str, default="weighted")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--save_json", type=str, default="")
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    eval_transform = build_transforms(
        img_size=(256, 128),
        is_train=False,
        use_enhancement=args.use_enhancement,
    )
    query_set = AGReIDDataset(args.dataset_root, split="query", transform=eval_transform)
    gallery_set = AGReIDDataset(args.dataset_root, split="gallery", transform=eval_transform)
    query_loader = DataLoader(query_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    gallery_loader = DataLoader(gallery_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if args.model_type == "baseline":
        model = ResNet50Baseline(num_classes=590, pretrained=False)
    else:
        model = ViewAwareMultiScaleReIDModel(
            num_classes=590,
            pretrained=False,
            use_view_aware=args.use_view_aware,
            use_diffusion=args.use_diffusion,
            fusion_strategy=args.fusion_strategy,
        )

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)

    summary = evaluate_comprehensive(model, query_loader, gallery_loader, device, exp_name=args.exp_name)

    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved evaluation JSON to {args.save_json}")


if __name__ == "__main__":
    main()

