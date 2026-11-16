"""Generate Top-10 Retrieval Visualizations for the Full Proposed Model."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from torch.utils.data import DataLoader

from src.data.dataset import AGReIDDataset
from src.data.transforms import build_transforms
from src.models.view_aware_fusion_model import ViewAwareMultiScaleReIDModel
from src.evaluation.evaluator import extract_features
from src.visualize_retrieval import visualize_topk


def main():
    parser = argparse.ArgumentParser(description="Visualize Retrieval Results for Proposed Model")
    parser.add_argument("--dataset_root", type=str, default="datasets/AG-ReID.v2")
    parser.add_argument("--checkpoint", type=str, default="results/checkpoints/proposed_full_best.pth")
    parser.add_argument("--out_dir", type=str, default="results/retrieval_examples_proposed")
    parser.add_argument("--num_examples", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    eval_tf = build_transforms(img_size=(256, 128), is_train=False, use_enhancement=True)

    query_set = AGReIDDataset(args.dataset_root, split="query", transform=eval_tf)
    gallery_set = AGReIDDataset(args.dataset_root, split="gallery", transform=eval_tf)
    query_loader = DataLoader(query_set, batch_size=64, shuffle=False, num_workers=0)
    gallery_loader = DataLoader(gallery_set, batch_size=64, shuffle=False, num_workers=0)

    model = ViewAwareMultiScaleReIDModel(
        num_classes=590,
        pretrained=False,
        use_view_aware=True,
        use_diffusion=True,
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    q_feats, q_pids, q_cams, q_paths, _ = extract_features(model, query_loader, device)
    g_feats, g_pids, g_cams, g_paths, _ = extract_features(model, gallery_loader, device)

    # Pick exact same diverse camera queries as visualize_retrieval.py
    sample_indices = []
    cams_found = set()
    for idx, c in enumerate(q_cams):
        if c not in cams_found:
            sample_indices.append(idx)
            cams_found.add(c)
        if len(sample_indices) >= args.num_examples:
            break
    for idx in range(len(q_cams)):
        if len(sample_indices) >= args.num_examples:
            break
        if idx not in sample_indices:
            sample_indices.append(idx)

    os.makedirs(args.out_dir, exist_ok=True)
    for q_idx in sample_indices:
        out_f = visualize_topk(
            q_idx, q_feats, q_pids, q_cams, q_paths,
            g_feats, g_pids, g_cams, g_paths,
            topk=10, out_dir=args.out_dir,
        )
        print(f"Saved proposed retrieval visualization: {out_f}")


if __name__ == "__main__":
    main()

