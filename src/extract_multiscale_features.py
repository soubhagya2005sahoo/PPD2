"""Extract and Save Multi-Scale Feature Representations.

Extracts three distinct visual scale pooled feature vectors:
- fine_feats   : (N, 256)
- medium_feats : (N, 512)
- coarse_feats : (N, 2048)
along with ground-truth Person IDs, Camera tags, and image paths.

Saves separate feature tensors for train and test partitions under:
features/multiscale/{input_mode}_{split}_features.pt
"""

import os
import sys
import argparse
from typing import Dict
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import AGReIDDataset
from src.data.transforms import build_transforms
from src.models.resnet50_multiscale import ResNet50MultiScale
from src.features.feature_pooling import MultiScaleFeaturePooler


def extract_split_features(
    model: ResNet50MultiScale,
    pooler: MultiScaleFeaturePooler,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Extract fine, medium, and coarse pooled feature vectors."""
    model.eval()
    pooler.eval()

    fine_list = []
    med_list = []
    coarse_list = []
    pid_list = []
    cam_list = []
    path_list = []

    with torch.no_grad():
        for imgs, pids, cams, _, paths in tqdm(dataloader, desc="Extracting Multi-Scale Features"):
            imgs = imgs.to(device)
            feat_maps = model(imgs)
            pooled_vecs = pooler(feat_maps)

            fine_list.append(pooled_vecs["fine_vec"].cpu())
            med_list.append(pooled_vecs["medium_vec"].cpu())
            coarse_list.append(pooled_vecs["coarse_vec"].cpu())

            pid_list.extend(pids.numpy())
            cam_list.extend(cams.numpy())
            path_list.extend(paths)

    return {
        "fine_feats": torch.cat(fine_list, dim=0),
        "medium_feats": torch.cat(med_list, dim=0),
        "coarse_feats": torch.cat(coarse_list, dim=0),
        "pids": torch.tensor(pid_list, dtype=torch.long),
        "cams": torch.tensor(cam_list, dtype=torch.long),
        "paths": path_list,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract Multi-Scale ResNet-50 Features")
    parser.add_argument("--dataset_root", type=str, default="datasets/AG-ReID.v2")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="query", choices=["train_all", "query", "gallery", "all"])
    parser.add_argument("--input_mode", type=str, default="original", choices=["original", "enhanced"])
    parser.add_argument("--output_dir", type=str, default="features/multiscale")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Determine checkpoint
    ckpt = args.checkpoint
    if ckpt is None:
        if os.path.exists("results/checkpoints/best_model.pth"):
            ckpt = "results/checkpoints/best_model.pth"
        elif os.path.exists("results/checkpoints/checkpoint_batch_1200.pth"):
            ckpt = "results/checkpoints/checkpoint_batch_1200.pth"

    # Initialize model & pooler
    model = ResNet50MultiScale(pretrained=True, checkpoint_path=ckpt).to(device)
    pooler = MultiScaleFeaturePooler(l2_normalize=True).to(device)

    # Resolve dataset root
    ds_root = args.dataset_root
    if args.input_mode == "enhanced" and os.path.isdir("datasets/AG-ReID.v2_enhanced"):
        ds_root = "datasets/AG-ReID.v2_enhanced"

    transform = build_transforms(is_train=False)

    splits_to_process = ["query", "gallery"] if args.split == "all" else [args.split]

    for s in splits_to_process:
        print(f"\nProcessing split '{s}' (mode={args.input_mode}) from {ds_root}...")
        ds = AGReIDDataset(ds_root, split=s, transform=transform)
        if args.max_samples is not None:
            ds.samples = ds.samples[:args.max_samples]

        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        data_dict = extract_split_features(model, pooler, loader, device)

        out_file = os.path.join(args.output_dir, f"{args.input_mode}_{s}_features.pt")
        torch.save(data_dict, out_file)

        print(f"  ✓ Saved {len(ds)} feature entries to: {out_file}")
        print(f"    - Fine Vector Shape   : {list(data_dict['fine_feats'].shape)}")
        print(f"    - Medium Vector Shape : {list(data_dict['medium_feats'].shape)}")
        print(f"    - Coarse Vector Shape : {list(data_dict['coarse_feats'].shape)}")


if __name__ == "__main__":
    main()

