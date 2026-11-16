"""Extract and Save Multi-Scale Fused Feature Representations.

Extracts:
- fused_feats   : (N, 512) Unified fused feature vectors
- fine_feats    : (N, 256) Original fine pooled vectors
- medium_feats  : (N, 512) Original medium pooled vectors
- coarse_feats  : (N, 2048) Original coarse pooled vectors
- proj_fine     : (N, 512) Projected fine vectors
- proj_medium   : (N, 512) Projected medium vectors
- proj_coarse   : (N, 512) Projected coarse vectors
- pids, cams, paths

Saves to:
features/fused/{fusion_strategy}_{split}_features.pt
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
from src.models.multiscale_fusion_model import MultiScaleFusionModel


def extract_fused_features_from_loader(
    model: MultiScaleFusionModel,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Extract fused and individual multi-scale features."""
    model.eval()

    fused_list = []
    fine_list = []
    med_list = []
    coarse_list = []
    p_fine_list = []
    p_med_list = []
    p_coarse_list = []
    pid_list = []
    cam_list = []
    path_list = []

    with torch.no_grad():
        for imgs, pids, cams, _, paths in tqdm(dataloader, desc="Extracting Fused Features"):
            imgs = imgs.to(device)

            # Feature maps
            feat_maps = model.backbone(imgs)
            # Spatial pooling
            pooled_dict = model.pooler(feat_maps)
            # Projection
            proj_dict = model.projector(pooled_dict)
            # Fusion
            fused_feat, _ = model.fusion(proj_dict)
            # BNNeck & Normalization for evaluation
            bn_feat = model.bnneck(fused_feat)
            norm_fused = torch.nn.functional.normalize(bn_feat, p=2, dim=1)

            fused_list.append(norm_fused.cpu())
            fine_list.append(pooled_dict["fine_vec"].cpu())
            med_list.append(pooled_dict["medium_vec"].cpu())
            coarse_list.append(pooled_dict["coarse_vec"].cpu())
            p_fine_list.append(proj_dict["proj_fine"].cpu())
            p_med_list.append(proj_dict["proj_medium"].cpu())
            p_coarse_list.append(proj_dict["proj_coarse"].cpu())

            pid_list.extend(pids.numpy())
            cam_list.extend(cams.numpy())
            path_list.extend(paths)

    return {
        "fused_feats": torch.cat(fused_list, dim=0),
        "fine_feats": torch.cat(fine_list, dim=0),
        "medium_feats": torch.cat(med_list, dim=0),
        "coarse_feats": torch.cat(coarse_list, dim=0),
        "proj_fine": torch.cat(p_fine_list, dim=0),
        "proj_medium": torch.cat(p_med_list, dim=0),
        "proj_coarse": torch.cat(p_coarse_list, dim=0),
        "pids": torch.tensor(pid_list, dtype=torch.long),
        "cams": torch.tensor(cam_list, dtype=torch.long),
        "paths": path_list,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract Multi-Scale Fused Features")
    parser.add_argument("--dataset_root", type=str, default="datasets/AG-ReID.v2")
    parser.add_argument("--checkpoint", type=str, default="results/checkpoints/best_model.pth")
    parser.add_argument("--split", type=str, default="query", choices=["train_all", "query", "gallery"])
    parser.add_argument("--fusion_strategy", type=str, default="weighted", choices=["weighted", "concat"])
    parser.add_argument("--output_dir", type=str, default="features/fused")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    model = MultiScaleFusionModel(
        num_classes=590,
        fusion_strategy=args.fusion_strategy,
        common_dim=512,
        fused_dim=512,
        pretrained=True,
        checkpoint_path=args.checkpoint if os.path.exists(args.checkpoint) else None,
    ).to(device)

    transform = build_transforms(is_train=False)
    ds = AGReIDDataset(args.dataset_root, split=args.split, transform=transform)
    if args.max_samples is not None:
        ds.samples = ds.samples[:args.max_samples]

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    data_dict = extract_fused_features_from_loader(model, loader, device)

    out_file = os.path.join(args.output_dir, f"{args.fusion_strategy}_{args.split}_features.pt")
    torch.save(data_dict, out_file)
    print(f"\n✓ Saved {len(ds)} fused feature entries to: {out_file}")
    print(f"  - Fused Feature Shape   : {list(data_dict['fused_feats'].shape)}")
    print(f"  - Proj Fine Shape       : {list(data_dict['proj_fine'].shape)}")
    print(f"  - Proj Medium Shape     : {list(data_dict['proj_medium'].shape)}")
    print(f"  - Proj Coarse Shape     : {list(data_dict['proj_coarse'].shape)}")


if __name__ == "__main__":
    main()

