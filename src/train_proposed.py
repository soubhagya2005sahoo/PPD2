"""Unified Training Script for Step 7 Experiments on AG-ReID.v2.

Supports controlled ablation and full proposed model training:
- Baseline + CLAHE/Gamma Enhancement
- Multi-Scale Fusion + View-Aware Conditioning + Efficient Latent Diffusion

Computes Rank-1, Rank-5, Rank-10, mAP, mINP, and Aerial<->Ground breakdowns,
and logs per-view scale attention statistics.
"""

import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import AGReIDDataset
from src.data.transforms import build_transforms
from src.data.sampler import RandomIdentitySampler
from src.models.resnet50_baseline import ResNet50Baseline
from src.models.view_aware_fusion_model import ViewAwareMultiScaleReIDModel
from src.losses.triplet_loss import TripletLoss
from src.evaluation.evaluator import evaluate_comprehensive


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model_from_args(args, cfg: dict, num_classes: int) -> nn.Module:
    if args.model_type == "baseline":
        return ResNet50Baseline(
            num_classes=num_classes,
            pretrained=cfg["model"]["pretrained"],
            last_stride=cfg["model"]["last_stride"],
        )
    else:
        return ViewAwareMultiScaleReIDModel(
            num_classes=num_classes,
            common_dim=cfg["model"]["common_dim"],
            fused_dim=cfg["model"]["fused_dim"],
            pretrained=cfg["model"]["pretrained"],
            last_stride=cfg["model"]["last_stride"],
            use_view_aware=args.use_view_aware,
            use_diffusion=args.use_diffusion,
            fusion_strategy=args.fusion_strategy,
        )


def collect_view_scale_weights(model: nn.Module, dataloader: DataLoader, device: torch.device, max_batches: int = 30) -> dict:
    """Measure average fine/medium/coarse scale weights per camera view on eval set."""
    if not getattr(model, "use_view_aware", False) or getattr(model, "view_aware", None) is None:
        return {}

    model.eval()
    view_weights = {0: [], 2: [], 3: []}
    with torch.no_grad():
        for b_idx, (imgs, _, cams, _, _) in enumerate(dataloader):
            if b_idx >= max_batches:
                break
            imgs = imgs.to(device)
            cams_dev = cams.to(device)
            feat_maps = model.backbone(imgs)
            pooled = model.pooler(feat_maps)
            proj = model.projector(pooled)
            _, scale_w, _, _ = model.view_aware(proj, cam_ids=cams_dev)
            scale_np = scale_w.cpu().numpy()
            cams_np = cams.numpy()
            for cam_val in (0, 2, 3):
                mask = cams_np == cam_val
                if np.any(mask):
                    view_weights[cam_val].append(scale_np[mask])

    summary = {}
    cam_labels = {0: "Aerial_C0", 2: "Wearable_C2", 3: "CCTV_C3"}
    for cam_val, label in cam_labels.items():
        if view_weights[cam_val]:
            arr = np.concatenate(view_weights[cam_val], axis=0)
            mean_w = arr.mean(axis=0).tolist()
            summary[label] = {
                "fine": round(mean_w[0], 4),
                "medium": round(mean_w[1], 4),
                "coarse": round(mean_w[2], 4),
            }
    return summary


def run_smoke_test(args, cfg: dict, device: torch.device) -> bool:
    print("=" * 65)
    print(f"RUNNING SMOKE TEST: {args.exp_name} ({args.model_type})")
    print("=" * 65)
    tf = build_transforms(
        img_size=(cfg["dataset"]["height"], cfg["dataset"]["width"]),
        is_train=True,
        use_enhancement=args.use_enhancement,
        enhancement_mode=cfg["enhancement"]["mode"],
        clip_limit=cfg["enhancement"]["clip_limit"],
        gamma=cfg["enhancement"]["gamma"],
        use_random_erasing=args.use_random_erasing,
    )
    train_set = AGReIDDataset(cfg["dataset"]["root"], split="train_all", transform=tf)
    sampler = RandomIdentitySampler(train_set, num_pids_per_batch=4, num_instances_per_pid=4)
    loader = DataLoader(train_set, batch_size=16, sampler=sampler, num_workers=0)

    model = build_model_from_args(args, cfg, train_set.num_classes).to(device)
    model.train()

    imgs, pids, cams, _, _ = next(iter(loader))
    imgs, pids, cams = imgs.to(device), pids.to(device), cams.to(device)

    ce_id = nn.CrossEntropyLoss(label_smoothing=float(cfg["loss"]["label_smoothing"]))
    ce_view = nn.CrossEntropyLoss()
    tri = TripletLoss(margin=float(cfg["loss"]["triplet_margin"]))
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["optimizer"]["lr"]))

    if args.model_type == "baseline":
        cls_scores, feats = model(imgs)
        loss = ce_id(cls_scores, pids) + tri(feats, pids)
    else:
        cls_scores, feats, aux = model(imgs, cam_ids=cams)
        loss = ce_id(cls_scores, pids) + tri(feats, pids)
        if aux["view_logits"] is not None:
            loss = loss + float(cfg["loss"]["weight_view"]) * ce_view(aux["view_logits"], aux["view_ids"])
        if aux["diff_loss"] is not None:
            loss = loss + float(cfg["loss"]["weight_diffusion"]) * aux["diff_loss"]

    opt.zero_grad()
    loss.backward()
    opt.step()
    print(f"  ✓ 1-batch forward + backward + step succeeded! Loss = {loss.item():.4f}")
    return True


def train(args, cfg: dict, device: torch.device):
    dataset_root = cfg["dataset"]["root"]
    h, w = cfg["dataset"]["height"], cfg["dataset"]["width"]
    checkpoint_dir = cfg["training"]["checkpoint_dir"]
    log_dir = cfg["training"]["log_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    train_tf = build_transforms(
        img_size=(h, w),
        is_train=True,
        use_enhancement=args.use_enhancement,
        enhancement_mode=cfg["enhancement"]["mode"],
        clip_limit=cfg["enhancement"]["clip_limit"],
        gamma=cfg["enhancement"]["gamma"],
        use_random_erasing=args.use_random_erasing,
    )
    eval_tf = build_transforms(
        img_size=(h, w),
        is_train=False,
        use_enhancement=args.use_enhancement,
        enhancement_mode=cfg["enhancement"]["mode"],
        clip_limit=cfg["enhancement"]["clip_limit"],
        gamma=cfg["enhancement"]["gamma"],
    )

    print(f"\n[{args.exp_name}] Preparing Datasets (Enhancement={args.use_enhancement}, RandomErasing={args.use_random_erasing})...")
    train_set = AGReIDDataset(dataset_root, split="train_all", transform=train_tf)
    query_set = AGReIDDataset(dataset_root, split="query", transform=eval_tf)
    gallery_set = AGReIDDataset(dataset_root, split="gallery", transform=eval_tf)

    p = args.p_pids
    k = args.k_instances
    batch_size = p * k
    sampler = RandomIdentitySampler(train_set, num_pids_per_batch=p, num_instances_per_pid=k)

    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=0, drop_last=True)
    query_loader = DataLoader(query_set, batch_size=64, shuffle=False, num_workers=0)
    gallery_loader = DataLoader(gallery_set, batch_size=64, shuffle=False, num_workers=0)

    model = build_model_from_args(args, cfg, train_set.num_classes).to(device)
    ce_id = nn.CrossEntropyLoss(label_smoothing=float(cfg["loss"]["label_smoothing"]))
    ce_view = nn.CrossEntropyLoss()
    tri = TripletLoss(margin=float(cfg["loss"]["triplet_margin"]))

    base_lr = float(cfg["optimizer"]["lr"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=base_lr,
        weight_decay=float(cfg["optimizer"]["weight_decay"]),
    )

    epochs = args.epochs
    total_steps = max(1, epochs * len(train_loader))
    warmup_steps = max(1, int(0.10 * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.01, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    log_file = os.path.join(log_dir, f"{args.exp_name}_train_log.csv")
    with open(log_file, "w") as f:
        f.write("epoch,total_loss,ce_loss,tri_loss,view_loss,diff_loss,rank1,mAP,mINP\n")

    best_rank1 = -1.0
    best_summary = None
    best_ckpt_path = os.path.join(checkpoint_dir, f"{args.exp_name}_best.pth")

    print("\n" + "=" * 68)
    print(f"TRAINING EXPERIMENT: {args.exp_name} | Epochs={epochs} | Batch={batch_size} (P={p}, K={k}) | Device={device}")
    print("=" * 68)

    for epoch in range(1, epochs + 1):
        model.train()
        tot_s = ce_s = tri_s = view_s = diff_s = 0.0
        n_batches = 0
        t0 = time.time()

        for imgs, pids, cams, _, _ in tqdm(train_loader, desc=f"[{args.exp_name}] Epoch {epoch}/{epochs}"):
            imgs = imgs.to(device)
            pids = pids.to(device)
            cams = cams.to(device)

            optimizer.zero_grad()

            if args.model_type == "baseline":
                cls_scores, feats = model(imgs)
                l_ce = ce_id(cls_scores, pids)
                l_tri = tri(feats, pids)
                l_view = torch.tensor(0.0, device=device)
                l_diff = torch.tensor(0.0, device=device)
                loss = cfg["loss"]["weight_id"] * l_ce + cfg["loss"]["weight_triplet"] * l_tri
            else:
                cls_scores, feats, aux = model(imgs, cam_ids=cams)
                l_ce = ce_id(cls_scores, pids)
                l_tri = tri(feats, pids)
                l_view = (
                    ce_view(aux["view_logits"], aux["view_ids"])
                    if aux["view_logits"] is not None
                    else torch.tensor(0.0, device=device)
                )
                l_diff = aux["diff_loss"] if aux["diff_loss"] is not None else torch.tensor(0.0, device=device)
                loss = (
                    cfg["loss"]["weight_id"] * l_ce
                    + cfg["loss"]["weight_triplet"] * l_tri
                    + cfg["loss"]["weight_view"] * l_view
                    + cfg["loss"]["weight_diffusion"] * l_diff
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            tot_s += loss.item()
            ce_s += l_ce.item()
            tri_s += l_tri.item()
            view_s += l_view.item()
            diff_s += l_diff.item()
            n_batches += 1

        elapsed = time.time() - t0
        avg_tot = tot_s / max(1, n_batches)
        avg_ce = ce_s / max(1, n_batches)
        avg_tri = tri_s / max(1, n_batches)
        avg_view = view_s / max(1, n_batches)
        avg_diff = diff_s / max(1, n_batches)

        print(
            f"Epoch {epoch}/{epochs} [{elapsed:.1f}s] - Total: {avg_tot:.4f} "
            f"(CE: {avg_ce:.4f}, Tri: {avg_tri:.4f}, View: {avg_view:.4f}, Diff: {avg_diff:.4f})"
        )

        # Evaluate at end of each epoch
        summary = evaluate_comprehensive(
            model, query_loader, gallery_loader, device, exp_name=f"{args.exp_name}_ep{epoch}"
        )

        if args.model_type == "proposed" and args.use_view_aware:
            scale_stats = collect_view_scale_weights(model, query_loader, device)
            summary["view_scale_weights"] = scale_stats
            print(f"  View-Aware Scale Weights: {scale_stats}")

        r1 = summary["overall"]["Rank-1"]
        m_ap = summary["overall"]["mAP"]
        m_inp = summary["overall"]["mINP"]

        with open(log_file, "a") as f:
            f.write(
                f"{epoch},{avg_tot:.4f},{avg_ce:.4f},{avg_tri:.4f},{avg_view:.4f},{avg_diff:.4f},"
                f"{r1:.2f},{m_ap:.2f},{m_inp:.2f}\n"
            )

        if r1 > best_rank1:
            best_rank1 = r1
            best_summary = summary
            best_summary["experiment"] = args.exp_name
            best_summary["epochs_trained"] = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "exp_name": args.exp_name,
                    "model_state_dict": model.state_dict(),
                    "summary": best_summary,
                },
                best_ckpt_path,
            )
            print(f"  ★ Saved best checkpoint -> {best_ckpt_path} (Rank-1={r1:.2f}%, mAP={m_ap:.2f}%, mINP={m_inp:.2f}%)")

    eval_json_path = f"results/eval_{args.exp_name}.json"
    with open(eval_json_path, "w") as jf:
        json.dump(best_summary, jf, indent=2)
    print(f"Saved final evaluation metrics to {eval_json_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Proposed/Ablation Models on AG-ReID.v2")
    parser.add_argument("--config", type=str, default="configs/proposed.yaml")
    parser.add_argument("--exp_name", type=str, default="proposed_full")
    parser.add_argument("--model_type", type=str, default="proposed", choices=["baseline", "proposed"])
    parser.add_argument("--use_enhancement", action="store_true")
    parser.add_argument("--use_random_erasing", action="store_true")
    parser.add_argument("--use_view_aware", action="store_true")
    parser.add_argument("--use_diffusion", action="store_true")
    parser.add_argument("--fusion_strategy", type=str, default="weighted")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--p_pids", type=int, default=16)
    parser.add_argument("--k_instances", type=int, default=4)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    print(f"Compute device: {device}")

    if args.smoke_test:
        ok = run_smoke_test(args, cfg, device)
        sys.exit(0 if ok else 1)
    else:
        train(args, cfg, device)


if __name__ == "__main__":
    main()

