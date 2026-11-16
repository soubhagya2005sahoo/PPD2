"""Batch Dataset Enhancement Script for AG-ReID.v2.

Reads original images from datasets/AG-ReID.v2/, applies the selected
enhancement pipeline (CLAHE, Gamma, or CLAHE+Gamma), and writes enhanced
images into datasets/AG-ReID.v2_enhanced/ with identical directory hierarchy.

Features:
- Never modifies original dataset files.
- Resumable: skips already-processed images.
- Error logging.
- Preserves all file names, directory names, and camera/identity labels.
"""

import os
import sys
import argparse
import yaml

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.enhancement.image_enhancer import ImageEnhancer


def main():
    parser = argparse.ArgumentParser(description="Batch Enhance AG-ReID.v2 Dataset")
    parser.add_argument("--config", type=str, default="configs/enhancement.yaml")
    parser.add_argument("--max_images", type=int, default=None, help="Optional limit for testing")
    parser.add_argument("--no_resume", action="store_true", help="Force re-processing of existing files")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    enh_cfg = cfg["enhancement"]
    io_cfg = cfg["io"]

    # Locate input root (handling nested AG-ReID.v2 if present)
    in_root = io_cfg["input_dir"]
    if os.path.isdir(os.path.join(in_root, "AG-ReID.v2")):
        in_root = os.path.join(in_root, "AG-ReID.v2")

    out_root = io_cfg["output_dir"]

    enhancer = ImageEnhancer(
        mode=enh_cfg["method"],
        clip_limit=enh_cfg["clahe"]["clip_limit"],
        tile_grid_size=tuple(enh_cfg["clahe"]["tile_grid_size"]),
        gamma=enh_cfg["gamma"]["value"],
        order=enh_cfg["order"],
    )

    print("=" * 65)
    print("AG-ReID.v2 DATASET ENHANCEMENT PROCESSOR")
    print("=" * 65)
    print(f"Input Directory  : {in_root}")
    print(f"Output Directory : {out_root}")
    print(f"Enhancement Mode : {enh_cfg['method']} ({enh_cfg['order']})")
    print(f"CLAHE Parameters : clip_limit={enh_cfg['clahe']['clip_limit']}, grid={enh_cfg['clahe']['tile_grid_size']}")
    print(f"Gamma Value      : {enh_cfg['gamma']['value']}")
    print("=" * 65)

    stats = enhancer.process_dataset(
        input_root=in_root,
        output_root=out_root,
        resume=not args.no_resume,
        max_images=args.max_images,
    )

    print("\nSummary:")
    print(f"  Total Considered : {stats['total_input']}")
    print(f"  Newly Processed  : {stats['processed']}")
    print(f"  Skipped (Cached) : {stats['skipped']}")
    print(f"  Errors           : {stats['errors']}")
    print("=" * 65)


if __name__ == "__main__":
    main()

