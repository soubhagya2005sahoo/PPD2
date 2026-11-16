"""Unified Image Enhancement Pipeline for AG-ReID.v2.

Combines CLAHE (in LAB color space) and Gamma Correction (via LUT)
with support for multiple selectable enhancement modes and configurable execution order:
1. 'none' / 'original'   : Pass-through (no modification)
2. 'clahe'               : CLAHE contrast enhancement only
3. 'gamma'               : Gamma brightness correction only
4. 'clahe_gamma'         : Combined CLAHE and Gamma correction
"""

import os
import glob
from typing import Optional, Tuple, Union
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

from .clahe import apply_clahe
from .gamma import apply_gamma_correction


class ImageEnhancer:
    """Configurable image enhancement engine for Aerial-Ground Person Re-ID."""

    VALID_MODES = ["none", "original", "clahe", "gamma", "clahe_gamma", "combined"]
    VALID_ORDERS = ["clahe_first", "gamma_first"]

    def __init__(
        self,
        mode: str = "clahe_gamma",
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
        gamma: float = 1.2,
        order: str = "clahe_first",
    ):
        """Initialize the ImageEnhancer.

        Args:
            mode: One of 'none', 'clahe', 'gamma', 'clahe_gamma'.
            clip_limit: Contrast clipping limit for CLAHE.
            tile_grid_size: Contextual window size for CLAHE.
            gamma: Power-law parameter for Gamma correction (>1 brightens shadows).
            order: Execution order for combined mode ('clahe_first' or 'gamma_first').
        """
        self.mode = mode.lower()
        if self.mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode '{self.mode}'. Choose from: {self.VALID_MODES}")

        self.clip_limit = float(clip_limit)
        self.tile_grid_size = tile_grid_size
        self.gamma = float(gamma)
        self.order = order.lower()
        if self.order not in self.VALID_ORDERS:
            raise ValueError(f"Invalid order '{self.order}'. Choose from: {self.VALID_ORDERS}")

    def enhance(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Apply the configured enhancement pipeline to an RGB image.

        Returns:
            Enhanced RGB image as numpy uint8 array (H, W, 3).
        """
        if self.mode in ["none", "original"]:
            if isinstance(image, Image.Image):
                return np.array(image.convert("RGB"), dtype=np.uint8)
            return image.copy()

        if self.mode == "clahe":
            return apply_clahe(image, clip_limit=self.clip_limit, tile_grid_size=self.tile_grid_size)

        if self.mode == "gamma":
            return apply_gamma_correction(image, gamma=self.gamma)

        # Combined Mode ('clahe_gamma' / 'combined')
        if self.order == "clahe_first":
            step1 = apply_clahe(image, clip_limit=self.clip_limit, tile_grid_size=self.tile_grid_size)
            step2 = apply_gamma_correction(step1, gamma=self.gamma)
            return step2
        else:
            step1 = apply_gamma_correction(image, gamma=self.gamma)
            step2 = apply_clahe(step1, clip_limit=self.clip_limit, tile_grid_size=self.tile_grid_size)
            return step2

    def enhance_pil(self, image: Union[np.ndarray, Image.Image]) -> Image.Image:
        """Apply enhancement and return a PIL Image."""
        result_array = self.enhance(image)
        return Image.fromarray(result_array)

    def enhance_file(self, input_path: str, output_path: str) -> bool:
        """Read an image file from disk, enhance it, and write to output path."""
        try:
            with Image.open(input_path) as img:
                enhanced_pil = self.enhance_pil(img)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            enhanced_pil.save(output_path, format="JPEG", quality=95)
            return True
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            return False

    def process_dataset(
        self,
        input_root: str,
        output_root: str,
        resume: bool = True,
        max_images: Optional[int] = None,
    ) -> dict:
        """Batch-process an entire dataset directory hierarchy safely."""
        os.makedirs(output_root, exist_ok=True)
        log_file = os.path.join(output_root, "enhancement_log.txt")

        # Discover all image files
        all_files = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            all_files.extend(glob.glob(os.path.join(input_root, "**", ext), recursive=True))

        all_files = sorted(all_files)
        if max_images is not None:
            all_files = all_files[:max_images]

        total = len(all_files)
        success_count = 0
        skipped_count = 0
        error_count = 0

        print(f"Processing {total:,} images from '{input_root}' -> '{output_root}'...")
        print(f"Enhancement mode: {self.mode} (order: {self.order}, gamma: {self.gamma}, clip_limit: {self.clip_limit})")

        with open(log_file, "a") as log:
            for in_path in tqdm(all_files, desc="Enhancing Dataset"):
                rel_path = os.path.relpath(in_path, input_root)
                out_path = os.path.join(output_root, rel_path)

                if resume and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                    skipped_count += 1
                    continue

                ok = self.enhance_file(in_path, out_path)
                if ok:
                    success_count += 1
                else:
                    error_count += 1
                    log.write(f"FAILED: {in_path}\n")

        print(f"Processing Complete: {success_count} processed, {skipped_count} skipped (resumed), {error_count} errors.")
        return {
            "total_input": total,
            "processed": success_count,
            "skipped": skipped_count,
            "errors": error_count,
        }

