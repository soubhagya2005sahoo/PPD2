"""Gamma Correction for Controlled Luminance and Brightness Adjustment.

In Aerial-Ground Person Re-ID, pedestrians under drones or CCTV frequently move
between deep building shadows and harsh open sunlight.

Gamma Correction uses a non-linear power-law transform:
    V_out = 255 * (V_in / 255) ** (1 / gamma)
- gamma > 1.0 : Brightens shadowed and underexposed regions while preserving highlights.
- gamma = 1.0 : Identity (no change).
- gamma < 1.0 : Compresses bright highlights.

Implemented using precomputed 256-element Look-Up Tables (LUT) for maximum speed.
"""

from typing import Union
import numpy as np
import cv2
from PIL import Image


def build_gamma_lut(gamma: float) -> np.ndarray:
    """Precompute a 256-element uint8 lookup table for power-law gamma transformation."""
    if gamma <= 0:
        raise ValueError(f"Gamma value must be strictly positive, got {gamma}")
    inv_gamma = 1.0 / gamma
    lut = np.array([((i / 255.0) ** inv_gamma) * 255.0 for i in range(256)]).astype(np.uint8)
    return lut


def apply_gamma_correction(
    image: Union[np.ndarray, Image.Image],
    gamma: float = 1.2,
) -> np.ndarray:
    """Apply non-linear gamma correction to an image using a lookup table.

    Args:
        image: Input image as numpy uint8 array (H, W, 3) or PIL Image.
        gamma: Gamma parameter. Values > 1.0 brighten dark regions (default: 1.2).

    Returns:
        Gamma-corrected image as numpy uint8 array (H, W, 3).
    """
    if isinstance(image, Image.Image):
        rgb_array = np.array(image.convert("RGB"), dtype=np.uint8)
    elif isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            rgb_array = np.clip(image, 0, 255).astype(np.uint8)
        else:
            rgb_array = image.copy()
    else:
        raise TypeError(f"Unsupported image type: {type(image)}. Expected np.ndarray or PIL.Image.")

    # Short-circuit if gamma is 1.0
    if abs(gamma - 1.0) < 1e-4:
        return rgb_array

    # 1. Build lookup table
    lut = build_gamma_lut(gamma)

    # 2. Fast mapping using cv2.LUT
    corrected = cv2.LUT(rgb_array, lut)
    return corrected

