"""Contrast Limited Adaptive Histogram Equalization (CLAHE) for Color Images.

In Aerial-Ground Person Re-ID, high-altitude UAV images frequently suffer from
low local contrast, haze, and strong directional lighting variations.

Applying standard histogram equalization on RGB directly causes severe color
distortion. In contrast, applying CLAHE strictly to the Lightness (L) channel
in the CIELAB color space improves local contrast while preserving genuine
chromaticity (clothing colors, skin tones).
"""

from typing import Tuple, Union
import numpy as np
import cv2
from PIL import Image


def apply_clahe(
    image: Union[np.ndarray, Image.Image],
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE to the L (Lightness) channel of an RGB image in LAB color space.

    Args:
        image: Input RGB image as numpy uint8 array (H, W, 3) or PIL Image.
        clip_limit: Contrast clipping limit threshold (default: 2.0).
                    Higher values increase contrast; lower values prevent noise amplification.
        tile_grid_size: Size of contextual grid for local equalization (default: (8, 8)).

    Returns:
        Enhanced RGB image as numpy uint8 array of shape (H, W, 3).
    """
    # Convert PIL Image to RGB NumPy array if necessary
    if isinstance(image, Image.Image):
        rgb_array = np.array(image.convert("RGB"), dtype=np.uint8)
    elif isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            rgb_array = np.clip(image, 0, 255).astype(np.uint8)
        else:
            rgb_array = image.copy()
    else:
        raise TypeError(f"Unsupported image type: {type(image)}. Expected np.ndarray or PIL.Image.")

    # Validate image shape
    if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
        raise ValueError(f"Expected 3-channel RGB image (H, W, 3), got shape {rgb_array.shape}")

    # 1. Convert from RGB to CIELAB color space
    lab = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # 2. Initialize CLAHE operator
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # 3. Equalize only the Lightness (L) channel
    l_enhanced = clahe.apply(l_channel)

    # 4. Merge enhanced Lightness with original Chromatic (A, B) channels
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])

    # 5. Convert back to standard RGB color space
    rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    return rgb_enhanced

