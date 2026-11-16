"""Image Enhancement Module for Aerial-Ground Person Re-Identification.

Provides:
- CLAHE (Contrast Limited Adaptive Histogram Equalization) in LAB color space
- Gamma Correction via high-speed lookup tables (LUT)
- Unified ImageEnhancer supporting multiple enhancement modes
"""

from .clahe import apply_clahe
from .gamma import apply_gamma_correction
from .image_enhancer import ImageEnhancer

__all__ = ["apply_clahe", "apply_gamma_correction", "ImageEnhancer"]

