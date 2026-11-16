"""Image Enhancement Service (CLAHE and Look-Up Table Gamma Correction)."""

import io
import base64
from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image

from src.enhancement.image_enhancer import ImageEnhancer


def pil_to_base64(img: Image.Image, format: str = "JPEG", quality: int = 90) -> str:
    """Encode PIL image to base64 Data URL."""
    buffered = io.BytesIO()
    img.save(buffered, format=format, quality=quality)
    encoded = base64.b64encode(buffered.getvalue()).decode("ascii")
    return f"data:image/{format.lower()};base64,{encoded}"


def compute_image_stats(img: Image.Image) -> Dict[str, float]:
    """Compute mean intensity, standard deviation, RMS contrast, and Shannon entropy."""
    gray = np.array(img.convert("L"), dtype=np.float32)
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    rms_contrast = float(std_val / (mean_val + 1e-6))

    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    p = hist / (hist.sum() + 1e-9)
    p = p[p > 0]
    entropy_bits = float(-np.sum(p * np.log2(p)))

    return {
        "mean": round(mean_val, 2),
        "std": round(std_val, 2),
        "rms_contrast": round(rms_contrast, 4),
        "entropy_bits": round(entropy_bits, 4),
    }


class EnhancementService:
    """Service providing CLAHE and Gamma Correction pipelines."""

    def __init__(self, clip_limit: float = 2.0, gamma: float = 1.2):
        self.default_enhancer = ImageEnhancer(mode="clahe_gamma", clip_limit=clip_limit, gamma=gamma)

    def enhance_pil(self, img: Image.Image) -> Image.Image:
        """Apply default CLAHE + Gamma pipeline."""
        return self.default_enhancer.enhance_pil(img)

    def process_all_variations(
        self,
        img: Image.Image,
        clip_limit: float = 2.0,
        gamma: float = 1.2,
    ) -> Dict[str, Any]:
        """Generate Original, CLAHE-only, Gamma-only, and Combined images with quality statistics."""
        enh_clahe = ImageEnhancer(mode="clahe", clip_limit=clip_limit).enhance_pil(img)
        enh_gamma = ImageEnhancer(mode="gamma", gamma=gamma).enhance_pil(img)
        enh_combined = ImageEnhancer(mode="clahe_gamma", clip_limit=clip_limit, gamma=gamma).enhance_pil(img)

        raw_stats = compute_image_stats(img)
        enhanced_stats = compute_image_stats(enh_combined)

        rms_gain = (enhanced_stats["rms_contrast"] - raw_stats["rms_contrast"]) / max(1e-4, raw_stats["rms_contrast"])
        entropy_gain = (enhanced_stats["entropy_bits"] - raw_stats["entropy_bits"]) / max(1e-4, raw_stats["entropy_bits"])

        return {
            "raw_image": pil_to_base64(img),
            "clahe_only": pil_to_base64(enh_clahe),
            "gamma_only": pil_to_base64(enh_gamma),
            "combined": pil_to_base64(enh_combined),
            "raw_stats": raw_stats,
            "enhanced_stats": enhanced_stats,
            "improvements": {
                "mean_change": round(enhanced_stats["mean"] - raw_stats["mean"], 2),
                "std_change": round(enhanced_stats["std"] - raw_stats["std"], 2),
                "rms_contrast_gain": round(rms_gain, 4),
                "entropy_gain": round(entropy_gain, 4),
            },
        }


enhancement_service = EnhancementService()

