"""Data Preprocessing and Augmentation Transforms for Person Re-ID.

Defines standard transforms used in modern deep learning person Re-ID benchmarks:
- Optional CLAHE + Gamma Correction image enhancement on PIL images.
- Input height: 256 pixels, Input width: 128 pixels (standard 2:1 pedestrian aspect ratio).
- Training: Random horizontal flip, random padding & cropping, ImageNet normalization,
  and optional Random Erasing regularization.
- Evaluation / Testing: Deterministic resizing and ImageNet normalization.
"""

from typing import Tuple
from torchvision import transforms
from src.enhancement.image_enhancer import ImageEnhancer


class PILImageEnhancementTransform:
    """Callable PIL-to-PIL wrapper around ImageEnhancer for DataLoader pipelines."""

    def __init__(
        self,
        mode: str = "clahe_gamma",
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
        gamma: float = 1.2,
        order: str = "clahe_first",
    ):
        self.enhancer = ImageEnhancer(
            mode=mode,
            clip_limit=clip_limit,
            tile_grid_size=tile_grid_size,
            gamma=gamma,
            order=order,
        )

    def __call__(self, img):
        return self.enhancer.enhance_pil(img)


def build_transforms(
    img_size: Tuple[int, int] = (256, 128),
    is_train: bool = True,
    use_enhancement: bool = False,
    enhancement_mode: str = "clahe_gamma",
    clip_limit: float = 2.0,
    gamma: float = 1.2,
    use_random_erasing: bool = False,
) -> transforms.Compose:
    """Build the image transformation pipeline.

    Args:
        img_size: (height, width) tuple, defaults to (256, 128).
        is_train: Whether to include data augmentations (flip, crop).
        use_enhancement: Whether to apply CLAHE/Gamma enhancement first.
        enhancement_mode: 'clahe_gamma', 'clahe', 'gamma', or 'none'.
        clip_limit: CLAHE clip limit.
        gamma: Gamma correction factor.
        use_random_erasing: Apply RandomErasing during training for occlusion robustness.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    height, width = img_size
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    transform_list = []

    if use_enhancement and enhancement_mode not in ("none", "original"):
        transform_list.append(
            PILImageEnhancementTransform(
                mode=enhancement_mode,
                clip_limit=clip_limit,
                gamma=gamma,
            )
        )

    if is_train:
        transform_list.extend([
            transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(10),
            transforms.RandomCrop((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
        if use_random_erasing:
            transform_list.append(transforms.RandomErasing(p=0.5, value="random"))
    else:
        transform_list.extend([
            transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])

    return transforms.Compose(transform_list)

