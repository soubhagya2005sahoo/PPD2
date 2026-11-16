"""Custom PyTorch Dataset for the AG-ReID.v2 Benchmark.

Parses pedestrian bounding boxes across UAV (C0), Wearable (C2),
and CCTV (C3) cameras while extracting Person IDs, Altitude levels,
and frame numbers from filename nomenclature.
"""

import os
import re
from typing import Callable, Dict, List, Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset


class AGReIDDataset(Dataset):
    """AG-ReID.v2 Person Re-Identification Dataset."""

    FILENAME_REGEX = re.compile(
        r"^P(?P<pid>\d+)T(?P<tid>\d+)A(?P<alt>\d+)C(?P<cam>\d+)F(?P<frame>\d+[A-Z]?)\.(?P<ext>jpg|jpeg|png)$",
        re.IGNORECASE,
    )

    def __init__(
        self,
        dataset_root: str,
        split: str = "train_all",
        transform: Optional[Callable] = None,
        pid_to_label: Optional[Dict[str, int]] = None,
    ):
        """Initialize the AG-ReID.v2 Dataset.

        Args:
            dataset_root: Path to AG-ReID.v2 root directory.
            split: One of 'train_all', 'train', 'query', 'gallery'.
            transform: torchvision transformation pipeline.
            pid_to_label: Optional dictionary mapping string Person ID to integer class index.
        """
        super().__init__()
        self.dataset_root = dataset_root
        self.split = "train_all" if split == "train" else split
        self.transform = transform

        # Resolve split folder path (handle potential nested directory)
        self.split_dir = self._resolve_split_path(dataset_root, self.split)
        if not os.path.isdir(self.split_dir):
            raise FileNotFoundError(f"Split directory does not exist: {self.split_dir}")

        # Scan files and parse metadata
        self.samples: List[Tuple[str, str, int, int]] = []  # (path, pid_str, cam_id, alt_id)
        self._scan_dataset()

        # Build or apply PID -> class label mapping
        raw_pids = sorted(list(set(s[1] for s in self.samples)))
        if pid_to_label is not None:
            self.pid_to_label = pid_to_label
        else:
            self.pid_to_label = {pid_str: idx for idx, pid_str in enumerate(raw_pids)}

        self.num_classes = len(self.pid_to_label)

        # Index samples by person identity for identity-based sampling
        self.pid_to_indices: Dict[int, List[int]] = {}
        for idx, (_, pid_str, _, _) in enumerate(self.samples):
            label = self.pid_to_label.get(pid_str, -1)
            if label not in self.pid_to_indices:
                self.pid_to_indices[label] = []
            self.pid_to_indices[label].append(idx)

    def _resolve_split_path(self, root: str, split: str) -> str:
        candidates = [
            os.path.join(root, split),
            os.path.join(root, "AG-ReID.v2", split),
        ]
        for c in candidates:
            if os.path.isdir(c):
                return c
        return os.path.join(root, split)

    def _scan_dataset(self) -> None:
        """Scan directory and parse filenames."""
        for root, _, files in os.walk(self.split_dir):
            for f in sorted(files):
                if f.startswith("."):
                    continue
                match = self.FILENAME_REGEX.match(f)
                if match:
                    pid_str = match.group("pid")
                    cam_id = int(match.group("cam"))
                    alt_id = int(match.group("alt"))
                    img_path = os.path.join(root, f)
                    self.samples.append((img_path, pid_str, cam_id, alt_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int, int, str]:
        """Fetch a sample.

        Returns:
            image_tensor: Normalized image tensor of shape (3, H, W).
            pid_label: Integer class label for cross-entropy loss (or -1 if unmapped).
            cam_id: Camera sensor index (0 = UAV, 2 = Wearable, 3 = CCTV).
            alt_id: UAV altitude index (0 = Low, 1 = Med, 2 = High).
            img_path: Absolute path to the original image.
        """
        img_path, pid_str, cam_id, alt_id = self.samples[index]

        try:
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")
        except Exception as e:
            # Fallback to a black image if an error occurs to prevent crashing
            print(f"Warning: Failed to load image {img_path}: {e}")
            img_rgb = Image.new("RGB", (128, 256), (0, 0, 0))

        if self.transform is not None:
            image_tensor = self.transform(img_rgb)
        else:
            image_tensor = torch.zeros(3, 256, 128)

        pid_label = self.pid_to_label.get(pid_str, -1)
        return image_tensor, pid_label, cam_id, alt_id, img_path

