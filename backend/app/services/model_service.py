"""Model Service: Handles View-Aware Multi-Scale Re-ID Model loading and inference.

Loads the trained PyTorch model checkpoint:
results/checkpoints/proposed_full_best.pth (26.17M parameters)
Device: Apple Silicon GPU (MPS) if available, else CUDA, else CPU.
"""

import io
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from backend.app.core.config import settings
from backend.app.services.enhancement_service import pil_to_base64
from src.data.transforms import build_transforms
from src.models.view_aware_fusion_model import ViewAwareMultiScaleReIDModel
from src.models.view_aware import map_cam_to_view


class ModelService:
    """Singleton service for model loading and feature extraction."""

    def __init__(self):
        self.device = self._get_device()
        self.model: Optional[ViewAwareMultiScaleReIDModel] = None
        self.transform = None
        self.is_loaded = False
        self.total_params = 0

    def _get_device(self) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def load_model(self):
        """Load the trained checkpoint weights into memory."""
        if self.is_loaded:
            return

        print(f"[ModelService] Initializing on device: {self.device}")
        checkpoint_path = settings.CHECKPOINT_PROPOSED

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found at: {checkpoint_path}. "
                "Ensure results/checkpoints/proposed_full_best.pth is present."
            )

        # Instantiate proposed architecture (identical to training & evaluation)
        self.model = ViewAwareMultiScaleReIDModel(
            num_classes=settings.NUM_CLASSES,
            pretrained=False,
            use_view_aware=True,
            use_diffusion=True,
        )

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.total_params = sum(p.numel() for p in self.model.parameters())

        # Exact evaluation transform pipeline (Resize to (256, 128) + ImageNet Norm)
        self.transform = build_transforms(
            img_size=settings.IMG_SIZE,
            is_train=False,
            use_enhancement=True,
        )

        self.is_loaded = True
        print(f"[ModelService] ✓ Proposed Re-ID Model loaded ({self.total_params / 1e6:.2f}M parameters).")

    def extract_embedding(
        self,
        img: Image.Image,
        cam_id: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Extract 512-d L2-normalized feature embedding from a PIL image.

        Args:
            img: PIL Image in RGB mode.
            cam_id: Optional camera id (0: UAV, 2: Wearable, 3: CCTV).

        Returns:
            embedding: 512-dim numpy array (L2 normalized).
            info: Dictionary containing detected viewpoint, view ID, and scale attention weights.
        """
        if not self.is_loaded:
            self.load_model()

        # Preprocess PIL image through exact evaluation pipeline
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)

        cam_tensor = None
        if cam_id is not None:
            cam_tensor = torch.tensor([cam_id], dtype=torch.long, device=self.device)

        with torch.no_grad():
            # Run forward pass through backbone, pooler, projector, view-aware, and diffusion
            feat_maps = self.model.backbone(input_tensor)
            pooled_dict = self.model.pooler(feat_maps)
            proj_dict = self.model.projector(pooled_dict)

            # View-aware attention and conditioning
            fused_feat, scale_weights, view_logits, view_ids = self.model.view_aware(
                proj_dict, cam_ids=cam_tensor
            )

            # Latent diffusion refinement
            refined_feat, _ = self.model.latent_diffusion(fused_feat, view_ids=view_ids)

            # BNNeck and L2 normalization
            bn_feat = self.model.bnneck(refined_feat)
            emb = F.normalize(bn_feat, p=2, dim=1).cpu().numpy().squeeze(0).astype(np.float32)

            # Extract view prediction if available
            pred_view_idx = int(view_ids.squeeze(0).item()) if view_ids is not None else 0
            view_names = {0: "Aerial UAV (C0)", 1: "Wearable (C2)", 2: "Ground CCTV (C3)"}
            detected_view = view_names.get(pred_view_idx, "Unknown")

            # Extract scale weights
            sw = scale_weights.squeeze(0).cpu().numpy()
            scale_info = {
                "fine": round(float(sw[0]), 4),
                "medium": round(float(sw[1]), 4),
                "coarse": round(float(sw[2]), 4),
            }

        return emb, {
            "detected_view": detected_view,
            "view_id": pred_view_idx,
            "scale_weights": scale_info,
        }

    def compare_images(
        self,
        img1: Image.Image,
        img2: Image.Image,
        cam1: Optional[int] = None,
        cam2: Optional[int] = None,
        threshold: float = 0.40,
    ) -> Dict[str, Any]:
        """Compare two person images and compute cosine similarity."""
        t0 = time.time()
        emb1, info1 = self.extract_embedding(img1, cam_id=cam1)
        emb2, info2 = self.extract_embedding(img2, cam_id=cam2)

        # Cosine similarity between unit-norm vectors
        sim = float(np.dot(emb1, emb2))
        sim = max(-1.0, min(1.0, sim))

        elapsed_ms = (time.time() - t0) * 1000.0
        is_same = sim >= threshold

        confidence = "High" if abs(sim - threshold) > 0.15 else "Moderate"

        return {
            "status": "success",
            "similarity": round(sim, 4),
            "similarity_pct": f"{sim * 100:.1f}%",
            "threshold": threshold,
            "is_same": is_same,
            "verdict": "SAME PERSON" if is_same else "DIFFERENT PERSONS",
            "confidence": confidence,
            "view1": info1["detected_view"],
            "view2": info2["detected_view"],
            "time_ms": round(elapsed_ms, 2),
            "im1_raw": pil_to_base64(img1),
            "im1_enhanced": pil_to_base64(img1),  # already enhanced via transform
            "im2_raw": pil_to_base64(img2),
            "im2_enhanced": pil_to_base64(img2),
        }


model_service = ModelService()

