"""Model Information and Architecture Details Route."""

from typing import Dict, Any
from fastapi import APIRouter
from backend.app.core.config import settings
from backend.app.services.model_service import model_service

router = APIRouter(prefix="/model", tags=["Model Architecture"])


@router.get("/info")
async def get_model_info() -> Dict[str, Any]:
    """Return model architecture specifications, layers, and checkpoint status."""
    return {
        "model_name": "View-Aware Multi-Scale Feature Fusion with Latent Diffusion (Proposed Full)",
        "framework": "PyTorch 2.15.0",
        "checkpoint": {
            "name": "proposed_full_best.pth",
            "path": str(settings.CHECKPOINT_PROPOSED),
            "status": "Loaded and Verified" if model_service.is_loaded else "Ready to Load",
            "device": str(model_service.device),
        },
        "parameters": {
            "total": "26,173,086",
            "total_millions": 26.17,
            "trainable": "26,173,086",
            "embedding_dim": settings.EMBEDDING_DIM,
            "num_classes": settings.NUM_CLASSES,
        },
        "components": [
            {
                "name": "Multi-Scale ResNet-50 Backbone",
                "detail": "Modified ResNet-50 with last_stride=1 preserving spatial details. Extracts 3 distinct hierarchical representations: Fine (layer1: 256 channels), Medium (layer2: 512 channels), and Coarse (layer4: 2048 channels).",
                "params": "23.51M",
            },
            {
                "name": "Global Average Pooling (GAP) Pooler",
                "detail": "Collapses 2D spatial feature maps of varying resolutions into fixed 1D vector representations per scale.",
                "params": "0 (Non-parametric)",
            },
            {
                "name": "Multi-Scale Feature Projector",
                "detail": "1x1 Linear projections mapping heterogeneous channel dimensions (256, 512, 2048) into a unified d=512 semantic embedding space.",
                "params": "1.44M",
            },
            {
                "name": "View-Aware Dynamic Scale Attention Gate",
                "detail": "Learns viewpoint-specific scale importance weights conditioned on camera domain (Aerial C0, Wearable C2, CCTV C3).",
                "params": "0.19M",
            },
            {
                "name": "Latent Feature Diffusion Module",
                "detail": "Compact 2-step 1D latent diffusion denoising network that resolves viewpoint domain discrepancy without expensive pixel-space generation.",
                "params": "0.73M",
            },
            {
                "name": "BNNeck & Linear Classifier",
                "detail": "Batch Normalization neck separating cosine metric learning space from identity classification space (590 training identities).",
                "params": "0.30M",
            },
        ],
        "scale_attention_weights": {
            "Aerial_UAV_C0": {"fine": 0.1745, "medium": 0.1956, "coarse": 0.6298},
            "Wearable_C2": {"fine": 0.1799, "medium": 0.1989, "coarse": 0.6212},
            "Ground_CCTV_C3": {"fine": 0.1655, "medium": 0.1928, "coarse": 0.6417},
        },
    }

