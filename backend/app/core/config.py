"""Backend Configuration Settings."""

import os
from pathlib import Path
from typing import List
from pydantic import BaseModel

# Project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseModel):
    PROJECT_NAME: str = "View-Aware Aerial-Ground Person Re-Identification"
    API_V1_PREFIX: str = "/api"

    # Dataset & Weights Paths
    DATASET_ROOT: Path = ROOT_DIR / "datasets" / "AG-ReID.v2"
    CHECKPOINT_BASELINE: Path = ROOT_DIR / "results" / "checkpoints" / "best_model.pth"
    CHECKPOINT_PROPOSED: Path = ROOT_DIR / "results" / "checkpoints" / "proposed_full_best.pth"
    GALLERY_CACHE: Path = ROOT_DIR / "features" / "gallery_cache_proposed.npz"

    # CORS Settings
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "*",
    ]

    # Model Parameters
    EMBEDDING_DIM: int = 512
    NUM_CLASSES: int = 590
    IMG_SIZE: tuple = (256, 128)

    # Re-ID Camera Mappings
    CAM_NAMES: dict = {
        0: "Aerial UAV (C0)",
        2: "Wearable Camera (C2)",
        3: "Ground CCTV (C3)",
    }


settings = Settings()

