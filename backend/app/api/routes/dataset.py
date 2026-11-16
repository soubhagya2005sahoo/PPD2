"""Dataset Information and Sample Query Routes."""

import base64
from pathlib import Path
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
from backend.app.core.config import settings
from backend.app.schemas import DatasetSample

router = APIRouter(prefix="/dataset", tags=["Dataset"])

# Verified curated query samples from AG-ReID.v2
SAMPLE_PROBES = [
    {
        "id": "p0867_uav",
        "title": "Person 0867 (Aerial UAV)",
        "desc": "High-altitude aerial capture from UAV drone C0 with scale variation.",
        "camera": "Aerial UAV (C0)",
        "pid": "0867",
        "rel_path": "AG-ReID.v2/query/P0867T04070A0/P0867T04070A0C0F13861.jpg",
    },
    {
        "id": "p0867_cctv",
        "title": "Person 0867 (Ground CCTV)",
        "desc": "Ground-level surveillance camera C3 showing frontal person view.",
        "camera": "Ground CCTV (C3)",
        "pid": "0867",
        "rel_path": "AG-ReID.v2/query/P0867T04070A0/P0867T04070A0C3F07831.jpg",
    },
    {
        "id": "p0195_uav",
        "title": "Person 0195 (Aerial UAV)",
        "desc": "Aerial UAV top-down viewpoint under ambient outdoor illumination.",
        "camera": "Aerial UAV (C0)",
        "pid": "0195",
        "rel_path": "AG-ReID.v2/query/P0195T02210A2/P0195T02210A2C0F121.jpg",
    },
    {
        "id": "p0195_cctv",
        "title": "Person 0195 (Ground CCTV)",
        "desc": "Ground camera perspective showing matching identity in different stance.",
        "camera": "Ground CCTV (C3)",
        "pid": "0195",
        "rel_path": "AG-ReID.v2/query/P0195T02210A2/P0195T02210A2C3F5221.jpg",
    },
    {
        "id": "p0610_uav",
        "title": "Person 0610 (Aerial UAV)",
        "desc": "Aerial UAV perspective in complex urban background.",
        "camera": "Aerial UAV (C0)",
        "pid": "0610",
        "rel_path": "AG-ReID.v2/query/P0610T04070A0/P0610T04070A0C0F11311.jpg",
    },
    {
        "id": "p0610_cctv",
        "title": "Person 0610 (Ground CCTV)",
        "desc": "CCTV perspective of Person 0610 walking along campus path.",
        "camera": "Ground CCTV (C3)",
        "pid": "0610",
        "rel_path": "AG-ReID.v2/query/P0610T04070A0/P0610T04070A0C3F05521.jpg",
    },
]


@router.get("/info")
async def get_dataset_info() -> Dict[str, Any]:
    """Return empirical dataset summary and statistics."""
    return {
        "dataset_name": "AG-ReID.v2",
        "full_name": "Aerial–Ground Person Re-Identification Benchmark v2",
        "total_images": 100502,
        "total_identities": 1044,
        "splits": {
            "train": {"images": 51530, "identities": 590, "tracklets": 807},
            "query": {"images": 8499, "identities": 649, "tracklets": 808},
            "gallery": {"images": 40473, "identities": 649, "tracklets": 808},
        },
        "camera_sensors": [
            {"code": "C0", "platform": "Aerial UAV Drone", "images": 46779, "pct": 46.5},
            {"code": "C2", "platform": "Wearable Smart Glasses", "images": 36550, "pct": 36.4},
            {"code": "C3", "platform": "Ground CCTV Surveillance", "images": 17173, "pct": 17.1},
        ],
        "altitudes": [
            {"code": "A0", "level": "Low Altitude", "images": 32174, "pct": 32.0},
            {"code": "A1", "level": "Medium Altitude", "images": 38926, "pct": 38.7},
            {"code": "A2", "level": "High Altitude", "images": 29402, "pct": 29.3},
        ],
        "image_resolution": {
            "aspect_ratio": "2:1 standard pedestrian",
            "mean_size": "86.4 x 201.2 px",
            "input_resolution": "256 x 128 px",
        },
    }


@router.get("/samples", response_model=List[DatasetSample])
async def get_dataset_samples():
    """Return list of sample query images with base64 data URLs for 1-click testing."""
    samples = []
    for s in SAMPLE_PROBES:
        img_path = settings.DATASET_ROOT / s["rel_path"]
        b64_str = None
        if img_path.is_file():
            try:
                with open(img_path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("ascii")
                    b64_str = f"data:image/jpeg;base64,{encoded}"
            except Exception:
                pass

        samples.append(
            DatasetSample(
                id=s["id"],
                title=s["title"],
                desc=s["desc"],
                camera=s["camera"],
                pid=s["pid"],
                path=s["rel_path"],
                thumbnail_base64=b64_str,
            )
        )
    return samples

