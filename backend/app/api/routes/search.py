"""Search and Re-Identification Routes."""

import io
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import FileResponse
from PIL import Image

from backend.app.core.config import settings
from backend.app.schemas import SearchResponse, PairCompareResponse
from backend.app.services.retrieval_service import retrieval_service
from backend.app.services.model_service import model_service

router = APIRouter(tags=["Search & Retrieval"])


@router.post("/search", response_model=SearchResponse)
async def search_query(
    file: UploadFile = File(..., description="Query person image (JPG/PNG)"),
    view: Optional[str] = Form("auto", description="Optional viewpoint: uav, cctv, wearable, auto"),
    top_k: int = Form(5, description="Number of results to retrieve (5 or 10)"),
    exclude_same_camera: bool = Form(False, description="Filter out same-camera matches"),
):
    """Execute complete automated Person Re-Identification on uploaded query image."""
    # 1. Validate content type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents))
        pil_img.verify()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {str(e)}")

    # 2. Run automated pipeline
    top_k = max(1, min(50, top_k))
    response = retrieval_service.process_query(
        img=pil_img,
        view_str=view,
        top_k=top_k,
        exclude_same_camera=exclude_same_camera,
    )
    return response


@router.get("/results/{search_id}", response_model=SearchResponse)
async def get_search_result(search_id: str):
    """Retrieve previously computed search result by ID."""
    result = retrieval_service.get_result(search_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Search result expired or not found.")
    return result


@router.post("/compare", response_model=PairCompareResponse)
async def compare_pair(
    file1: UploadFile = File(..., description="First person image"),
    file2: UploadFile = File(..., description="Second person image"),
    cam1: Optional[str] = Form("auto"),
    cam2: Optional[str] = Form("auto"),
    threshold: float = Form(0.40, description="Decision threshold (default: 0.40)"),
):
    """Perform 1-to-1 cross-view pairwise comparison and person verification."""
    try:
        data1 = await file1.read()
        im1 = Image.open(io.BytesIO(data1)).convert("RGB")
        data2 = await file2.read()
        im2 = Image.open(io.BytesIO(data2)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decoding failed: {str(e)}")

    cam1_id = retrieval_service.parse_view_str(cam1)
    cam2_id = retrieval_service.parse_view_str(cam2)

    result = model_service.compare_images(
        img1=im1,
        img2=im2,
        cam1=cam1_id,
        cam2=cam2_id,
        threshold=threshold,
    )
    return PairCompareResponse(**result)


@router.get("/images/gallery/{file_path:path}")
async def serve_gallery_image(file_path: str):
    """Safely stream an image from the AG-ReID.v2 dataset."""
    safe_path = file_path.lstrip("/")
    target_path = (settings.DATASET_ROOT / safe_path).resolve()

    # Prevent path traversal attacks
    try:
        target_path.relative_to(settings.DATASET_ROOT.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied: invalid file path.")

    if not target_path.is_file():
        raise HTTPException(status_code=404, detail="Gallery image not found.")

    return FileResponse(target_path, media_type="image/jpeg")
