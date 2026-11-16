"""Image Enhancement API Routes (CLAHE + Gamma Correction)."""

import io
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from PIL import Image

from backend.app.schemas import EnhancementResponse
from backend.app.services.enhancement_service import enhancement_service

router = APIRouter(prefix="/enhance", tags=["Image Enhancement"])


@router.post("", response_model=EnhancementResponse)
async def enhance_image(
    file: UploadFile = File(..., description="Query or test image"),
    clip_limit: float = Form(2.0, description="CLAHE clip limit (default: 2.0)"),
    gamma: float = Form(1.2, description="Gamma correction coefficient (default: 1.2)"),
):
    """Generate Raw, CLAHE, Gamma, and Combined enhanced images with contrast and entropy metrics."""
    try:
        data = await file.read()
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decoding failed: {str(e)}")

    result = enhancement_service.process_all_variations(
        img=pil_img,
        clip_limit=clip_limit,
        gamma=gamma,
    )
    result["status"] = "success"
    return EnhancementResponse(**result)

