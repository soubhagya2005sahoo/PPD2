"""Health and System Status API Routes."""

from fastapi import APIRouter
from backend.app.core.config import settings
from backend.app.schemas import HealthResponse, ProjectStatusResponse
from backend.app.services.model_service import model_service
from backend.app.services.gallery_service import gallery_service

router = APIRouter(tags=["Status & Health"])


@router.get("/health", response_model=HealthResponse)
async def get_health():
    """Healthcheck endpoint reporting model and gallery readiness."""
    return HealthResponse(
        status="healthy",
        service="Aerial-Ground Person Re-ID Engine",
        device=str(model_service.device),
        model_loaded=model_service.is_loaded,
        gallery_loaded=gallery_service.is_loaded,
        gallery_count=gallery_service.count,
    )


@router.get("/project/status", response_model=ProjectStatusResponse)
async def get_project_status():
    """Detailed research project status, checkpoint, and gallery details."""
    params_str = f"{model_service.total_params / 1e6:.2f}M" if model_service.is_loaded else "26.17M"
    return ProjectStatusResponse(
        project_title=settings.PROJECT_NAME,
        dataset_name="AG-ReID.v2",
        dataset_path=str(settings.DATASET_ROOT),
        checkpoint_exists=settings.CHECKPOINT_PROPOSED.is_file(),
        checkpoint_path=str(settings.CHECKPOINT_PROPOSED),
        gallery_cache_exists=settings.GALLERY_CACHE.is_file(),
        gallery_cache_path=str(settings.GALLERY_CACHE),
        total_parameters=params_str,
        device=str(model_service.device),
    )

