"""API Router Initialization."""

from fastapi import APIRouter
from backend.app.api.routes.health import router as health_router
from backend.app.api.routes.dataset import router as dataset_router
from backend.app.api.routes.search import router as search_router
from backend.app.api.routes.model import router as model_router
from backend.app.api.routes.evaluation import router as eval_router
from backend.app.api.routes.enhancement import router as enhance_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(dataset_router)
api_router.include_router(search_router)
api_router.include_router(model_router)
api_router.include_router(eval_router)
api_router.include_router(enhance_router)

