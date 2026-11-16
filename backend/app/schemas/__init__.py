"""Pydantic Schemas for API Requests and Responses."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str
    device: str
    model_loaded: bool
    gallery_loaded: bool
    gallery_count: int


class ProjectStatusResponse(BaseModel):
    project_title: str
    dataset_name: str
    dataset_path: str
    checkpoint_exists: bool
    checkpoint_path: str
    gallery_cache_exists: bool
    gallery_cache_path: str
    total_parameters: str
    device: str


class DatasetSample(BaseModel):
    id: str
    title: str
    desc: str
    camera: str
    pid: str
    path: str
    thumbnail_base64: Optional[str] = None


class SearchResultItem(BaseModel):
    rank: int
    similarity: float
    similarity_pct: str
    pid: str
    cam_id: int
    cam_name: str
    is_cross_view: bool
    path: str
    image_base64: Optional[str] = None
    image_url: str


class SearchResponse(BaseModel):
    search_id: str
    status: str
    search_time_ms: float
    total_gallery_count: int
    detected_view: str
    query_view: str
    query_raw_base64: str
    query_enhanced_base64: str
    matches: List[SearchResultItem]


class PairCompareResponse(BaseModel):
    status: str
    similarity: float
    similarity_pct: str
    threshold: float
    is_same: bool
    verdict: str
    confidence: str
    view1: str
    view2: str
    time_ms: float
    im1_raw: str
    im1_enhanced: str
    im2_raw: str
    im2_enhanced: str


class ImageStats(BaseModel):
    mean: float
    std: float
    rms_contrast: float
    entropy_bits: float


class EnhancementImprovements(BaseModel):
    mean_change: float
    std_change: float
    rms_contrast_gain: float
    entropy_gain: float


class EnhancementResponse(BaseModel):
    status: str
    raw_image: str
    clahe_only: str
    gamma_only: str
    combined: str
    raw_stats: ImageStats
    enhanced_stats: ImageStats
    improvements: EnhancementImprovements

