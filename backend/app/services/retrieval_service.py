"""Retrieval Service: Coordinates query preprocessing, feature extraction, and search."""

import io
import time
import uuid
from typing import Dict, Any, Optional
from PIL import Image

from backend.app.core.config import settings
from backend.app.schemas import SearchResponse, SearchResultItem
from backend.app.services.enhancement_service import enhancement_service, pil_to_base64
from backend.app.services.model_service import model_service
from backend.app.services.gallery_service import gallery_service


VIEW_STR_TO_CAM = {
    "uav": 0,
    "aerial": 0,
    "c0": 0,
    "wearable": 2,
    "c2": 2,
    "cctv": 3,
    "ground": 3,
    "c3": 3,
    "auto": None,
    "unknown": None,
}


class RetrievalService:
    """Manages the full inference and retrieval pipeline for incoming query images."""

    def __init__(self):
        self._history_cache: Dict[str, SearchResponse] = {}

    def parse_view_str(self, view_str: Optional[str]) -> Optional[int]:
        if not view_str:
            return None
        return VIEW_STR_TO_CAM.get(view_str.strip().lower(), None)

    def process_query(
        self,
        img: Image.Image,
        view_str: Optional[str] = None,
        top_k: int = 5,
        exclude_same_camera: bool = False,
    ) -> SearchResponse:
        """Execute full end-to-end aerial-ground Re-ID retrieval.

        Pipeline:
        1. Query validation and RGB normalization
        2. Visual enhancement preview (CLAHE + Gamma)
        3. Feature extraction via Proposed View-Aware Model (MPS/GPU/CPU)
        4. Cosine similarity matching against 40,473 AG-ReID.v2 gallery database
        5. Top-K ranking and response packaging
        """
        t_start = time.time()
        search_id = str(uuid.uuid4())

        # Ensure image is in RGB format
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Camera view mapping
        cam_id = self.parse_view_str(view_str)

        # 1. Base64 previews for user inspection
        raw_b64 = pil_to_base64(img, quality=88)
        enhanced_pil = enhancement_service.enhance_pil(img)
        enhanced_b64 = pil_to_base64(enhanced_pil, quality=88)

        # 2. Extract unit L2-normalized 512-dim embedding from proposed model
        embedding, model_info = model_service.extract_embedding(img, cam_id=cam_id)

        # 3. Fast gallery search across 40,473 vectors
        matches = gallery_service.search(
            query_embedding=embedding,
            top_k=top_k,
            query_cam_id=cam_id,
            exclude_same_camera=exclude_same_camera,
        )

        elapsed_ms = (time.time() - t_start) * 1000.0

        detected_view = model_info.get("detected_view", "Unknown")
        query_view_display = (
            settings.CAM_NAMES.get(cam_id, "Unknown / Automatic")
            if cam_id is not None
            else f"Auto-Detected: {detected_view}"
        )

        response = SearchResponse(
            search_id=search_id,
            status="success",
            search_time_ms=round(elapsed_ms, 2),
            total_gallery_count=gallery_service.count,
            detected_view=detected_view,
            query_view=query_view_display,
            query_raw_base64=raw_b64,
            query_enhanced_base64=enhanced_b64,
            matches=matches,
        )

        # Store in LRU cache
        if len(self._history_cache) > 50:
            oldest_key = next(iter(self._history_cache))
            self._history_cache.pop(oldest_key, None)
        self._history_cache[search_id] = response

        return response

    def get_result(self, search_id: str) -> Optional[SearchResponse]:
        """Fetch previously computed retrieval result by ID."""
        return self._history_cache.get(search_id, None)


retrieval_service = RetrievalService()

