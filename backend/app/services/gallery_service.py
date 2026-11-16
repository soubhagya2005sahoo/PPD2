"""Gallery Service: In-memory cache and cosine similarity search engine.

Loads precomputed L2-normalized 512-d feature embeddings for all 40,473 AG-ReID.v2
gallery images and computes cosine similarity in <15 milliseconds.
"""

import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from backend.app.core.config import settings
from backend.app.schemas import SearchResultItem


class GalleryService:
    """In-memory gallery database and vector search engine."""

    def __init__(self):
        self.feats: Optional[np.ndarray] = None
        self.pids: Optional[np.ndarray] = None
        self.cams: Optional[np.ndarray] = None
        self.rel_paths: Optional[np.ndarray] = None
        self.is_loaded: bool = False
        self.count: int = 0

    def load_gallery(self):
        """Load gallery embeddings and metadata into memory."""
        if self.is_loaded:
            return

        cache_path = settings.GALLERY_CACHE
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"Gallery cache not found at: {cache_path}. "
                "Run `python scripts/build_gallery.py` to precompute gallery embeddings."
            )

        print(f"[GalleryService] Loading gallery cache from: {cache_path}...")
        t0 = time.time()
        data = np.load(cache_path, allow_pickle=True)

        self.feats = data["feats"].astype(np.float32)
        self.pids = data["pids"]
        self.cams = data["cams"].astype(np.int32)
        self.rel_paths = data["rel_paths"]
        self.count = len(self.feats)

        # Ensure unit L2 normalization
        norms = np.linalg.norm(self.feats, axis=1, keepdims=True)
        if not np.allclose(norms, 1.0, atol=1e-3):
            self.feats = self.feats / np.maximum(norms, 1e-12)

        self.is_loaded = True
        elapsed = (time.time() - t0) * 1000.0
        print(f"[GalleryService] ✓ Loaded {self.count:,} gallery vectors ({self.feats.shape[1]}-d) in {elapsed:.1f}ms.")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        query_cam_id: Optional[int] = None,
        exclude_same_camera: bool = False,
    ) -> List[SearchResultItem]:
        """Perform fast matrix-vector cosine similarity search across all gallery images.

        Args:
            query_embedding: 512-dim unit L2-normalized query vector.
            top_k: Number of nearest matches to retrieve (default: 5 or 10).
            query_cam_id: Optional query camera ID (0: UAV, 2: Wearable, 3: CCTV).
            exclude_same_camera: If True, only returns cross-view matches.

        Returns:
            List of SearchResultItem ordered by rank.
        """
        if not self.is_loaded:
            self.load_gallery()

        # In-memory dot product across all 40,473 vectors (<10ms)
        scores = np.dot(self.feats, query_embedding)

        # Optional camera filter
        if exclude_same_camera and query_cam_id is not None:
            valid_mask = self.cams != query_cam_id
            scores = np.where(valid_mask, scores, -999.0)

        # Retrieve top indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[SearchResultItem] = []
        for rank_idx, idx in enumerate(top_indices, start=1):
            cam_val = int(self.cams[idx])
            sim_val = float(scores[idx])
            sim_val = max(-1.0, min(1.0, sim_val))
            rel_path_str = str(self.rel_paths[idx])

            cam_name = settings.CAM_NAMES.get(cam_val, f"Camera {cam_val}")
            is_cross_view = (query_cam_id is not None) and (cam_val != query_cam_id)

            item = SearchResultItem(
                rank=rank_idx,
                similarity=round(sim_val, 4),
                similarity_pct=f"{sim_val * 100:.1f}%",
                pid=str(self.pids[idx]),
                cam_id=cam_val,
                cam_name=cam_name,
                is_cross_view=is_cross_view,
                path=rel_path_str,
                image_url=f"/api/images/gallery/{rel_path_str}",
            )
            results.append(item)

        return results


gallery_service = GalleryService()

