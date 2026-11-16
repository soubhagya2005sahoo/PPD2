"""Automated API Test Suite for Aerial-Ground Person Re-ID (Standard Library Unittest)."""

import io
import unittest
from PIL import Image
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.model_service import model_service
from backend.app.services.gallery_service import gallery_service


class TestReIDAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Warm up model and gallery before testing."""
        model_service.load_model()
        gallery_service.load_gallery()
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])
        self.assertTrue(data["gallery_loaded"])
        self.assertEqual(data["gallery_count"], 40473)

    def test_project_status_endpoint(self):
        response = self.client.get("/api/project/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["dataset_name"], "AG-ReID.v2")
        self.assertTrue(data["checkpoint_exists"])
        self.assertTrue(data["gallery_cache_exists"])

    def test_dataset_info_endpoint(self):
        response = self.client.get("/api/dataset/info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_images"], 100502)
        self.assertEqual(data["total_identities"], 1044)
        self.assertEqual(len(data["camera_sensors"]), 3)

    def test_dataset_samples_endpoint(self):
        response = self.client.get("/api/dataset/samples")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreaterEqual(len(data), 4)
        self.assertIn("pid", data[0])
        self.assertIn("thumbnail_base64", data[0])

    def test_search_endpoint_with_image(self):
        img = Image.new("RGB", (128, 256), color=(120, 80, 190))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        response = self.client.post(
            "/api/search",
            files={"file": ("test_query.jpg", buf, "image/jpeg")},
            data={"view": "auto", "top_k": 5},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(len(data["matches"]), 5)
        self.assertEqual(data["total_gallery_count"], 40473)
        self.assertEqual(data["matches"][0]["rank"], 1)
        self.assertIn("similarity", data["matches"][0])
        self.assertIn("similarity_pct", data["matches"][0])

    def test_pair_compare_endpoint(self):
        im1 = Image.new("RGB", (128, 256), color=(200, 100, 50))
        im2 = Image.new("RGB", (128, 256), color=(200, 100, 50))

        b1 = io.BytesIO()
        im1.save(b1, format="JPEG")
        b1.seek(0)

        b2 = io.BytesIO()
        im2.save(b2, format="JPEG")
        b2.seek(0)

        response = self.client.post(
            "/api/compare",
            files={
                "file1": ("im1.jpg", b1, "image/jpeg"),
                "file2": ("im2.jpg", b2, "image/jpeg"),
            },
            data={"cam1": "uav", "cam2": "cctv", "threshold": 0.40},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("similarity", data)
        self.assertIn("verdict", data)
        self.assertIn("confidence", data)

    def test_model_info_endpoint(self):
        response = self.client.get("/api/model/info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("26,173,086", data["parameters"]["total"])
        self.assertGreaterEqual(len(data["components"]), 5)

    def test_evaluation_endpoint(self):
        response = self.client.get("/api/evaluation")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("baseline", data)
        self.assertIn("proposed", data)
        self.assertGreaterEqual(len(data["comparison_summary"]), 5)

    def test_security_directory_traversal(self):
        # Percent-encoded path traversal attempt must be blocked with 403
        response = self.client.get("/api/images/gallery/%2e%2e/%2e%2e/etc/passwd")
        self.assertEqual(response.status_code, 403)

        # Non-existent file must return 404
        res404 = self.client.get("/api/images/gallery/nonexistent_image.jpg")
        self.assertEqual(res404.status_code, 404)


if __name__ == "__main__":
    unittest.main()
