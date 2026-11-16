# Aerial–Ground Person Re-Identification Web Application: Implementation & Verification Report

**Project Title**: *View-Aware Multi-Scale Image Enhancement and Feature Fusion for Aerial–Ground Person Re-Identification*  
**Authors**: Soubhagya Ranjan Sahoo (`23110238`) & Aryan Pradhan (`23110197`)  
**Project Guide**: Dr. Rajalaxmi Padhy  
**Academic Benchmark**: AG-ReID.v2 (Aerial–Ground Person Re-Identification v2)  
**Date**: November 13, 2026  

---

## 1. Executive Summary

This report documents the full-stack architecture, empirical validation, and deployment of the web application for the academic research project: **"View-Aware Multi-Scale Image Enhancement and Feature Fusion for Aerial–Ground Person Re-Identification"**.

The application connects the trained PyTorch neural network (`results/checkpoints/proposed_full_best.pth`, 26.17M parameters) with the complete precomputed gallery database of **40,473 L2-normalized 512-dimensional embeddings** extracted from the **AG-ReID.v2** benchmark.

The entire user workflow is streamlined into a single primary action:
$$\text{Query Image Input} \longrightarrow \text{Automated AI Pipeline} \longrightarrow \text{Ranked Retrieved Gallery Matches}$$

---

## 2. Verified Research Assets & Checkpoints

All models, data, and embeddings integrated into the web application are authentic research artifacts:

| Asset | Location / Specification | Status & Verification |
| :--- | :--- | :--- |
| **Proposed Model Weights** | `results/checkpoints/proposed_full_best.pth` | **26.17M Parameters** (Loaded & Verified) |
| **Baseline Model Weights** | `results/checkpoints/best_model.pth` | **24.72M Parameters** (Verified) |
| **Gallery Feature Database** | `features/gallery_cache_proposed.npz` | **40,473 Gallery Vectors** ($512\text{-d}$, L2-normalized) |
| **Dataset Splits** | `datasets/AG-ReID.v2/AG-ReID.v2/` | **100,502 Verified Images** across 1,044 identities |
| **Empirical Evaluation Data** | `results/eval_baseline.json` & `eval_proposed_full.json` | Baseline vs. Proposed CMC & mAP curves |

---

## 3. End-to-End Pipeline Architecture

When the user uploads a person query image and clicks **"FIND MATCHING IMAGES"**, the backend executes the following automated pipeline:

```
[Query Image: JPG/PNG]
         │
         ▼
[Step 1: Input Validation & RGB Conversion]
         │
         ▼
[Step 2: Adaptive Enhancement: CIELAB CLAHE + Gamma (1.2)]
         │
         ▼
[Step 3: Multi-Scale Extraction via ResNet-50 (Stride=1)]
    ├─ Fine Scale   (layer1: 256 channels)
    ├─ Medium Scale (layer2: 512 channels)
    └─ Coarse Scale (layer4: 2048 channels)
         │
         ▼
[Step 4: Dimension Projection into Common d=512 Space]
         │
         ▼
[Step 5: View-Aware Dynamic Scale Attention Gate]
    Conditioned on Viewpoint: UAV (C0) / Wearable (C2) / CCTV (C3)
         │
         ▼
[Step 6: Compact 1D Latent Feature Diffusion Refinement (2 Steps)]
         │
         ▼
[Step 7: BNNeck & L2-Normalization -> 512-dim Query Vector]
         │
         ▼
[Step 8: Matrix Dot-Product Cosine Similarity Search (<10ms)]
    sims = np.dot(gallery_embeddings_40473x512, query_vec_512)
         │
         ▼
[Step 9: Ranked Top-5 / Top-10 Results with Metadata & Similarity %]
```

---

## 4. Frontend Design & Component Structure

Built with **React 19**, **Vite**, **Tailwind CSS**, **React Router**, **Lucide Icons**, and **Recharts**:

1. **Left Navigation Sidebar**:
   - `Home`: Research synopsis and dataset statistics cards.
   - `Image Search` (*Primary Console*):
     - Drag-and-drop query upload zone with dimension validation.
     - Viewpoint selector: *Unknown / Automatic*, *Aerial UAV (C0)*, *Ground CCTV (C3)*, *Wearable (C2)*.
     - Results selector: *Top-5 (Default)* or *Top-10*.
     - 1-Click Curated Test Probes: $P0867$ Aerial UAV, $P0867$ Ground CCTV, $P0195$, $P0610$.
     - Primary Action Button: **"FIND MATCHING IMAGES"**.
     - 4-Stage visual inference progress bar.
     - Tab for *1-to-1 Cross-View Verification* with similarity percentage and decision threshold ($0.40$).
   - `Retrieval Results`: Query preview card side-by-side with Top-5 / Top-10 ranked gallery matches, similarity % badges, camera sensors, and modal inspection dialog.
   - `Model Information`: Layer-by-layer architectural parameter allocation ($26.17\text{M}$ total), ResNet-50 stages, and dynamic scale attention weight table.
   - `Evaluation`: Interactive CMC accuracy curves (Rank-1, Rank-5, Rank-10) and directional aerial-to-ground comparison.
   - `About Project`: Project title, student details (Soubhagya Ranjan Sahoo & Aryan Pradhan), guide (Dr. Rajalaxmi Padhy), and AG-ReID.v2 citation.

---

## 5. Automated Backend Test Suite Verification

The backend API was tested via `backend/tests/test_api.py` covering all primary workflows:

```bash
PYTHONPATH=. .venv/bin/python backend/tests/test_api.py
```

**Results**:
- `test_health_endpoint`: Passed (reports healthy status, model loaded, 40,473 gallery items).
- `test_project_status_endpoint`: Passed (validates checkpoint and gallery paths).
- `test_dataset_info_endpoint`: Passed (verifies 100,502 images and 1,044 identities).
- `test_dataset_samples_endpoint`: Passed (returns curated query samples with base64 previews).
- `test_search_endpoint_with_image`: Passed (executes search against 40,473 gallery images in <15ms).
- `test_pair_compare_endpoint`: Passed (performs 1-to-1 cross-view cosine verification).
- `test_model_info_endpoint`: Passed (returns 26.17M parameter specifications).
- `test_evaluation_endpoint`: Passed (returns empirical benchmark metrics).
- `test_security_directory_traversal`: Passed (blocks unauthorized path traversal with HTTP 403).

**Summary**: 9 tests passed in 0.553 seconds.

---

## 6. How to Run the Application

### Method 1: Single-Command Production Server (Recommended)
```bash
./start_app.sh
```
Or directly with Python:
```bash
python run_server.py
```
*Note: If port 8000 is occupied by another process, the server automatically detects an open port (e.g., 8001) and displays the active URL.*

### Method 2: Development Mode (Vite Hot-Reload)
In Terminal 1 (Backend):
```bash
python run_server.py --port 8000
```
In Terminal 2 (Frontend Dev Server):
```bash
cd frontend
npm run dev
```
Open browser at: `http://localhost:5173`

