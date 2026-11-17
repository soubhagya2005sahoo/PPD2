# View-Aware Multi-Scale Image Enhancement and Feature Fusion for Aerial–Ground Person Re-Identification

**Authors:** Soubhagya Ranjan Sahoo (`23110238`), Aryan Pradhan (`23110197`)  
**Guide:** Dr. Rajalaxmi Padhy  

---

## Web Application Quick Start 🚀

A complete full-stack web dashboard (React 19 + FastAPI + PyTorch) connects the trained model and the 40,473 AG-ReID.v2 gallery database.

### Single-Command Start:
```bash
./start_app.sh
```
Or directly with Python:
```bash
python run_server.py
```
Open your browser at **`http://localhost:8000`** (or the port displayed if port 8000 is occupied).

### Core Features:
- **Image Search (Primary Page)**: Upload any person image or select a 1-click test probe ($P0867$, $P0195$, $P0610$), choose optional viewpoint (`UAV`, `CCTV`, `Wearable`, or `Automatic`), and click **"FIND MATCHING IMAGES"**.
- **Automated AI Pipeline**: Preprocessing $\to$ CLAHE & Gamma enhancement $\to$ ResNet-50 multi-scale extraction $\to$ View-aware scale attention $\to$ Latent diffusion refinement $\to$ Cosine similarity ranking against 40,473 gallery images in <15ms.
- **Top-5 & Top-10 Retrieval**: View ranked matches with cosine similarity %, identity tags, and cross-view indicators.
- **1-to-1 Cross-View Verification**: Compare any two images across Aerial UAV, CCTV, and Wearable cameras with calibrated similarity scoring and automated verdict. See full architectural documentation in [`docs/CROSS_VIEW_VERIFICATION.md`](docs/CROSS_VIEW_VERIFICATION.md).
- **Model Architecture & Evaluation**: Interactive parameter allocation tables (26.17M params) and CMC curves.

---

## Project Overview
This research project implements and evaluates a lightweight, **View-Aware Multi-Scale Image Enhancement, Feature Fusion, and Efficient Latent Diffusion** framework for Aerial–Ground Person Re-Identification (AG-ReID.v2).

The framework addresses severe viewpoint variations (UAV `C0` vs. Wearable `C2` vs. Ground CCTV `C3`), scale discrepancies, and low-contrast / shadow degradations while running efficiently at **158 FPS** on Apple Silicon (`mps`).

## Current Project Status: Completed ✅
All 7 project phases are implemented, tested, trained, and evaluated on **AG-ReID.v2** (`100,502` images; `51,530` train / `8,499` query / `40,473` gallery):

| Model | Rank-1 | mAP | mINP | Aerial → Ground Rank-1 | Ground → Aerial Rank-1 | Params | Speed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline ResNet-50** | 22.94% | 12.61% | 1.11% | 47.13% | 39.80% | 24.72M | 249.0 FPS |
| **Proposed Full Model** | **26.89%** | **18.49%** | **2.72%** | **53.45%** | **49.27%** | **26.17M** | **158.0 FPS** |

See the complete final findings in [`results/final_project_report.md`](results/final_project_report.md).

---

## Directory Structure
```text
Aerial_Ground_Person_ReID/
├── configs/
│   ├── baseline.yaml
│   ├── enhancement.yaml
│   ├── multiscale.yaml
│   ├── fusion.yaml
│   └── proposed.yaml
├── datasets/AG-ReID.v2/          # Official AG-ReID.v2 dataset (untouched)
├── src/
│   ├── data/                     # Dataset loader, transforms (+CLAHE/Gamma), PK sampler
│   ├── enhancement/              # Step 4: CIELAB CLAHE + LUT Gamma Correction
│   ├── features/                 # Step 5: GAP pooling & feature map visualizations
│   ├── losses/                   # Batch-Hard Triplet Loss
│   ├── models/
│   │   ├── resnet50_baseline.py
│   │   ├── resnet50_multiscale.py
│   │   ├── feature_projection.py
│   │   ├── feature_fusion.py
│   │   ├── view_aware.py         # Step 7: View-Aware Scale Attention & FiLM
│   │   ├── latent_diffusion.py   # Step 7: 0.63M Latent Feature Diffusion
│   │   └── view_aware_fusion_model.py
│   ├── evaluation/evaluator.py   # Rank-1/5/10, mAP, mINP, Aerial<->Ground, Speed
│   ├── train_baseline.py
│   ├── train_proposed.py
│   ├── predict.py                # Interactive CLI inference & visual retrieval
│   ├── visualize_final_results.py
│   └── visualize_proposed_retrieval.py
├── tests/
│   ├── test_image_processing.py
│   ├── test_metrics.py           # Unit tests for CMC, mAP, and mINP
│   ├── test_multiscale.py
│   ├── test_fusion.py
│   └── test_proposed_model.py    # Unit tests for View-Aware + Latent Diffusion
└── results/
    ├── final_project_report.md
    ├── final_visualizations/
    ├── retrieval_examples/
    └── retrieval_examples_proposed/
```

---

## Quick Commands

### 1. Activate Environment
```bash
source .venv/bin/activate
```

### 2. Run Unit Tests
```bash
python tests/test_metrics.py
python tests/test_multiscale.py
python tests/test_fusion.py
python tests/test_proposed_model.py
```

### 3. Evaluate Trained Models
```bash
# Baseline ResNet-50
python src/evaluation/evaluator.py \
  --model_type baseline \
  --checkpoint results/checkpoints/best_model.pth \
  --exp_name Baseline_ResNet50

# Proposed View-Aware Multi-Scale + Latent Diffusion Model
python src/evaluation/evaluator.py \
  --model_type proposed \
  --checkpoint results/checkpoints/proposed_full_best.pth \
  --exp_name Proposed_Full \
  --use_enhancement --use_view_aware --use_diffusion
```

### 4. Interactive Inference & Image Retrieval (Custom Input)
```bash
# Pair comparison (verify whether two images are the same person)
python src/predict.py \
  --img1 datasets/AG-ReID.v2/AG-ReID.v2/query/P0867T04070A0/P0867T04070A0C0F13861.jpg \
  --img2 datasets/AG-ReID.v2/AG-ReID.v2/query/P0867T04070A0/P0867T04070A0C3F07531.jpg \
  --out results/custom_pair_result.png

# Query search against the 40,473-image gallery
python src/predict.py \
  --query datasets/AG-ReID.v2/AG-ReID.v2/query/P0867T04070A0/P0867T04070A0C0F13861.jpg \
  --topk 10 \
  --out results/custom_query_search.png
```

### 5. Re-Train Full Proposed Model
```bash
python src/train_proposed.py \
  --exp_name proposed_full \
  --model_type proposed \
  --use_enhancement --use_random_erasing --use_view_aware --use_diffusion \
  --epochs 2 --p_pids 16 --k_instances 4
```
