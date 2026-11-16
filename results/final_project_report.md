# Final Project Report
## View-Aware Multi-Scale Image Enhancement and Feature Fusion for Aerial–Ground Person Re-Identification

**Students:** Soubhagya Ranjan Sahoo (`23110238`), Aryan Pradhan (`23110197`)  
**Guide:** Dr. Rajalaxmi Padhy  
**Benchmark Dataset:** AG-ReID.v2 (`100,502` images, `1,044` unique Person IDs)  
**Hardware Platform:** Apple Silicon M5 (macOS, PyTorch MPS GPU Acceleration)

---

## 1. Executive Summary

This project develops and validates a complete, computationally accessible **View-Aware Multi-Scale Image Enhancement, Feature Fusion, and Efficient Latent Diffusion** framework for **Aerial–Ground Person Re-Identification (AG-ReID)**.

Compared to our standard ResNet-50 baseline on AG-ReID.v2 (`8,499` query images matched against `40,473` gallery images), our full proposed framework achieves substantial accuracy improvements across **all evaluation metrics (Rank-1, Rank-5, mAP, and mINP)** and **both cross-view directions (Aerial → Ground and Ground → Aerial)** while remaining **>32× smaller** than pixel-space diffusion models (`26.17M` parameters vs. `~860M+` parameters in Stable Diffusion / SD-ReID U-Nets) and running at **`158.0 FPS`** on laptop Apple Silicon hardware.

---

## 2. End-to-End Methodology Pipeline

```text
Input Image (UAV C0 / Wearable C2 / CCTV C3)
   │
   ▼
[Step 4: Image Enhancement]
   ├─ CIELAB CLAHE on L-channel (clip_limit=2.0, grid=8×8)
   └─ Gamma Power-Law Correction via LUT (γ=1.2)
   │
   ▼
[Step 5: ResNet-50 Multi-Scale Backbone (last_stride=1)]
   ├─ Fine Scale   : layer1 (256 × 64 × 32)  ──▶ GAP ──▶ Linear(256 → 512)
   ├─ Medium Scale : layer2 (512 × 32 × 16)  ──▶ GAP ──▶ Linear(512 → 512)
   └─ Coarse Scale : layer4 (2048 × 16 × 8)  ──▶ GAP ──▶ Linear(2048 → 512)
   │
   ▼
[Step 6 & 7: View-Aware Conditioning & Dynamic Multi-Scale Fusion]
   ├─ Camera / Predicted View Embedding (C0 Aerial, C2 Wearable, C3 CCTV)
   ├─ View-Aware Scale Attention Gate: α(x, v) = [α_fine, α_medium, α_coarse]
   └─ Zero-Initialized Residual FiLM Calibration: (1 + Δγ_v) ⊙ z + β_v
   │
   ▼
[Step 7: Efficient View-Conditioned Latent Feature Diffusion (0.63M params)]
   ├─ Operates in compact 512-dim identity latent space (not pixel space)
   └─ 2-step deterministic DDIM denoising refinement at inference
   │
   ▼
[Metric Learning Head]
   └─ BNNeck + Label-Smoothed Cross-Entropy + Batch-Hard Triplet + Auxiliary View & Diffusion Losses
```

---

## 3. Experimental Results on AG-ReID.v2

### 3.1 Overall Benchmark Performance (8,499 Query × 40,473 Gallery)

| Model | Rank-1 (%) | Rank-5 (%) | Rank-10 (%) | mAP (%) | mINP (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Raw ResNet-50)** | 22.94% | 35.60% | 42.33% | 12.61% | 1.11% |
| **Proposed Full Framework** | **26.89%** | **42.96%** | **50.94%** | **18.49%** | **2.72%** |
| **Absolute Gain** | **+3.95%** | **+7.36%** | **+8.61%** | **+5.88%** | **+1.61%** |
| **Relative Gain** | **+17.2%** | **+20.7%** | **+20.3%** | **+46.6%** | **+145.0%** |

### 3.2 Cross-View Breakdown (Aerial ↔ Ground)

| Model | Aerial → Ground Rank-1 | Aerial → Ground mAP | Aerial → Ground mINP | Ground → Aerial Rank-1 | Ground → Aerial mAP | Ground → Aerial mINP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Raw ResNet-50)** | 47.13% | 23.72% | 3.01% | 39.80% | 24.36% | 2.87% |
| **Proposed Full Framework** | **53.45%** | **31.81%** | **5.67%** | **49.27%** | **32.25%** | **6.77%** |
| **Absolute Gain** | **+6.32%** | **+8.09%** | **+2.66%** | **+9.47%** | **+7.89%** | **+3.90%** |

### 3.3 Learned View-Aware Scale Attention Weights

During evaluation, our [`ViewAwareModule`](file:///Users/soubhagyaranjansahoo/person%20re%20identification%20/src/models/view_aware.py) automatically adjusted how much weight each spatial scale receives per camera platform:

| Camera Platform | Fine (`layer1`) | Medium (`layer2`) | Coarse (`layer4`) |
| :--- | :---: | :---: | :---: |
| **Aerial UAV (`C0`)** | 0.1745 | 0.1956 | 0.6298 |
| **Wearable (`C2`)** | 0.1799 | 0.1989 | 0.6212 |
| **Ground CCTV (`C3`)** | 0.1655 | 0.1928 | 0.6417 |

### 3.4 Image Enhancement Quality Analysis (Step 4)

Applied strictly to the lightness ($L$) channel in CIELAB space to preserve true clothing colors:
- **UAV Aerial (`C0`)**: RMS contrast increased by **+11.2%** (`0.1986 → 0.2209`), Shannon entropy increased by **+0.37 bits** (`7.294 → 7.668`).
- **Ground CCTV (`C3`)**: RMS contrast increased by **+14.8%** (`0.2021 → 0.2320`), Shannon entropy increased by **+0.16 bits**.

### 3.5 Computational Efficiency vs. Heavy Diffusion Baselines

| Architecture | Parameters (M) | Inference Latency (ms/img) | Throughput (FPS) | Hardware Required |
| :--- | :---: | :---: | :---: | :--- |
| **Pixel/Spatial Diffusion (SD-ReID U-Net ref.)** | ~860.0+ M | Hundreds of ms (20–50 steps) | < 5 FPS | Multi-GPU / High VRAM Server |
| **Baseline ResNet-50** | 24.72 M | 4.02 ms | 249.0 FPS | Laptop (Apple M5 MPS) |
| **Proposed Full (View-Aware + Latent Diffusion)** | **26.17 M** | **6.33 ms** | **158.0 FPS** | **Laptop (Apple M5 MPS)** |

*The compact `LatentFeatureDiffusion` block adds only **0.63M parameters** and performs **2 deterministic DDIM steps** in 512-dim latent space rather than expensive spatial pixel generation.*

---

## 4. Key Artifacts & Visualizations

- **Overall Metric Comparison Chart**: [`results/final_visualizations/model_comparison_metrics.png`](file:///Users/soubhagyaranjansahoo/person%20re%20identification%20/results/final_visualizations/model_comparison_metrics.png)
- **Cross-View Comparison Chart**: [`results/final_visualizations/cross_view_comparison.png`](file:///Users/soubhagyaranjansahoo/person%20re%20identification%20/results/final_visualizations/cross_view_comparison.png)
- **View-Aware Scale Attention Chart**: [`results/final_visualizations/view_scale_weights.png`](file:///Users/soubhagyaranjansahoo/person%20re%20identification%20/results/final_visualizations/view_scale_weights.png)
- **Parameter Efficiency Chart**: [`results/final_visualizations/efficiency_vs_diffusion.png`](file:///Users/soubhagyaranjansahoo/person%20re%20identification%20/results/final_visualizations/efficiency_vs_diffusion.png)
- **Baseline Retrieval Examples**: [`results/retrieval_examples/`](file:///Users/soubhagyaranjansahoo/person%20re%20identification%20/results/retrieval_examples)
- **Proposed Retrieval Examples**: [`results/retrieval_examples_proposed/`](file:///Users/soubhagyaranjansahoo/person%20re%20identification%20/results/retrieval_examples_proposed)
- **Best Trained Proposed Checkpoint**: [`results/checkpoints/proposed_full_best.pth`](file:///Users/soubhagyaranjansahoo/person%20re%20identification%20/results/checkpoints/proposed_full_best.pth)
- **Evaluation JSON Logs**: [`results/eval_baseline.json`](file:///Users/soubhagyaranjansahoo/person%20re%20identification%20/results/eval_baseline.json) & [`results/eval_proposed_full.json`](file:///Users/soubhagyaranjansahoo/person%20re%20identification%20/results/eval_proposed_full.json)

---

## 5. Verification Checklist

- [x] **Step 1**: Environment & Apple Silicon MPS verification ([`src/check_environment.py`](file:///Users/soubhagyaranjansahoo/person%20re%20identification%20/src/check_environment.py))
- [x] **Step 2**: AG-ReID.v2 dataset parsing & analysis ([`results/dataset_analysis.md`](file:///Users/soubhagyaranjansahoo/person%20re%20identification%20/results/dataset_analysis.md))
- [x] **Step 3**: Baseline ResNet-50 training & evaluation ([`results/baseline_report.md`](file:///Users/soubhagyaranjansahoo/person%20re%20identification%20/results/baseline_report.md))
- [x] **Step 4**: CLAHE & Gamma image enhancement ([`results/step4_image_enhancement_report.md`](file:///Users/soubhagyaranjansahoo/person%20re%20identification%20/results/step4_image_enhancement_report.md))
- [x] **Step 5**: Multi-scale feature extraction (`layer1`, `layer2`, `layer4`) ([`results/step5_multiscale_feature_extraction_report.md`](file:///Users/soubhagyaranjansahoo/person%20re%20identification%20/results/step5_multiscale_feature_extraction_report.md))
- [x] **Step 6**: Multi-scale feature projection & fusion ([`results/step6_feature_fusion_report.md`](file:///Users/soubhagyaranjansahoo/person%20re%20identification%20/results/step6_feature_fusion_report.md))
- [x] **Step 7**: View-Aware conditioning, efficient latent feature diffusion, `mINP` & cross-view evaluation, full training & comparison report ([`results/final_project_report.md`](file:///Users/soubhagyaranjansahoo/person%20re%20identification%20/results/final_project_report.md))

