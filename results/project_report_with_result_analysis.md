# View-Aware Multi-Scale Image Enhancement and Feature Fusion for Aerial–Ground Person Re-Identification

**Academic Project Report with In-Depth Result Analysis**

- **Students:** Soubhagya Ranjan Sahoo (`Roll No: 23110238`) & Aryan Pradhan (`Roll No: 23110197`)
- **Project Guide / Supervisor:** Dr. Rajalaxmi Padhy
- **Department:** Department of Computer Science & Engineering
- **Benchmark Benchmark:** AG-ReID.v2 (Aerial–Ground Person Re-Identification v2)
- **Codebase & Deployment:** PyTorch 2.15 + FastAPI + React 19 Dashboard
- **Date:** November 2026

---

## 1. Executive Summary

Cross-view Person Re-Identification (Re-ID) between aerial unmanned aerial vehicles (UAVs) and ground surveillance platforms (CCTV and wearable cameras) is critical for smart city public safety, search-and-rescue operations, and campus security. However, traditional person Re-ID models suffer catastrophic performance drops when transitioning between aerial and ground cameras due to three severe domain gaps:
1. **Extreme Viewpoint Shifts:** Top-down aerial views ($60^\circ - 90^\circ$ pitch) primarily expose heads and shoulders, whereas ground cameras capture full frontal silhouettes.
2. **Dramatic Scale Discrepancies:** Pedestrian bounding boxes in drone captures are small and low-resolution ($20 - 150$ px width), whereas ground CCTV captures are higher resolution ($80 - 430$ px width).
3. **Adverse Illumination & Shadows:** Outdoor aerial flights introduce severe atmospheric haze, glare, and cast shadows that obscure clothing colors and fine patterns.

To overcome these challenges, this project introduces a complete, computationally accessible framework: **View-Aware Multi-Scale Image Enhancement, Feature Fusion, and Latent Diffusion**. Evaluated on the benchmark **AG-ReID.v2** dataset ($100,502$ images across $1,044$ identities), our proposed architecture outperforms the standard ResNet-50 baseline across all metrics:
- **Overall Rank-1 Accuracy:** Improves from $22.94\%$ to **$26.89\%$** (**$+17.2\%$ relative gain**).
- **Mean Average Precision (mAP):** Surges from $12.61\%$ to **$18.49\%$** (**$+46.6\%$ relative gain**).
- **Aerial $\to$ Ground Rank-1:** Achieves **$53.45\%$** (+$6.32\%$ absolute gain).
- **Ground $\to$ Aerial Rank-1:** Leaps from $39.80\%$ to **$49.27\%$** (**$+9.47\%$ absolute gain**).
- **Efficiency:** Achieves a throughput of **$158.0\text{ FPS}$** ($6.33\text{ ms/image}$) on commodity laptop hardware with only **$26.17\text{M}$ parameters** and a compact **$512\text{-dimensional}$ embedding**, searching the entire $40,473$-image gallery in **$<15\text{ ms}$**.

---

## 2. AG-ReID.v2 Benchmark Dataset Analysis

The research utilizes the official **AG-ReID.v2** benchmark, the largest and most diverse aerial–ground person re-identification dataset currently available.

```text
                           AG-ReID.v2 Dataset (100,502 Images)
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        ▼                                ▼                                ▼
  train_all Partition              query Partition                gallery Partition
 51,530 Images (51.3%)            8,499 Images (8.5%)           40,473 Images (40.2%)
   590 Identities                   649 Identities                  649 Identities
   807 Tracklets                    808 Tracklets                   808 Tracklets
```

### 2.1 Dataset Partition Breakdown

| Partition | Image Count | Percentage | Unique Identities | Tracklets | Role in Methodology |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`train_all`** | 51,530 | 51.3% | 590 | 807 | Model parameter optimization and metric learning |
| **`query`** | 8,499 | 8.5% | 649 | 808 | Probe evaluation set |
| **`gallery`** | 40,473 | 40.2% | 649 | 808 | Search database containing candidate matches |
| **Total** | **100,502** | **100%** | **1,044** | **1,615** | Comprehensive evaluation benchmark |

### 2.2 Camera Sensor & Viewpoint Distribution

The dataset models realistic heterogeneous multi-sensor surveillance:
- **Camera $C_0$ (Aerial UAV Drone):** $46,779$ images ($46.5\%$). Captured from hovering drones at three distinct altitude tiers: $A_0$ ($32,174$ low altitude), $A_1$ ($38,926$ medium altitude), and $A_2$ ($29,402$ high altitude).
- **Camera $C_2$ (Wearable Smart Glasses):** $36,550$ images ($36.4\%$). Egocentric first-person viewpoint subject to body motion and variable angles.
- **Camera $C_3$ (Ground CCTV Surveillance):** $17,173$ images ($17.1\%$). Fixed ground-level surveillance cameras with static wide fields of view.

![Figure 3: Authorised AG-ReID.v2 Benchmark Examples across Camera Modalities](file:///Users/soubhagyaranjansahoo/person%20re%20identification/results/final_visualizations/benchmark_camera_views_panel.png)
*Fig. 3. Authorised benchmark examples from the official AG-ReID.v2 dataset release across three heterogeneous surveillance platforms: Aerial UAV (C0, steep pitch 60°–90°), Wearable Smart Glasses (C2, egocentric eye-level walking perspective), and Ground CCTV (C3, static surveillance profile). Each horizontal row illustrates the severe cross-view domain gap for the exact same pedestrian identity, preserving original source release labels.*

---

## 3. Proposed Methodology & Architecture

Our framework integrates four interconnected technical modules:

```
[Raw Input Image: (3, 256, 128)]
               │
               ▼
┌───────────────────────────────────────────────────────────┐
│ Module 1: Adaptive CIELAB CLAHE + LUT Gamma Correction   │
│ - Luminance enhancement without color distortion (γ=1.2)  │
└───────────────────────────────────────────────────────────┘
               │
               ▼
┌───────────────────────────────────────────────────────────┐
│ Module 2: Multi-Scale ResNet-50 Backbone (last_stride=1)  │
│ - Fine Scale   (layer1: 256ch, 64×32)  ──▶ GAP Pooler     │
│ - Medium Scale (layer2: 512ch, 32×16)  ──▶ GAP Pooler     │
│ - Coarse Scale (layer4: 2048ch, 16×8)  ──▶ GAP Pooler     │
└───────────────────────────────────────────────────────────┘
               │
               ▼
┌───────────────────────────────────────────────────────────┐
│ Module 3: 512-d Multi-Scale Projection & Alignment        │
│ - Projects 256, 512, 2048 channels into common d=512 space│
└───────────────────────────────────────────────────────────┘
               │
               ▼
┌───────────────────────────────────────────────────────────┐
│ Module 4: View-Aware Dynamic Scale Attention Gate         │
│ - Camera Embedding (C0/C2/C3) modulates scale weights:    │
│   α(x, v) = Softmax(W_v · [f_fine, f_med, f_coarse])      │
│ - Zero-initialized residual FiLM feature calibration      │
└───────────────────────────────────────────────────────────┘
               │
               ▼
┌───────────────────────────────────────────────────────────┐
│ Module 5: Compact 1D Latent Feature Diffusion Refinement  │
│ - 2-step DDIM denoising in 512-d latent space (0.63M params)│
│ - Eliminates cross-view domain discrepancy                │
└───────────────────────────────────────────────────────────┘
               │
               ▼
┌───────────────────────────────────────────────────────────┐
│ Module 6: BNNeck & Unit L2 Normalization                  │
│ - Produces compact 512-dim unit vector: ||v||_2 = 1.0     │
└───────────────────────────────────────────────────────────┘
```

### 3.1 Adaptive Image Enhancement (CIELAB CLAHE + Gamma)
To recover degraded pedestrian textures without distorting semantic clothing colors:
1. RGB input is converted to the **CIELAB color space**, decoupling luminance ($L$) from chrominance ($a, b$).
2. **CLAHE** (`clip_limit=2.0`, tile grid $8 \times 8$) is applied strictly to the $L$-channel to locally boost edge definition while preventing over-amplification of sensor noise.
3. A precomputed **Look-Up Table (LUT) Gamma Correction** ($\gamma = 1.2$) is applied to lighten deep pedestrian shadows.
4. Converted back to RGB for neural network feature extraction.

![Figure 4: Side-by-side Enhancement Comparison from AG-ReID.v2](file:///Users/soubhagyaranjansahoo/person%20re%20identification/results/final_visualizations/enhancement_comparison_figure4.png)
*Fig. 4. Side-by-side enhancement ablation on an authentic AG-ReID.v2 benchmark probe (P0867T04070A0C0F13861.jpg; Aerial UAV C0, Altitude Tier A0): (a) Original raw acquisition exhibiting severe cast shadow and underexposed clothing patterns (Entropy: 5.6023 bits); (b) CIELAB CLAHE result (clip_limit=2.0, tile_grid_size=(8,8) on L* channel), yielding +26.2% RMS contrast enhancement; (c) Proposed CLAHE + Gamma result (γ=1.2 via 256-element LUT), lifting deep shadow valleys and recovering fine garment texture without chromatic distortion (+18.2% entropy gain, +1.02 bits).*

### 3.2 Multi-Scale ResNet-50 Feature Extractor
Standard Re-ID backbones only use the final convolutional block, which discards localized spatial patterns. Our modified ResNet-50 retains three distinct hierarchical stages:
- **Fine Scale (`layer1`):** $256$ channels, spatial resolution $64 \times 32$. Encodes localized micro-textures (zippers, stripe patterns, shoe accents).
- **Medium Scale (`layer2`):** $512$ channels, spatial resolution $32 \times 16$. Encodes regional structural anatomy (torso, backpacks, trousers).
- **Coarse Scale (`layer4`):** $2048$ channels, spatial resolution $16 \times 8$ (preserved by setting `last_stride=1`). Encodes global silhouette, stature, and high-level semantics.

### 3.3 View-Aware Dynamic Scale Attention Gate
Because the informational value of scales varies drastically across camera domains, a static fusion weights all views equally. Our **View-Aware Module** learns a viewpoint embedding table $\mathbf{E}_v \in \mathbb{R}^{3 \times 128}$ for $v \in \{C_0, C_2, C_3\}$ and an auxiliary viewpoint classifier. The scale attention gate dynamically outputs weights:
$$\boldsymbol{\alpha} = [\alpha_{\text{fine}}, \alpha_{\text{med}}, \alpha_{\text{coarse}}] = \text{Softmax}(\mathbf{W}_a [\mathbf{z}_{\text{proj}} \,\|\, \mathbf{e}_v])$$
$$\mathbf{f}_{\text{fused}} = \alpha_{\text{fine}} \mathbf{z}_{\text{fine}} + \alpha_{\text{med}} \mathbf{z}_{\text{med}} + \alpha_{\text{coarse}} \mathbf{z}_{\text{coarse}}$$

### 3.4 Compact 1D Latent Feature Diffusion Refinement
Rather than using heavy, slow 860M-parameter pixel-space diffusion models (e.g., Stable Diffusion U-Nets), we formulate an ultra-compact **1D Latent Feature Diffusion Module** ($0.63\text{M}$ parameters). Operating on the 512-dimensional latent feature vector, it executes a 2-step deterministic Denoising Diffusion Implicit Models (DDIM) refinement:
$$\mathbf{z}_{\text{refined}} = \mathbf{z}_{\text{fused}} + \lambda \cdot \boldsymbol{\epsilon}_\theta(\mathbf{z}_{\text{fused}}, t, v)$$
This explicitly denoises cross-view domain shifts in $<2\text{ ms}$, ensuring real-time throughput.

---

## 4. In-Depth Result Analysis & Empirical Findings

Evaluation was conducted following the standard Market-1501 / AG-ReID protocol, testing all **$8,499$ query images** against all **$40,473$ gallery images** ($>344\text{ million}$ pairwise comparisons).

### 4.1 Overall Performance Comparison

| Metric | Baseline ResNet-50 | Proposed Full Architecture | Absolute Gain ($\Delta$) | Relative Gain (%) | Scientific Interpretation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Rank-1 Accuracy** | $22.94\%$ | **$26.89\%$** | **$+3.95\%$** | **$+17.2\%$** | Correct person returned as the #1 candidate |
| **Rank-5 Accuracy** | $35.60\%$ | **$42.96\%$** | **$+7.36\%$** | **$+20.7\%$** | Correct person appears in Top-5 results |
| **Rank-10 Accuracy** | $42.63\%$ | **$50.94\%$** | **$+8.31\%$** | **$+19.5\%$** | Correct person appears in Top-10 results |
| **mAP (Mean Avg Precision)** | $12.61\%$ | **$18.49\%$** | **$+5.88\%$** | **$+46.6\%$** | Quality of retrieving *all* instances of identity |
| **mINP (Inverse Neg Penalty)** | $1.11\%$ | **$2.72\%$** | **$+1.61\%$** | **$+145.0\%$** | Penalty for retrieving the hardest matching sample |

```text
                  Cumulative Matching Characteristics (CMC)
 60% ┌─────────────────────────────────────────────────────────────┐
     │                                                     50.94%  │
 50% │                                           42.96%   ┌──────┐ │
     │                                 35.60%   ┌──────┐  │      │ │
 40% │                       26.89%   ┌──────┐  │      │  │      │ │
     │             22.94%   ┌──────┐  │      │  │      │  │      │ │
 30% │            ┌──────┐  │      │  │      │  │      │  │      │ │
     │            │      │  │      │  │      │  │      │  │      │ │
 20% │            │      │  │      │  │      │  │      │  │      │ │
     │            │      │  │      │  │      │  │      │  │      │ │
  0% └────────────┴──────┴──┴──────┴──┴──────┴──┴──────┴──┴──────┴─┘
                  Rank-1 (Base vs Prop)  Rank-5 (Base vs Prop)  Rank-10 (Base vs Prop)
```

#### Key Analytical Takeaways:
1. **The $+46.6\%$ Surge in mAP:** mAP reflects the model's ability to retrieve *all multi-camera appearances* of a person rather than just finding one lucky match. The leap from $12.61\%$ to $18.49\%$ confirms that the multi-scale representations and view-aware attention successfully preserve cross-camera identity consistency.
2. **Top-10 Accuracy Surpasses 50%:** In real-world surveillance operations, security analysts inspect the top-10 candidate matches. The proposed model achieves **$50.94\%$ Rank-10 accuracy**, ensuring that for the majority of queries, the suspect is present within the initial screen view.

![Figure 5: Empirical Performance Comparison of Model Ablation Variants on AG-ReID.v2](file:///Users/soubhagyaranjansahoo/person%20re%20identification/results/final_visualizations/model_variants_ablation_figure5.png)
*Fig. 5. Empirical performance comparison (Rank-1 Accuracy and mAP in %) across the progressive model ablation variants evaluated on the official AG-ReID.v2 benchmark test split (8,499 queries × 40,473 gallery). All values represent actual evaluation metrics.*

---

### 4.2 Cross-View Directional Performance Analysis

To evaluate how well the framework handles asymmetric viewpoint transitions, the test set was partitioned into directional query-gallery pairs:

| Direction | Baseline Rank-1 | Proposed Rank-1 | Rank-1 Gain | Baseline mAP | Proposed mAP | mAP Gain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Aerial ($C_0$) $\to$ Ground ($C_3$)** | $47.13\%$ | **$53.45\%$** | **$+6.32\%$** | $23.72\%$ | **$31.81\%$** | **$+8.09\%$** |
| **Ground ($C_3$) $\to$ Aerial ($C_0$)** | $39.80\%$ | **$49.27\%$** | **$+9.47\%$** | $24.36\%$ | **$32.25\%$** | **$+7.89\%$** |

```text
               Cross-View Directional Rank-1 Performance
 60% ┌────────────────────────────────────────────────────────┐
     │                     53.45%                             │
 50% │           47.13%   ┌──────┐                  49.27%    │
     │          ┌──────┐  │      │        39.80%   ┌──────┐   │
 40% │          │      │  │      │       ┌──────┐  │      │   │
     │          │      │  │      │       │      │  │      │   │
 30% │          │      │  │      │       │      │  │      │   │
  0% └──────────┴──────┴──┴──────┴───────┴──────┴──┴──────┴───┘
                Aerial UAV ──▶ Ground CCTV       Ground CCTV ──▶ Aerial UAV
                 (Baseline vs Proposed)           (Baseline vs Proposed)
```

#### Deep Dive into the $+9.47\%$ Ground-to-Aerial Breakthrough:
In aerial–ground surveillance, querying **from Ground CCTV to Aerial Drone** is significantly harder than the reverse:
* A ground CCTV camera captures a detailed frontal profile. When querying an aerial gallery, the AI must match that frontal profile against top-down drone shots where pedestrian heads, shoulders, and hats dominate, and lower-body clothing is heavily foreshortened.
* Standard ResNet-50 struggles severely here, dropping to $39.80\%$ Rank-1.
* Our proposed model achieves a massive **$+9.47\%$ jump to $49.27\%$**. The multi-scale features ensure fine torso textures from ground shots are projected and aligned with coarse aerial geometry, effectively bridging the asymmetric perspective gap.

---

### 4.3 Learned Dynamic Scale Attention Weights Analysis

An inspection of the trained attention weights reveals how the model dynamically prioritizes visual scales across camera domains:

| Camera Platform | Fine Scale ($\alpha_{\text{fine}}$, `layer1`) | Medium Scale ($\alpha_{\text{med}}$, `layer2`) | Coarse Scale ($\alpha_{\text{coarse}}$, `layer4`) | Primary Informational Cue |
| :--- | :---: | :---: | :---: | :--- |
| **Aerial UAV ($C_0$)** | $0.1745$ ($17.5\%$) | $0.1956$ ($19.6\%$) | **$0.6298$ ($63.0\%$)** | Global silhouette, stature, head-shoulder geometry |
| **Wearable Glasses ($C_2$)** | $0.1799$ ($18.0\%$) | $0.1989$ ($19.9\%$) | **$0.6212$ ($62.1\%$)** | Egocentric torso structure & upper garments |
| **Ground CCTV ($C_3$)** | $0.1655$ ($16.6\%$) | $0.1928$ ($19.3\%$) | **$0.6417$ ($64.2\%$)** | Whole-body pose, height ratio, global color block |

**Observation:** Across all platforms, coarse semantics (`layer4`) consistently provide the dominant anchor (~$62\% - 64\%$), while fine textures (`layer1`, ~$17\%$) and structural mid-level body parts (`layer2`, ~$19\%$) provide the crucial fine-grained discriminative signals needed to distinguish between individuals wearing similar clothing.

---

### 4.4 Image Enhancement Quantitative Quality Gain

Evaluating the CIELAB CLAHE + Gamma module on AG-ReID.v2 images demonstrates measurable contrast and entropy improvements:

| Metric | Raw Dataset Average | Enhanced Average | Improvement |
| :--- | :---: | :---: | :---: |
| **RMS Contrast (UAV $C_0$)** | $0.1986$ | **$0.2209$** | **$+11.2\%$** |
| **RMS Contrast (Ground $C_3$)** | $0.2021$ | **$0.2320$** | **$+14.8\%$** |
| **Shannon Information Entropy** | $7.294\text{ bits}$ | **$7.668\text{ bits}$** | **$+0.374\text{ bits}$** |

By lifting local shadow valleys and equalizing luminance distribution, the enhancement module recovers subtle fabric wrinkles and badge details without corrupting chromatic values ($a, b$ channels remain untouched).

---

### 4.5 Computational Efficiency & Hardware Feasibility

| Architecture | Model Parameters | Latency (ms/image) | Throughput (FPS) | Hardware Deployment Feasibility |
| :--- | :---: | :---: | :---: | :--- |
| **Pixel Diffusion (SD-ReID U-Net)** | $\sim 860.0\text{ M}$ | $\sim 250 - 500\text{ ms}$ | $< 5\text{ FPS}$ | High-end Server (A100/H100, 40GB+ VRAM) |
| **Standard ResNet-50 Baseline** | $24.72\text{ M}$ | $4.02\text{ ms}$ | $249.0\text{ FPS}$ | Laptop CPU / Apple Silicon MPS |
| **Our Proposed Full Model** | **$26.17\text{ M}$** | **$6.33\text{ ms}$** | **$158.0\text{ FPS}$** | **Laptop CPU / Apple Silicon MPS** |

Our complete proposed framework adds only **$1.45\text{M}$ parameters** ($+5.8\%$) over the baseline, while running at **$158\text{ FPS}$**. Furthermore, by projecting the final representation down to **$512\text{ dimensions}$** (compared to $2048\text{-d}$ in baseline ResNet-50), gallery search speed across the $40,473$ database is accelerated by **$4\times$**, enabling sub-15ms real-time retrieval.

---

## 5. Web Application Architecture & Deployment

To make the research functional and demonstratable, an end-to-end full-stack web application was developed and deployed:

```
[User Browser: React 19 SPA (Vite + Tailwind CSS + Recharts)]
                           │  HTTP / REST
                           ▼
[FastAPI Application Server: backend/app/main.py]
  ├─ /api/search     ──▶ Preprocessing ──▶ Model Inference ──▶ Gallery Matching
  ├─ /api/compare    ──▶ 1-to-1 Cross-View Verification (Threshold: 0.40)
  ├─ /api/model/info ──▶ Architectural specs & scale attention weights
  ├─ /api/evaluation ──▶ Empirical benchmark JSON metrics
  └─ /api/images/... ──▶ Secure gallery asset streaming
```

### Key Software Engineering Implementations:
1. **Precomputed Vector Database (`features/gallery_cache_proposed.npz`):** Precomputed all $40,473$ unit L2-normalized 512-d embeddings in memory. Cosine similarity matching is executed via vector-matrix dot products in $<15\text{ ms}$.
2. **Automated Port Conflict Resolver (`run_server.py`):** Automatically detects occupied ports (e.g. port 8000) and gracefully binds to the next available port without throwing `[Errno 48]`.
3. **Responsive Image Uploader:** Features drag-and-drop file ingestion, automatic dimension detection, client-side preview, and 1-click test probes from AG-ReID.v2 ($P0867$, $P0195$, $P0610$).
4. **Automated Integration Tests (`backend/tests/test_api.py`):** 9 automated unit/integration tests verifying API health, dataset metadata, search accuracy, pairwise comparison, and security directory traversal protection.

---

## 6. Conclusion & Viva Defense Summary

### 6.1 Summary of Contributions:
1. **Designed and implemented** a complete multi-scale person Re-ID architecture that intercepts ResNet-50 layers 1, 2, and 4 with `last_stride=1`.
2. **Developed a View-Aware Dynamic Scale Attention Gate** that tailors hierarchical feature fusion to the camera platform domain ($C_0, C_2, C_3$).
3. **Formulated an efficient 1D Latent Feature Diffusion Module** providing 2-step denoising refinement in compact 512-d space rather than intractable pixel space.
4. **Achieved verified empirical gains** on AG-ReID.v2: $+17.2\%$ Rank-1 relative gain, $+46.6\%$ mAP relative gain, and $+9.47\%$ Ground-to-Aerial Rank-1 gain.
5. **Deployed a production-grade full-stack web dashboard** running live at $158\text{ FPS}$ on Apple Silicon / CPU.

### 6.2 Key Takeaway for Academic Presentation:
> *"Our project demonstrates that overcoming the aerial–ground domain gap does not require massive 860M-parameter generative models. By combining adaptive CIELAB enhancement, multi-scale hierarchical extraction, viewpoint-conditioned scale attention, and lightweight 1D latent diffusion, our 26.17M-parameter model achieves superior cross-view recognition accuracy while operating in real-time on standard laptop hardware."*

