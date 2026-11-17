# 1-to-1 Cross-View Person Verification System

**Project:** *View-Aware Multi-Scale Image Enhancement and Feature Fusion for Aerial–Ground Person Re-Identification*  
**Authors:** Soubhagya Ranjan Sahoo (`23110238`) & Aryan Pradhan (`23110197`)  
**Guide:** Dr. Rajalaxmi Padhy | **Benchmark:** AG-ReID.v2 | **Platform:** PyTorch + FastAPI + React 19

---

## 1. Overview & Purpose

In multi-camera surveillance networks involving uncrewed aerial vehicles (UAVs) and ground cameras, surveillance operators face two distinct operational tasks:
1. **1-to-N Gallery Retrieval:** Querying one image against tens of thousands of database images to locate candidate matches.
2. **1-to-1 Cross-View Verification:** Given two specific image captures from completely different camera platforms (e.g., an overhead drone shot and a ground CCTV frame), verify whether they depict the **exact same human identity** or **two different individuals**.

This module provides a dedicated, forensic-grade **1-to-1 Cross-View Verification Engine** that aligns severe perspective discrepancies and outputs a calibrated matching score and definitive verdict in under **$15\text{ milliseconds}$**.

---

## 2. The Core Scientific Challenge

Comparing two images across aerial and ground cameras is significantly harder than standard CCTV-to-CCTV matching due to three fundamental domain gaps:

```text
       Aerial UAV Drone (C0)                   Ground CCTV (C3)
     [Steep Pitch Angle: 60°-90°]            [Eye-Level / Static: 0°-15°]
                 │                                        │
                 ▼                                        ▼
    • Heads & shoulders dominate            • Full-body vertical stature
    • Stature severely foreshortened        • Footwear & accessories visible
    • Deep solar shadows & haze             • Sharp horizontal clothing textures
```

Standard deep neural networks fail because the raw pixel distributions of the same person across these two viewpoints share virtually zero geometric overlap. Our pipeline explicitly bridges this gap using multi-scale extraction, viewpoint attention, and latent diffusion.

---

## 3. End-to-End Verification Pipeline (How It Is Done)

When two images ($I_A$ and $I_B$) are submitted to the verification system, the following 8-step pipeline executes synchronously:

```
Image A (e.g. UAV C0)       Image B (e.g. CCTV C3)
        │                           │
        ▼                           ▼
[1. CIELAB Decoupling]     [1. CIELAB Decoupling]
        │                           │
        ▼                           ▼
[2. CLAHE + Gamma]         [2. CLAHE + Gamma]
        │                           │
        ▼                           ▼
[3. Multi-Scale ResNet]    [3. Multi-Scale ResNet]
  (layer1, layer2, layer4)   (layer1, layer2, layer4)
        │                           │
        ▼                           ▼
[4. 512-d GAP Projections] [4. 512-d GAP Projections]
        │                           │
        ▼                           ▼
[5. View Attention Gate]   [5. View Attention Gate]
        │                           │
        ▼                           ▼
[6. 2-Step 1D Diffusion]   [6. 2-Step 1D Diffusion]
        │                           │
        ▼                           ▼
[7. BNNeck + L2 Norm]      [7. BNNeck + L2 Norm]
        │                           │
        ▼                           ▼
  Embedding u ∈ S^511         Embedding v ∈ S^511
        └─────────────┬─────────────┘
                      ▼
        [8. Cosine Dot Product: u · v]
                      ▼
        [9. Calibrated Decision Gate]
                      │
      ┌───────────────┴───────────────┐
      ▼                               ▼
Similarity >= 0.40             Similarity < 0.40
[VERDICT: SAME PERSON]     [VERDICT: DIFFERENT PERSONS]
```

### Detailed Pipeline Steps:

#### Step 1: CIELAB Color Space Transformation
Both images are decoupled into the device-independent **CIELAB** color space. Lightness ($L^*$) is separated from chromaticity ($a^*, b^*$), ensuring enhancement operations do not alter genuine clothing colors.

#### Step 2: Adaptive CLAHE & LUT Gamma Preprocessing
* **CLAHE:** Applied strictly to the $L^*$ channel (`clip_limit=2.0`, tile grid $8 \times 8$) to amplify local edge definition.
* **LUT Gamma Correction:** Power-law transformation ($L_{\text{new}} = 100 \cdot (L/100)^{1/1.2}$) lifts deep shadow valleys caused by harsh overhead sunlight.

#### Step 3: Multi-Scale Hierarchical Feature Extraction
Both images pass through our modified **ResNet-50 backbone** (`last_stride=1`), simultaneously extracting three spatial representations:
* **Fine Scale (`layer1`):** $256 \times 64 \times 32$ — fabric textures, shoe accents, edge details.
* **Medium Scale (`layer2`):** $512 \times 32 \times 16$ — regional anatomy, backpacks, trousers.
* **Coarse Scale (`layer4`):** $2048 \times 16 \times 8$ — holistic stature, body build, and silhouette.

#### Step 4: Dimension Alignment & GAP Pooling
Each feature map is pooled via Global Average Pooling (GAP) and linearly projected into a common **$512\text{-dimensional}$** subspace ($\mathbf{z}_{\text{fine}}, \mathbf{z}_{\text{med}}, \mathbf{z}_{\text{coarse}} \in \mathbb{R}^{512}$).

#### Step 5: View-Aware Dynamic Scale Gating & FiLM Calibration
An auxiliary viewpoint classifier detects whether each image originated from a Drone ($C_0$), Smart Glasses ($C_2$), or CCTV ($C_3$):
* For the **Drone image ($C_0$)**: The attention gate automatically assigns higher weight to **Coarse Scale ($\approx 63\%$)** to focus on silhouette and torso color blocks.
* For the **Ground image ($C_3$)**: The attention gate elevates **Fine ($\approx 17\%$) and Medium ($\approx 19\%$)** scales to capture logos, footwear, and accessories.
* **FiLM Modulation:** Feature-wise Linear Modulation $(1 + \boldsymbol{\gamma}) \odot \mathbf{f}_{\text{fused}} + \boldsymbol{\beta}$ calibrates camera-specific distribution shifts.

#### Step 6: 2-Step 1D Latent Diffusion Denoising
A compact **1D Latent Diffusion Module** ($0.63\text{M}$ parameters) applies 2 deterministic DDIM reverse steps directly to the 512-dimensional vector. It treats cross-view perspective distortion as structured domain noise and purges it, aligning the two representations into a view-invariant identity manifold.

#### Step 7: BNNeck & $L_2$ Unit Hypersphere Normalization
Features pass through a Batch Normalization neck and are projected onto a **512-dimensional unit hypersphere** ($\mathbb{S}^{511}$):
$$\mathbf{\hat{u}} = \frac{\mathbf{u}}{\|\mathbf{u}\|_2}, \quad \mathbf{\hat{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2} \quad \left(\|\mathbf{\hat{u}}\|_2 = 1.0, \; \|\mathbf{\hat{v}}\|_2 = 1.0\right)$$

#### Step 8: Metric Computation & Pairwise Scoring
Because both vectors have unit length, the Cosine Similarity simplifies to a fast vector dot product:
$$\text{Similarity}(\mathbf{\hat{u}}, \mathbf{\hat{v}}) = \mathbf{\hat{u}} \cdot \mathbf{\hat{v}} = \sum_{i=1}^{512} \hat{u}_i \cdot \hat{v}_i \quad \in [-1.0, +1.0]$$

#### Step 9: Calibrated Decision Thresholding
The similarity score is evaluated against an empirically calibrated decision boundary $\tau = 0.40$ ($40.0\%$):
* **$\text{Similarity} \ge 0.40 \implies \mathbf{\text{SAME PERSON}}$** (Positive verification)
* **$\text{Similarity} < 0.40 \implies \mathbf{\text{DIFFERENT PERSONS}}$** (Impostor rejection)
* **Confidence Rating:**
  * **High Confidence:** $|\text{Similarity} - 0.40| > 0.15$
  * **Moderate Confidence:** $|\text{Similarity} - 0.40| \le 0.15$

---

## 4. Empirical Verification Performance on AG-ReID.v2

Tested across all $8,499$ queries and $40,473$ gallery images on the official AG-ReID.v2 benchmark:

| Cross-View Protocol | Baseline ResNet-50 | Proposed Framework | Improvement | Operational Impact |
| :--- | :---: | :---: | :---: | :--- |
| **Aerial UAV ($C_0$) $\to$ Ground CCTV ($C_3$)** | $47.13\%$ Rank-1 | **$53.45\%$ Rank-1** | **$+6.32\%$** | Accurately identifies aerial drone suspects in ground CCTV |
| **Ground CCTV ($C_3$) $\to$ Aerial UAV ($C_0$)** | $39.80\%$ Rank-1 | **$49.27\%$ Rank-1** | **$+9.47\%$** | Bridges extreme ground-to-air foreshortening gap |
| **Mean Average Precision (mAP)** | $24.04\%$ avg | **$32.03\%$ avg** | **$+7.99\%$** | Consistent identity separation across diverse camera angles |
| **Verification Speed (Pairwise)** | $8.04\text{ ms}$ | **$12.66\text{ ms}$** | Real-Time | Sub-15ms forensic response time |

---

## 5. API Reference & Code Usage

### REST API Endpoint

```http
POST /api/v1/search/compare
Content-Type: multipart/form-data
```

#### Request Parameters:
| Field | Type | Required | Description |
| :--- | :--- | :---: | :--- |
| `file1` | Binary Image File | **Yes** | First person image (JPEG/PNG) |
| `file2` | Binary Image File | **Yes** | Second person image (JPEG/PNG) |
| `cam1` | String | No | Viewpoint of Image 1 (`auto`, `uav`, `cctv`, `wearable`) |
| `cam2` | String | No | Viewpoint of Image 2 (`auto`, `uav`, `cctv`, `wearable`) |
| `threshold` | Float | No | Decision threshold (Default: `0.40`) |

#### JSON Response Schema:
```json
{
  "status": "success",
  "similarity": 0.7482,
  "similarity_pct": "74.8%",
  "threshold": 0.40,
  "is_same": true,
  "verdict": "SAME PERSON",
  "confidence": "High",
  "view1": "Aerial UAV (C0)",
  "view2": "Ground CCTV (C3)",
  "time_ms": 12.66
}
```

### Python Client Example (`requests`):

```python
import requests

url = "http://localhost:8000/api/v1/search/compare"

files = {
    "file1": open("datasets/AG-ReID.v2/AG-ReID.v2/query/P0867T04070A0/P0867T04070A0C0F13861.jpg", "rb"),
    "file2": open("datasets/AG-ReID.v2/AG-ReID.v2/query/P0867T04070A0/P0867T04070A0C3F07831.jpg", "rb"),
}
data = {
    "cam1": "auto",
    "cam2": "auto",
    "threshold": 0.40,
}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"Similarity: {result['similarity_pct']}")
print(f"Verdict:    {result['verdict']} ({result['confidence']} Confidence)")
print(f"Latency:    {result['time_ms']} ms")
```

---

## 6. Web Dashboard Integration

The 1-to-1 cross-view verification console is fully integrated into the production web UI:
1. Navigate to **`http://localhost:8000`** in your browser.
2. Select the **"Cross-View 1-to-1 Verification"** tab on the main dashboard.
3. Upload any two images or click the pre-configured 1-click test probe pairs:
   * **Same-Person Cross-View Probe (`P0610`):** UAV image $C_0$ vs. CCTV image $C_3$ $\implies$ Similarity: $\approx 72\%$, Verdict: **SAME PERSON** (Green).
   * **Different-Persons Impostor Probe:** Person `P0610` vs. Person `P0867` $\implies$ Similarity: $\approx 18\%$, Verdict: **DIFFERENT PERSONS** (Red).
4. Inspect the dynamic scale attention breakdown and enhanced image previews side-by-side.
