# Step 5 Research Report: Multi-Scale Feature Extraction

## 1. Research Objective
This phase of the research project, **"View-Aware Multi-Scale Image Enhancement and Feature Fusion for Aerial–Ground Person Re-Identification"**, implements and verifies a CNN-based multi-scale feature extraction pipeline.

The goal is to extract visual representations across multiple receptive field granularities—fine scale, medium scale, and coarse scale—from a deep ResNet-50 backbone, without prematurely fusing them.

---

## 2. Why Multi-Scale Feature Extraction Is Critical in Aerial–Ground Re-ID

Aerial–ground person matching presents extreme scale and viewpoint discrepancies:
1. **Extreme Scale Variance**: A pedestrian viewed from an aerial drone at high altitude (`A2`) occupies as few as $20 \times 38$ pixels, whereas ground-level CCTV (`C3`) captures them at up to $430 \times 1080$ pixels.
2. **Receptive Field Granularity**:
   * Shallow CNN layers with small receptive fields capture high-frequency details (shoe colors, stripe patterns, small bags, backpack straps).
   * Deeper CNN layers with large receptive fields capture holistic body silhouettes, head-to-shoulder structures, and posture.
3. **Information Loss in Standard Backbones**: A standard ResNet-50 only retains the final layer (layer4, 2048-dim) after Global Average Pooling. All fine-grained spatial textures from earlier layers are irreversibly discarded.

By extracting representations from multiple stages, the model retains complementary multi-resolution visual evidence.

---

## 3. ResNet-50 Intermediate Stages & Mathematical Dimensions

Given an input image of standard Person Re-ID dimension:
$$\mathbf{X} \in \mathbb{R}^{B \times 3 \times 256 \times 128}$$

The feature representations extracted by [`src/models/resnet50_multiscale.py`](file:///Users/soubhagyaranjansahoo/person%20re%20identification%20/src/models/resnet50_multiscale.py) are:

| Scale Level | ResNet-50 Stage | 2D Feature Map Shape | Spatial Stride | Semantic Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Fine Scale** | `layer1` | $\mathbf{F}_{\text{fine}} \in \mathbb{R}^{B \times 256 \times 64 \times 32}$ | $4\times$ | High-frequency edges, fabric patterns, localized color cues |
| **Medium Scale** | `layer2` | $\mathbf{F}_{\text{med}} \in \mathbb{R}^{B \times 512 \times 32 \times 16}$ | $8\times$ | Regional body anatomy (torso, upper/lower clothing, accessories) |
| **Coarse Scale** | `layer4` | $\mathbf{F}_{\text{coarse}} \in \mathbb{R}^{B \times 2048 \times 16 \times 8}$ | $16\times$ (stride-1) | Global human posture, holistic viewpoint, high-level semantics |

*Note: In Person Re-ID, the stride of the first conv block in `layer4` is set to 1 (instead of the standard 2). This preserves a $16 \times 8$ spatial resolution for coarse features rather than collapsing to $8 \times 4$.*

---

## 4. Feature Pooling via Global Average Pooling (GAP)

To transform 2D spatial feature maps into fixed-length embedding vectors suitable for retrieval and downstream fusion, the [`MultiScaleFeaturePooler`](file:///Users/soubhagyaranjansahoo/person%20re%20identification%20/src/features/feature_pooling.py) applies Global Average Pooling independently to each scale:

$$\mathbf{v}_{\text{scale}} = \frac{1}{H_s W_s} \sum_{h=1}^{H_s} \sum_{w=1}^{W_s} \mathbf{F}_{\text{scale}}[:, :, h, w]$$

### Pooled Vector Dimensions:
- **Fine Vector ($\mathbf{v}_{\text{fine}}$)**: $\mathbb{R}^{B \times 256}$
- **Medium Vector ($\mathbf{v}_{\text{med}}$)**: $\mathbb{R}^{B \times 512}$
- **Coarse Vector ($\mathbf{v}_{\text{coarse}}$)**: $\mathbb{R}^{B \times 2048}$

Each pooled vector is optionally $L_2$-normalized:
$$\mathbf{v}_{\text{norm}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}$$

> [!IMPORTANT]
> **No Feature Fusion in Step 5**: In strict adherence to Step 5 requirements, the three vectors are stored as separate tensors and are **not** concatenated or projected together. Feature fusion is the core focus of Step 6.

---

## 5. Visualizing Multi-Scale Activation Heatmaps

The module [`src/features/feature_visualization.py`](file:///Users/soubhagyaranjansahoo/person%20re%20identification%20/src/features/feature_visualization.py) generates spatial activation heatmaps by computing the mean absolute response across channels:
$$A(h, w) = \frac{1}{C} \sum_{c=1}^C |F(c, h, w)|$$

Saved figures in [`results/multiscale_visualizations/`](file:///Users/soubhagyaranjansahoo/person%20re%20identification%20/results/multiscale_visualizations) demonstrate:
* **Fine-scale heatmaps**: Show sharp, localized activations along pedestrian contours, shoes, and clothing boundaries.
* **Medium-scale heatmaps**: Group into distinct upper-body and lower-body regional clusters.
* **Coarse-scale heatmaps**: Cover the entire human silhouette with smooth, holistic semantic activations.

---

## 6. Validation and Numerical Unit Testing

The unit test [`tests/test_multiscale.py`](file:///Users/soubhagyaranjansahoo/person%20re%20identification%20/tests/test_multiscale.py) verified:
1. Correct tensor shapes for all three stages on Apple Silicon (`mps`).
2. Numerical validity: zero `NaN` or `Inf` values across all 2D maps and pooled vectors.
3. Unit $L_2$ norm consistency ($\|\mathbf{v}\|_2 = 1.0000 \pm 10^{-4}$).

---

## 7. Next Step: Step 6 (Feature Fusion & Cross-View Alignment)

With three decoupled scale representations ($\mathbb{R}^{256}$, $\mathbb{R}^{512}$, $\mathbb{R}^{2048}$), the project is ready for:
1. Dimension projection into a shared latent space (e.g. projecting 256, 512, and 2048 into a common $d=512$ space).
2. Adaptive or attention-based multi-scale feature fusion.
3. Cross-view alignment between aerial UAV (`C0`) and ground CCTV/Wearable (`C3`/`C2`) viewpoints.

