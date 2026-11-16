# Step 4 Research Report: Image Enhancement Using CLAHE and Gamma Correction

## 1. Research Objective
This phase of the research project, **"View-Aware Multi-Scale Image Enhancement and Feature Fusion for Aerial–Ground Person Re-Identification"**, develops and validates an adaptive image enhancement pipeline.

The goal is to mitigate environmental and sensor discrepancies between high-altitude Unmanned Aerial Vehicle (UAV) cameras and ground-based surveillance cameras (CCTV and Wearable smart glasses) without corrupting identity-critical color and texture cues.

---

## 2. Why Image Enhancement Is Essential for Aerial–Ground Person Re-ID

Aerial–ground person matching suffers from several domain-specific visual degradation factors:
1. **Atmospheric Attenuation & Haze**: UAV images captured from medium and high altitudes (`A1`, `A2`) suffer from reduced dynamic range and faded contrast due to atmospheric scattering.
2. **Extreme Lighting & Deep Shadows**: Pedestrians on ground level cast steep shadows or walk in occlusion zones near tall buildings, hiding fine clothing textures.
3. **Severe Perspective & Scale Distortion**: Aerial views compress the human body vertically and feature lower spatial resolution per pedestrian bounding box compared to horizontal CCTV views.

By applying localized contrast enhancement and non-linear luminance correction, we expand the discriminative dynamic range of pedestrian features before feature extraction.

---

## 3. Mathematical Foundations & Implementation

### A. Contrast Limited Adaptive Histogram Equalization (CLAHE)
Standard Global Histogram Equalization flattens the global intensity histogram, which over-amplifies noise and produces unnatural color artifacts.

CLAHE solves this via two mechanisms:
1. **Local Context Grids**: Divides the image into contextual tiles (default: $8 \times 8$ grid) and equalizes histograms locally.
2. **Contrast Limiting (Clipping)**: Clips histogram bins exceeding a predefined threshold (`clipLimit = 2.0`), redistributing the clipped pixels uniformly across all bins to prevent localized noise over-amplification.
3. **Bilinear Interpolation**: Blends adjacent tile boundaries to eliminate artificial block borders.

#### Preservation of True Color in CIELAB Space
To prevent color distortion, CLAHE is never applied to RGB channels directly. Instead:
$$\text{RGB} \xrightarrow{\text{cvtColor}} \text{CIELAB} \quad (L, a, b)$$
* **$L$ Channel (Lightness)**: Processed by CLAHE ($L \rightarrow L_{\text{enhanced}}$).
* **$a, b$ Channels (Chroma / Color Opponents)**: Kept 100% unaltered.
$$\text{Merge}(L_{\text{enhanced}}, a, b) \xrightarrow{\text{cvtColor}} \text{RGB}_{\text{enhanced}}$$
This strictly preserves clothing colors (e.g. distinguishing a red shirt from a maroon shirt).

### B. Gamma Correction (Power-Law Transform)
Gamma Correction applies a non-linear luminance mapping:
$$V_{\text{out}} = 255 \times \left(\frac{V_{\text{in}}}{255}\right)^{\frac{1}{\gamma}}$$
* For $\gamma > 1.0$ (e.g. $\gamma = 1.2$): Expands dark shadows and brings out under-exposed details while keeping bright highlights from blowing out.
* High-speed implementation: Evaluated via a precomputed 256-entry lookup table (LUT) using `cv2.LUT`.

### C. Combined Enhancement Pipeline
The unified `ImageEnhancer` supports four operational modes:
1. `none`: Identity pass-through.
2. `clahe`: Local contrast enhancement only.
3. `gamma`: Non-linear luminance brightening only.
4. `clahe_gamma`: Combined sequential enhancement (`clahe_first` or `gamma_first`).

---

## 4. Quantitative Image Quality Evaluation

We measured four statistical and informational metrics across representative sample images from UAV (`C0`), Wearable (`C2`), and Ground CCTV (`C3`) cameras:
- **Mean Brightness (Luminance)**: Average intensity level ($[0, 255]$).
- **RMS Contrast**: Root-mean-square contrast of pixel intensities ($\frac{\sigma}{255}$).
- **Shannon Information Entropy**: Measures information content and texture richness in bits ($H = -\sum p_i \log_2 p_i$).

### Measured Metrics Summary ([`results/enhancement_metrics.csv`](file:///Users/soubhagyaranjansahoo/person%20re%20identification%20/results/enhancement_metrics.csv))

| Sample File | Camera Platform | Enhancement Method | Mean Brightness | RMS Contrast | Shannon Entropy (bits) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `P2207T03241A0C0F26761.jpg` | **UAV Aerial (`C0`)** | **Original** | 86.72 | 0.1986 | 7.2942 |
| `P2207T03241A0C0F26761.jpg` | UAV Aerial (`C0`) | CLAHE | 113.67 | **0.2209** (+11.2%) | **7.6684** (+0.37 bits) |
| `P2207T03241A0C0F26761.jpg` | UAV Aerial (`C0`) | Gamma ($\gamma=1.2$) | 100.99 | 0.1937 | 7.3228 |
| `P2207T03241A0C0F26761.jpg` | UAV Aerial (`C0`) | CLAHE + Gamma | **127.33** | 0.2094 | 7.6278 |
| `P2207T03241A0C2F24871.jpg` | **Wearable (`C2`)** | **Original** | 79.87 | 0.2226 | 7.4120 |
| `P2207T03241A0C2F24871.jpg` | Wearable (`C2`) | CLAHE | 98.49 | **0.2314** | **7.7125** (+0.30 bits) |
| `P2207T03241A0C2F24871.jpg` | Wearable (`C2`) | CLAHE + Gamma | **111.96** | 0.2249 | 7.7267 |
| `P2207T03241A0C3F12181.jpg` | **Ground CCTV (`C3`)** | **Original** | 146.63 | 0.2021 | 7.4329 |
| `P2207T03241A0C3F12181.jpg` | Ground CCTV (`C3`) | CLAHE | 167.19 | **0.2320** (+14.8%) | **7.5903** (+0.16 bits) |
| `P2207T03241A0C3F12181.jpg` | Ground CCTV (`C3`) | CLAHE + Gamma | **177.01** | 0.2116 | 7.4503 |

### Observations:
1. **Entropy Gain**: CLAHE consistently increases Shannon Information Entropy by **$+0.16$ to $+0.37$ bits per image**, proving that suppressed low-contrast features are brought into distinguishable dynamic ranges.
2. **Shadow Recovery**: Gamma correction with $\gamma = 1.2$ lifts underexposed UAV pedestrian crops from an average brightness of $86.7$ to $101.0$ without clipping bright sunlight highlights.
3. **Contrast Enhancement**: RMS contrast increases from $0.1986$ to $0.2209$ on UAV imagery.

> [!NOTE]
> Higher contrast and entropy indicate richer visual dynamic range, but do not automatically guarantee higher Re-ID accuracy. True person re-identification efficacy is determined by evaluating cross-view matching against the baseline.

---

## 5. Visual Comparisons

Visual comparison grid generated at:
[`results/enhancement_visualizations/enhancement_comparison_grid.png`](file:///Users/soubhagyaranjansahoo/person%20re%20identification%20/results/enhancement_visualizations/enhancement_comparison_grid.png)

Shows side-by-side:
1. **Column 1**: Original raw image.
2. **Column 2**: CLAHE enhancement (`clip_limit=2.0`).
3. **Column 3**: Gamma correction ($\gamma=1.2$).
4. **Column 4**: Combined CLAHE + Gamma enhancement.

---

## 6. Connection with the ResNet-50 Baseline Model

To ensure a scientifically valid, reproducible comparison:
* **Controlled Experiment Protocol**:
  * **Experiment A**: Raw AG-ReID.v2 $\rightarrow$ ResNet-50 Baseline.
  * **Experiment B**: Enhanced AG-ReID.v2 (CLAHE + Gamma) $\rightarrow$ Identical ResNet-50 Architecture.
* **Preservation of Benchmark Integrity**:
  * All original dataset files in `datasets/AG-ReID.v2/` remain 100% untouched.
  * Enhanced data is saved separately in `datasets/AG-ReID.v2_enhanced/`.
  * Training and testing splits, filenames, and identity labels are preserved identically.

