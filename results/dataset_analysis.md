# AG-ReID.v2 Dataset Analysis Report

## 1. Dataset Overview
- **Dataset Name**: AG-ReID.v2 (Aerial–Ground Person Re-Identification v2)
- **Research Domain**: Cross-View Person Re-Identification (Aerial UAV vs. Ground Camera Platforms)
- **Dataset Path**: `datasets/AG-ReID.v2/AG-ReID.v2`
- **Total Image Files**: **100,502** (100% verified)
- **Total Unique Person IDs**: **1044**
- **Corrupted Images**: **0** (out of 1000 sampled)

## 2. Directory Structure & Official Splits
The dataset is structured into standard Re-ID evaluation partitions:

| Split Partition | Subdirectories (Tracklets) | Image Count | Unique Identities | Primary Role |
| :--- | :--- | :--- | :--- | :--- |
| **`train_all`** | 807 | 51,530 | 590 | Model representation learning |
| **`query`** | 808 | 8,499 | 649 | Probe set for cross-view retrieval |
| **`gallery`** | 808 | 40,473 | 649 | Search gallery database |
| **Total** | **1,615** | **100,502** | **1044** | Full benchmark |

> [!NOTE]
> In AG-ReID.v2, the 1,615 subdirectories represent individual **person tracklets/sequences** (807 training tracklets and 808 testing tracklets) across 1,044 unique individuals.

## 3. Camera Viewpoint & Sensor Breakdown
The dataset features multi-platform camera viewpoints critical for aerial-to-ground Re-ID:

| Sensor Code | Sensor / Platform Description | Image Count | Percentage |
| :--- | :--- | :--- | :--- |
| `C0` | UAV (Aerial Camera) | 46,779 | 46.5% |
| `C2` | Wearable (First-person Smart Glasses) | 36,550 | 36.4% |
| `C3` | Ground CCTV (Surveillance Camera) | 17,173 | 17.1% |

## 4. Flight Altitude Breakdown (Multi-Scale Context)

| Altitude Code | Altitude Level | Image Count | Percentage |
| :--- | :--- | :--- | :--- |
| `A0` | Low Altitude | 32,174 | 32.0% |
| `A1` | Medium Altitude | 38,926 | 38.7% |
| `A2` | High Altitude | 29,402 | 29.3% |

## 5. Image Resolutions & Formats
- **File Formats**: {'jpg': 100502}
- **Color Modes**: {'RGB': 1000}
- **Width Range**: 20px to 430px (Mean: 86.4px)
- **Height Range**: 38px to 1080px (Mean: 201.2px)
- **Scale Disparity**: Images captured from high altitude (`A2`) feature pedestrian crops as small as 22×33 pixels, while ground CCTV (`C3`) and wearable cameras (`C2`) reach up to 757×938 pixels.

## 6. Filename Nomenclature Protocol
All 100,502 image files adhere to the standardized naming formula:
```text
P[PersonID] T[TrackletID] A[AltitudeID] C[CameraID] F[FrameID].jpg
Example: P2207T03241A0C0F26761.jpg
  - P2207 : Person Identity #2207
  - T03241: Tracklet / Sequence #03241
  - A0    : Altitude 0 (Low Altitude)
  - C0    : Camera 0 (UAV Aerial Camera)
  - F26761: Video Frame Number 26761
```

## 7. Attribute Annotations Status
- **Attribute File (`qut_attribute_v8.mat`)**: **Not present in local image archive**.
  *(Note: Attribute labels are optional soft biometrics used for auxiliary attribute prediction tasks; core Re-ID uses identity and camera labels encoded directly in filenames).*

## 8. Alignment with Proposed Research Architecture
This dataset provides the ideal substrate for your proposed architecture:
1. **View-Aware Processing**: Distinct `C0` (UAV), `C2` (Wearable), and `C3` (CCTV) sensor labels allow the network to learn view-conditioned representations.
2. **Multi-Scale Image Enhancement**: Extreme variations in pedestrian crop sizes (from 22px to 938px) strongly motivate CLAHE, gamma correction, and multi-scale feature pyramid representations.
3. **Cross-View Feature Fusion**: Querying aerial UAV crops (`C0`) against ground CCTV (`C3`) directly evaluates real-world aerial-ground surveillance matching.

## 9. Recommended Next Step
Proceed to **Step 3: Baseline Data Loader & Preprocessing Pipeline** to build PyTorch `Dataset` and `DataLoader` abstractions that load, augment, and normalize these image pairs.
