# Baseline ResNet-50 Model Evaluation Report

## 1. Project Objective
Baseline evaluation for **View-Aware Multi-Scale Image Enhancement and Feature Fusion for Aerial–Ground Person Re-Identification**.

## 2. Dataset Information
- **Benchmark**: AG-ReID.v2
- **Total Benchmark Images**: 100,502
- **Training Set (`train_all`)**: 51,530 images across 590 unique identities
- **Query Set**: 8,499 images (Probe set)
- **Gallery Set**: 40,473 images (Search database)
- **Sensor Modalities**: UAV Aerial (`C0`), Wearable Smart Glasses (`C2`), Ground CCTV (`C3`)

## 3. Baseline Architecture
- **Backbone**: ResNet-50 (ImageNet-1k Pretrained)
- **Modifications**: Last stride in layer4 set to 1 (preserves 2048 x 16 x 8 feature maps)
- **Pooling**: Global Average Pooling (GAP) -> 2048-dim embedding
- **Neck**: BNNeck (1D Batch Normalization without bias shift)
- **Classifier**: Linear(2048, 590) identity classification layer

## 4. Training Objectives & Losses
- **Identity Loss**: Cross-Entropy Loss over 590 training identities
- **Metric Loss**: Batch-Hard Triplet Loss with Euclidean Distance (margin = 0.3)
- **Combined Loss**: Total = 1.0 * CrossEntropy + 1.0 * Triplet

## 5. Hardware & Optimization Configuration
- **Compute Accelerator**: mps (Apple Silicon M5 GPU acceleration)
- **Mini-Batch Sampler**: RandomIdentitySampler (P=8 identities x K=4 images = 32 samples/batch)
- **Optimizer**: Adam (lr=0.00035, weight_decay=0.0005)
- **LR Scheduler**: Cosine Annealing
- **Training Epochs Completed**: 1

## 6. Baseline Performance Metrics
- **Rank-1 Accuracy**: **22.94%**
- **Rank-5 Accuracy**: (recorded in checkpoint)
- **Mean Average Precision (mAP)**: **12.61%**

## 7. Retrieval Visualizations
Sample retrieval rankings saved to:
`results/retrieval_examples/`

## 8. Limitations & Role in Next Steps
As expected from a raw baseline without image enhancement:
1. High-altitude UAV images (`C0` with `A2`) suffer from severe blur and low contrast, causing visual degradation.
2. Ground-to-aerial cross-view perspective changes remain challenging.
These baseline metrics serve as the primary benchmark for **Step 4 (CLAHE + Gamma Image Enhancement)** and subsequent multi-scale fusion modules.
