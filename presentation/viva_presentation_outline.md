# Viva Presentation Deck Outline (10-Minute Defense)
> **Course:** Applied Machine Learning & Deep Learning (AML-DL)  
> **Project 12:** Everyday Waste Sorting via Manual Image Descriptors vs. Transfer Learning

---

## Slide 1: Title & Team
- **Title:** Everyday Waste Sorting via Manual Descriptors vs. Calibrated Deep Transfer Learning
- **Subtitle:** Investigating the Visual Representation Gap & Selective Abstention on Post-Consumer Recyclables
- **Course & Group ID:** AML-DL Group T041

## Slide 2: Real-World Motivation & The "Recycling Trap"
- Single-stream recycling contamination causes rejection of entire municipal pulping/sorting batches.
- Cost asymmetry: Misclassifying unidentifiable contaminants is far worse than asking a human to inspect them.
- Visual Identification $\neq$ Disposal Permission (organic contamination & local sorting technology limits).

## Slide 3: Problem Formulation & Dataset
- **Dataset:** TrashNet (Stanford CS229 Benchmark - 2,527 images across 6 classes: Cardboard, Glass, Metal, Paper, Plastic, Trash).
- **Class Imbalance:** 4.3:1 ratio (Paper: 594 vs Trash: 137). Addressed via stratified 70/15/15 split and inverse-frequency loss weighting.
- **Deduplication:** Perceptual dHash check to prevent train/val/test data leakage.

## Slide 4: Classical Applied ML Pipeline (1,634 Dimensions)
- **HSV Color Histograms (48D):** Chromaticity independent of lighting brightness.
- **Histogram of Oriented Gradients (HOG) (1568D):** Structural edges and silhouette geometry.
- **Local Binary Patterns (LBP) (18D):** Uniform surface texture roughness.
- **Classifiers:** SVM (RBF kernel with Platt scaling) & Balanced Random Forest.

## Slide 5: Deep Transfer Learning (MobileNetV2)
- Why MobileNetV2? Edge-friendly (3.5M parameters, ~14MB footprint), low latency, inverted residuals with linear bottlenecks.
- Two-Stage Training:
  - Phase 1: Frozen backbone + training Dropout(0.3) -> Linear classification head.
  - Phase 2: Unfreezing top bottleneck blocks with cosine annealing schedule.

## Slide 6: Benchmark Results Comparison
| Metric | Zero-Rule Baseline | AML: SVM (RBF) | AML: Random Forest | DL: MobileNetV2 |
|:---|:---:|:---:|:---:|:---:|
| **Test Accuracy** | 23.54% | 73.28% | 74.87% | **84.39%** |
| **Macro F1** | 0.0500 | 0.7199 | 0.7405 | **0.8268** |
| **Expected Calibration Error (ECE)** | — | **0.0590** | 0.3168 | 0.1038 |
| **With Abstain ($\tau=0.60$)** | — | 89.27% | — | **94.60%** |

## Slide 7: The "Representation Gap" (Why AML Fails on Real Waste)
- Crushed metal beverage cans destroy parallel edge gradients $\rightarrow$ HOG breaks down (+18.0% recall boost with DL).
- Crumpled plastic bottles produce false texture micro-gradients $\rightarrow$ LBP breaks down (+16.6% recall boost with DL).
- Display qualitative failure cases where CNN learned invariant spatial hierarchies.

## Slide 8: Confidence Calibration & The Abstain Filter
- Reliability diagram demonstrating calibrated probabilities.
- Pareto curve: By rejecting the bottom 15.87% most uncertain predictions ($\tau = 0.60$), retained accuracy jumps from **84.39% to 94.60%**.
- High-ambiguity items are diverted to manual sorting bins instead of contaminating recycling streams.

## Slide 9: Interactive Web Application Demo
- Live demo of the Gradio prototype:
  - Side-by-side inference: Classical AML vs. MobileNetV2.
  - Calibrated probability bar meters.
  - Active warning banners when confidence drops below safety threshold $\tau = 0.60$.
  - Educational municipal disposal context note.

## Slide 10: Conclusion & Future Scope
- Validated that hierarchical convolutional features fundamentally outpace hand-crafted descriptors on non-rigid waste deformations.
- Selective abstention provides an operationally viable safety net for municipal recycling.
- Future work: Multi-label segmentation (YOLOv8-seg) for composite waste items and INT8 edge quantization.
