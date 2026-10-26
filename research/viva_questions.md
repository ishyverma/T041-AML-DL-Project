# Comprehensive Viva Voce Questions & Answers
> **Project Title:** Everyday Waste Sorting via Manual Image Descriptors vs. Transfer Learning  
> **Course:** Applied Machine Learning & Deep Learning (AML-DL)  
> **Dataset:** TrashNet (6 Classes: Cardboard, Glass, Metal, Paper, Plastic, Trash)

---

## Part 1: Scientific Motivation & Problem Formulation

### Q1: What is the core problem and scientific motivation of this project?
**Answer:**
Municipal recycling facilities face catastrophic batch contamination when non-recyclables or incompatible materials are tossed into single-stream bins (e.g., greasy cardboard or unwashed containers ruin entire pulping batches). Our project explores automated visual sorting by contrasting two paradigms:
1. **Classical Applied Machine Learning (AML):** Fixed, human-designed feature extractors (Color Histograms + HOG + LBP) paired with Support Vector Machines (SVM) and Random Forests.
2. **Deep Transfer Learning (DL):** Hierarchical representations learned via MobileNetV2 pre-trained on ImageNet.
We investigate *why* manual descriptors fail on deformed/crumpled post-consumer items, and demonstrate how confidence calibration and an uncertainty-based **abstain filter** ($\tau = 0.60$) prevent catastrophic disposal mistakes.

### Q2: Why formulate waste sorting as a "selective classification" / abstention problem?
**Answer:**
In everyday waste sorting, the cost of an error is asymmetric:
- If a bottle is classified as plastic, it goes into recycling.
- If an unidentifiable toxic chemical bottle or soiled food container is erroneously forced into recycling, it contaminates thousands of pounds of material.
Using Chow's rule for classification with an abstain option, when the model's posterior confidence is below $\tau = 0.60$, the system outputs **ABSTAIN** rather than guessing. On our held-out test set, this filter lifts effective accuracy from **84.39% to 94.60%** on retained decisions while abstaining on ~16% of high-ambiguity items.

### Q3: Why is visual material identification NOT equal to universal disposal instructions?
**Answer:**
A computer vision model only identifies the surface material composition (e.g., polyethylene terephthalate or cellulose fiber). Whether an item is accepted in a blue bin depends on:
1. **Organic Contamination:** A pizza box is made of recyclable cardboard, but grease and cheese residues cannot be separated during repulping.
2. **Municipal Sorting Infrastructure:** Many municipal optical sorters cannot handle black plastics or plastic bags that jam conveyor belts.
Therefore, a real-world system must explicitly warn users that material recognition is only a guidance layer, not an absolute municipal permit.

---

## Part 2: Applied Machine Learning & Feature Engineering

### Q4: Why combine HSV Color Histograms, HOG, and LBP? What does each descriptor provide?
**Answer:**
We fused three complementary, non-overlapping descriptors to mimic human visual perception:
1. **HSV Color Histograms (48 dimensions):** Captures chromaticity (Hue) and saturation (Saturation) separate from illumination intensity (Value). Differentiates green/brown glass and colored cardboard from white office paper.
2. **HOG — Histogram of Oriented Gradients (1,568 dimensions):** Evaluates local edge gradients across $16 \times 16$ pixel cells. Captures structural contours, silhouette geometry, and aspect ratios (e.g., straight bottle necks, rectangular boxes).
3. **LBP — Local Binary Patterns (18 dimensions):** Captures micro-texture and surface roughness by thresholding a circular neighborhood ($P=16, R=2$). Uniform LBP separates glossy reflective surfaces (glass/metal) from rough fibrous surfaces (cardboard/paper).
Total feature vector: $48 + 1568 + 18 = 1,634$ dimensions.

### Q5: Why did you use HSV instead of RGB for color histograms?
**Answer:**
In RGB color space, luminance and chromaticity are tightly coupled across all three channels. A shadow or dim room shifts all R, G, and B values simultaneously. In HSV, chromatic information ($H$ and $S$) is decoupled from lighting brightness ($V$), making color features significantly more invariant to shadows and non-uniform ambient light.

### Q6: Why did HOG and LBP struggle on deformed waste items?
**Answer:**
Fixed descriptors rely on spatial rigidity and stationary textures:
- **HOG breakdown:** When an aluminum soda can is crushed or stepped on, its canonical cylindrical shape and parallel edge gradients are replaced by chaotic, irregular creases. HOG interprets these sharp crumple gradients as crumpled paper.
- **LBP breakdown:** Clear plastic bottles or reflective cans produce harsh specular highlights and glares that register as high-contrast texture noise.
Deep CNNs overcome this because early convolutional layers extract low-level edges, while deeper layers form hierarchical, non-linear representations that are invariant to non-rigid deformations.

### Q7: Why did SVM (RBF) outperform Random Forest on manual descriptors?
**Answer:**
Our fused feature vector is 1,634 dimensions with high continuous correlation (smooth gradient histograms).
- **SVM with RBF Kernel:** Finds a maximum-margin hyperplane in a continuous Reproducing Kernel Hilbert Space (RKHS). It naturally handles high-dimensional, normalized continuous inputs ($L_2$-regularization prevents overfitting).
- **Random Forest:** Uses axis-aligned orthogonal decision splits. Partitioning smooth, correlated 1,634-dimensional gradient histograms with single-feature axis splits is sample-inefficient and tends to overfit on smaller datasets.
Result: SVM achieved **73.28% test accuracy (0.7199 Macro F1)** vs. Random Forest **74.87% accuracy with lower calibration (ECE 0.3168 vs SVM 0.0590)**.

---

## Part 3: Deep Transfer Learning & MobileNetV2

### Q8: Why select MobileNetV2 instead of heavier architectures like ResNet-50 or ViT?
**Answer:**
1. **Edge Feasibility:** Waste sorting happens at edge devices—smart bins, robotic sorting arms, and mobile phones. MobileNetV2 has only **3.5 million parameters** (~14 MB), compared to ResNet-50 (25.6M parameters, ~98 MB) and Vision Transformers (>85M parameters).
2. **Inverted Residuals & Depthwise Separable Convolutions:** By factoring standard 2D convolution into depthwise ($3\times3$) and pointwise ($1\times1$) convolutions, MobileNetV2 reduces computational complexity by 8–9x with minimal drop in accuracy.
3. **Small-Dataset Suitability:** TrashNet contains ~2,527 images. Heavy backbones overfit easily on small sample sizes, whereas MobileNetV2 acts as a regularized, efficient feature extractor.

### Q9: Describe your transfer learning training strategy.
**Answer:**
We employed a two-phase training protocol:
- **Phase 1 (Feature Extractor):** Froze all ImageNet pre-trained convolutional feature weights. Replaced the 1,000-class classifier with `Dropout(p=0.3) -> Linear(1280, 6)`. Trained the classification head for 10 epochs using AdamW ($lr = 10^{-3}$) and Cross-Entropy loss.
- **Phase 2 (Fine-Tuning):** Unfroze the top bottleneck inversion blocks (`features[14:]`) and trained end-to-end with a lower learning rate ($lr = 10^{-4}$) with cosine annealing schedule for 5 epochs.
- **Data Augmentation:** Conservative affine transformations (random horizontal flips, $\pm 15^\circ$ rotations, color jitter brightness $\pm 0.1$) to preserve material texture realism.

### Q10: How did you handle the severe class imbalance in TrashNet?
**Answer:**
TrashNet is naturally skewed: `paper` has 594 images, while `trash` has only 137 images (a 4.3:1 ratio).
1. **Stratified Splitting:** Partitioned the dataset using stratified 70% Train, 15% Validation, and 15% Test splits (378 test images).
2. **Cost-Sensitive Loss Weighting:** Applied inverse-frequency class weights in the PyTorch loss function:
   $$w_c = \frac{N}{K \cdot N_c}$$
   This penalized misclassifications on minority classes (`trash`) proportionally higher, lifting `trash` recall from 20% (unweighted baseline) to **70.0%**.

---

## Part 4: Calibration, Uncertainty & Real-World Safety

### Q11: What is Expected Calibration Error (ECE) and how is it calculated?
**Answer:**
A model is well-calibrated if its predicted probability reflects empirical accuracy: a prediction made with 80% confidence should be correct 80% of the time.
ECE partitions the predicted probability interval $[0, 1]$ into $M = 10$ equally spaced bins $B_1, \dots, B_M$. For each bin:
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
- SVM achieved an outstanding **ECE of 0.0590** (5.9%).
- MobileNetV2 achieved an **ECE of 0.1038** (10.4%).
- Uncalibrated Random Forest had a poor ECE of **0.3168** (31.7%), frequently predicting 0.5–0.6 probabilities with much higher error.

### Q12: How does the abstain threshold $\tau$ affect system throughput and accuracy?
**Answer:**
There is a fundamental Pareto trade-off between coverage (throughput) and precision (safety):
- At $\tau = 0.0$ (no rejection): 100% of images are classified; test accuracy is **84.39%**.
- At $\tau = 0.60$ (our chosen operational point): The model abstains on 15.87% of ambiguous items; accuracy on the accepted 84.13% throughput jumps to **94.60%**.
- At $\tau = 0.80$: 34.1% abstained; accuracy reaches **98.2%**.
For automated industrial sorting, rejected items are simply diverted to a secondary manual inspection belt rather than causing batch contamination.

---

## Part 5: Empirical Results & Defense Summary

### Q13: What are the main numerical results on the held-out test set (378 images)?
**Answer:**
- **Zero-Rule Baseline (Majority Class):** 23.54% Accuracy | 0.0500 Macro F1
- **Classical AML: SVM (Color + HOG + LBP):** 73.28% Accuracy | 0.7199 Macro F1 | 0.0590 ECE
- **Classical AML: Random Forest:** 74.87% Accuracy | 0.7405 Macro F1 | 0.3168 ECE
- **Deep Transfer Learning: MobileNetV2:** **84.39% Accuracy | 0.8268 Macro F1 | 0.1038 ECE**
- **MobileNetV2 with Abstain ($\tau = 0.60$):** **94.60% Retained Accuracy** (at 84.13% coverage)

### Q14: On which specific classes did MobileNetV2 produce the largest gains over AML?
**Answer:**
1. **Metal:** AML 75.4% $\rightarrow$ MobileNetV2 **93.44%** (+18.0% recall). Crushed beverage cans completely broke HOG's orientation grids.
2. **Plastic:** AML 65.3% $\rightarrow$ MobileNetV2 **81.94%** (+16.6% recall). Crinkled water bottles caused false texture signatures in LBP.
3. **Cardboard:** AML 72.1% $\rightarrow$ MobileNetV2 **88.52%** (+16.4% recall).

### Q15: What would you do differently if given another month on this project?
**Answer:**
1. **Multi-Label / Mixed Waste:** Real-world items are multi-material (e.g., a coffee cup has a paper sleeve, polyethylene lining, and polystyrene lid). We would formulate multi-label classification or instance segmentation (Mask R-CNN / YOLOv8-seg).
2. **Self-Supervised Domain Adaptation:** Fine-tune on unlabelled video streams from recycling conveyor belts using DINOv2 or SimCLR to adapt to dirty facility backgrounds.
3. **Hardware Deployment:** Quantize MobileNetV2 to INT8 via TensorRT or ONNX Runtime and benchmark inference latency on an NVIDIA Jetson Nano or Raspberry Pi 5.
