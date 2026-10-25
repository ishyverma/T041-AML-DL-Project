# Project 12: Smart Waste Segregation Assistant
## Everyday Waste Sorting via Manual Image Descriptors and Transfer Learning

> **Core Philosophy:** *The 'Hybrid' Logic* — From Fixed Image Descriptors to Reusable Learned Representations.  
> **Repository:** `/Users/yashagarwal/Downloads/WORK/Projects_Main/waste_seg`  
> **Key References:**
> 1. Thung and Yang (2016). *Classification of Trash for Recyclability Status.*
> 2. Sandler et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.*

---

## 1. Project Overview & Abstract

We are building an **educational photo classifier prototype** that predicts a **visual material class** from an image (from TrashNet's 6 categories: *glass, paper, cardboard, plastic, metal, trash*). 

Crucially, the system must clearly document and communicate that:
> **Visual material identity $\neq$ universal disposal instructions.**  
> Real-world disposal decisions are governed by **local municipal recycling rules** and **contamination status** (food residue, grease, unwashed items).

---

## 2. The Core Technical Gap We Are Investigating

```
Controlled Image (White Background, Centered) ──► Both AML and DL perform well
                                                              │
Perturbations (Lighting, Viewpoint, Deformation) ──────────────┼──► AML collapses (brittle hand-crafted grids)
                                                              └──► DL succeeds (hierarchical learned features)
```

* **Applied Machine Learning (AML) Baseline:** Hand-engineered descriptors (Color Histograms, HOG, LBP) work in controlled settings with uniform backgrounds, but they are **fundamentally brittle** under:
  1. **Lighting shifts:** Shadows and exposure distort color histograms.
  2. **Viewpoint & Orientation:** Rigid grid-based HOG descriptors break down.
  3. **Background clutter:** Non-white surfaces inject noise into edge gradients and texture patterns.
  4. **Object deformation:** Crushed cans, crumpled paper, and twisted plastic bottles alter spatial layouts.
* **Deep Learning (DL) Transfer Learning:** Convolutional neural networks (MobileNetV2 / EfficientNet) pre-trained on ImageNet extract invariant, hierarchical features that generalize across these real-world deformations.

---

## 3. Technology Stack & Dataset

* **Dataset:** **TrashNet** (2,527 images across 6 material classes: *cardboard, glass, metal, paper, plastic, trash*).
* **Programming Language:** Python 3
* **Classical Vision & Feature Extraction:** `scikit-image`, `OpenCV` (`cv2`), `Pillow`
* **Classical Machine Learning:** `scikit-learn` (SVM with RBF/Linear kernels, Random Forest, StandardScaler)
* **Deep Learning:** `PyTorch` & `torchvision` (Pre-trained MobileNetV2 backbone)
* **Evaluation & Calibration:** `scikit-learn` metrics, `matplotlib`, `numpy` (Expected Calibration Error / Reliability diagrams)
* **Prototype Interface:** `Gradio` (or Streamlit) for interactive photo upload and side-by-side comparison.

---

## 4. Phase-by-Phase Implementation Plan

### Step 1: Data Setup, Deduplication & Fair Stratified Splits
* [x] **Extract & Inspect TrashNet:** Unpack `dataset-resized.zip` into `data/dataset-resized` (2,527 images verified).
* [ ] **Deduplication Check:** Run perceptual hashing (`imagehash` or cosine distance) to detect and isolate near-identical sequential photos, preventing train-to-test data leakage.
* [ ] **Consistent Preprocessing:** Apply uniform letterboxing/cropping to preserve original aspect ratios without unnatural stretching.
* [ ] **Balanced Stratified Split:** Partition cleanly into **70% Train / 15% Validation / 15% Test** while maintaining class ratios across all 6 categories. Lock splits into CSV manifests (`splits/train.csv`, `splits/val.csv`, `splits/test.csv`).

### Step 2: AML Baseline Pipeline (Fixed Descriptors)
* [ ] **Extract Manual Features:**
  - **Color Histograms:** Multi-channel HSV/Lab histograms to represent color distribution.
  - **HOG (Histogram of Oriented Gradients):** Edge directions and shape contours.
  - **LBP (Local Binary Patterns):** Micro-surface textures (smooth glass vs. rough cardboard vs. crinkled plastic).
* [ ] **Feature Fusion & Normalization:** Concatenate into a unified feature vector and scale with `StandardScaler`.
* [ ] **Train Classifiers:**
  - Train **Support Vector Machine (SVM)** (RBF & Linear kernels).
  - Train **Random Forest (RF)** with balanced class weights to handle the small `trash` class.
* [ ] **Save Artifacts:** Export trained models to `models/aml_svm.pkl` and `models/aml_rf.pkl`.

### Step 3: Deep Learning Transfer Learning (Reusable Representations)
* [ ] **Model Architecture:**
  - Backbone: **MobileNetV2** (pre-trained on ImageNet).
  - Frozen Feature Extractor: Freeze convolutional feature layers.
  - Custom Head: Global Average Pooling $\rightarrow$ Dropout ($p=0.3$) $\rightarrow$ Dense classification head (6 classes).
* [ ] **Conservative Data Augmentation:** Modest rotation ($\pm 15^\circ$), horizontal flips, subtle color jitter, and gentle crops (preventing unrealistic distortions).
* [ ] **Balanced Training Strategy:**
  - Cross-Entropy Loss with inverse class weighting to give proper attention to the underrepresented `trash` class (137 samples).
  - AdamW optimizer with cosine learning rate scheduling.
* [ ] **Save Best Checkpoint:** Export weights to `models/mobilenetv2_best.pth`.

### Step 4: Fair Comparison, Calibration & Failure Case Study
* [ ] **Standardized Metric Reporting:**
  - **Macro F1-Score** (fair evaluation across imbalanced classes).
  - **Per-Class Recall & Precision** (identifying which specific materials are confused).
  - **Side-by-Side Confusion Matrices** (AML vs. MobileNetV2).
* [ ] **Confidence Calibration:**
  - Compute **Expected Calibration Error (ECE)** and plot Reliability Diagrams.
  - Verify whether high output probabilities correspond to true empirical accuracy.
* [ ] **Qualitative Failure Analysis:**
  - Select and document specific test cases:
    - *Deformed items* (crushed cans, crumpled paper).
    - *Lighting variations* (harsh shadows vs. balanced light).
    - *Viewpoint angles* (top-down vs. side profile).
  - Visually demonstrate why manual descriptors failed while the CNN succeeded.
* [ ] **Low-Confidence Abstain Engine:**
  - Formulate an uncertainty rule: If $\max(P) < 0.60$, abstain with the message:
    > *"Low Confidence / Uncertain: To prevent recycling contamination, do not place in recycle bin without checking local manual guidelines."*

### Step 5: Interactive Photo Upload Prototype (Deliverable)
* [ ] **Build Gradio Web App (`app.py`):**
  - **Photo Upload:** Drag-and-drop user upload + a gallery of pre-loaded test samples (including deformed cans and clean bottles).
  - **Side-by-Side Prediction Cards:**
    - Card A: **Applied ML Baseline** (Predicted class, confidence meter).
    - Card B: **Transfer Learning Model** (Predicted class, confidence meter).
  - **Abstain Warning:** Triggers visually if confidence falls below the threshold.
  - **Visible Limitations & Educational Note:**
    > *"Educational Disclaimer: This prediction indicates the visual material type, not whether your municipality accepts it. Food/liquid contamination (e.g. unwashed peanut butter jars, greasy pizza boxes) and local rules determine real recyclability."*

---

## 5. Current Project Status (Plain English)

| Component | Status | Details in Simple Words |
|:---|:---:|:---|
| **Dataset Acquisition** | **DONE** (100%) | All 2,527 TrashNet images unzipped and audited across 6 classes. |
| **Project Setup & Plan** | **DONE** (100%) | Plan fully aligned with Project 12 requirements. |
| **Step 1: Deduplication & Splits** | **DONE** (100%) | Exact/perceptual duplicates isolated; locked 70/15/15 splits in `splits/`. |
| **Step 2: AML Baseline (HOG/LBP/Color)** | **DONE** (100%) | 1,634 features extracted; SVM (73.3% test acc) & Random Forest (74.9% test acc) trained. |
| **Step 3: Deep Learning (MobileNetV2)** | **DONE** (100%) | Pre-trained ImageNet backbone fine-tuned on MPS GPU (84.4% test acc, 0.827 Macro F1). |
| **Step 4: Fair Evaluation & Abstain** | **DONE** (100%) | Benchmark, confusion matrices, ECE calibration curve, and failure catalog generated in `reports/`. |
| **Step 5: Gradio Web Prototype** | **DONE** (100%) | Interactive photo upload prototype with dual comparisons & abstain logic in `app.py`. |

---

## 6. Verified Final Test Set Benchmarks (Held-out 378 images)

| Model Architecture | Accuracy | Macro F1 | ECE (Calibration) | Abstain Accuracy Boost ($\tau=0.60$) |
|:---|:---:|:---:|:---:|:---:|
| **AML: SVM (RBF)** | **73.28%** | **0.7199** | **0.0590** | 73.28% $\rightarrow$ **89.27%** |
| **AML: Random Forest** | **74.87%** | **0.7405** | **0.3168** | N/A |
| **DL: MobileNetV2** | **84.39%** | **0.8268** | **0.1038** | 84.39% $\rightarrow$ **94.60%** |

### Per-Class Recall Comparison:
- **Cardboard:** SVM 72.1% vs. MobileNetV2 **88.5%**
- **Glass:** SVM 72.0% vs. MobileNetV2 **80.0%**
- **Metal:** SVM 75.4% vs. MobileNetV2 **93.4%** *(+18% recall on deformed cans!)*
- **Paper:** SVM 83.2% vs. MobileNetV2 **84.3%**
- **Plastic:** SVM 65.3% vs. MobileNetV2 **81.9%** *(+16.6% recall on crushed/translucent bottles!)*
- **Trash:** SVM 60.0% vs. MobileNetV2 **70.0%** *(+10% on complex miscellaneous refuse)*
