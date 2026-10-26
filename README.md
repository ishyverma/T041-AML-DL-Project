# Everyday Waste Sorting via Manual Image Descriptors vs. Transfer Learning
### Smart Waste Segregation Assistant (Project 12)
**Course:** Applied Machine Learning & Deep Learning (AML-DL) | Group T041

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch](https://img.shields.io/badge/framework-PyTorch%20%2F%20TorchVision-ee4c2c.svg)](https://pytorch.org/)
[![UI: Gradio](https://img.shields.io/badge/UI-Gradio-orange.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An academic machine learning research project investigating the representation transition from **Classical Applied ML (fixed hand-crafted descriptors)** to **Modern Deep Transfer Learning (reusable convolutional representations)** on the 6-class **TrashNet** benchmark (*cardboard, glass, metal, paper, plastic, trash*), incorporating probability calibration and risk-sensitive selective abstention.

---

## Project Structure

```
waste_seg/
├── README.md                      # Comprehensive project overview and documentation
├── requirements.txt               # Pinned Python package dependencies
├── app.py                         # Clean interactive Gradio dual-inference application
├── data/
│   ├── dataset-resized/           # Extracted 6-class benchmark images
│   └── features_aml.npz           # Cached 1,634-dim fused feature matrices
├── models/
│   ├── aml_scaler.joblib          # Fitted StandardScaler for continuous descriptors
│   ├── aml_svm.joblib             # Calibrated Support Vector Machine (RBF kernel)
│   ├── aml_rf.joblib              # Balanced Random Forest (200 estimators)
│   └── mobilenetv2_best.pth       # MobileNetV2 transfer learning checkpoint
├── notebooks/                     # Sequential, executed exploratory & evaluation notebooks
│   ├── 01_dataset_eda.ipynb       # Class distribution, resolutions, sample montage
│   ├── 02_feature_extraction_aml.ipynb # Visualizing HSV, HOG, and LBP descriptors
│   ├── 03_classical_aml_models.ipynb   # SVM & Random Forest training & tuning
│   ├── 04_deep_learning_mobilenetv2.ipynb # MobileNetV2 fine-tuning & evaluation
│   └── 05_evaluation_safety_tradeoff.ipynb # Confusion matrices, ECE & abstention curve
├── presentation/                  # Defense slides and presentation materials
│   └── viva_presentation_outline.md  # 10-slide deck outline for project defense
├── report/                        # Final IEEE-style academic report
│   ├── AML_Report.pdf             # Compiled final PDF document
│   ├── report.tex                 # LaTeX master source
│   └── references.bib             # BibTeX citations
├── research/                      # Evaluation & Viva Voce preparation
│   ├── viva_questions.md          # 15+ rigorous viva Q&A with model answers
│   ├── problem_formulation.md     # Mathematical formulation of representations & ECE
│   ├── research_gap.md            # Empirical research gap & testable hypotheses
│   └── literature_review_matrix.csv # 8-paper comparative analysis matrix
├── results/                       # Centralized experimental results
│   ├── figures/                   # High-resolution publication plots
│   │   ├── calibration_curve.png
│   │   ├── confusion_matrices.png
│   │   ├── dataset_overview.png
│   │   ├── failure_cases_aml_vs_dl.png
│   │   └── fig4_confidence_abstain_tradeoff.png
│   └── metrics/                   # Standalone CSV metric summaries
│       ├── master_model_comparison.csv
│       ├── classification_results.csv
│       ├── per_class_performance.csv
│       ├── abstain_threshold_sweep.csv
│       ├── dataset_split_summary.csv
│       ├── failure_cases_analysis.csv
│       └── feature_dictionary.csv
├── scripts/                       # Reproduction and utility scripts
│   └── generate_notebooks.py      # Notebook generator and validator
├── splits/                        # Deterministic, deduplicated manifests
│   ├── classes.json
│   ├── train.csv (70%)
│   ├── val.csv (15%)
│   └── test.csv (15%)
├── src/                           # Modular production pipeline
│   ├── data_split.py              # MD5 & dHash deduplication and stratified splitting
│   ├── dataset_loader.py          # PyTorch datasets with data augmentations
│   ├── feature_extraction.py      # Fused [HSV | HOG | LBP] feature engineering
│   ├── train_aml.py               # Classical ML training and serialization
│   ├── train_dl.py                # MobileNetV2 two-stage fine-tuning
│   └── evaluate.py                # Final test evaluation, ECE & failure analysis
└── tests/                         # Unit and smoke tests
    └── test_pipeline.py           # Pipeline verification test suite
```

---

## Benchmark Results (Held-Out Test Set: 378 Images)

| Model Architecture | Accuracy | Macro F1 | Calibration (ECE) | With Abstain Filter ($\tau = 0.60$) |
|:---|:---:|:---:|:---:|:---:|
| **Zero-Rule Baseline (Majority)** | 23.54% | 0.0500 | — | — |
| **AML: SVM (RBF Kernel)** | **73.28%** | **0.7199** | **0.0590** | 73.28% $\rightarrow$ **89.27%** |
| **AML: Random Forest** | **74.87%** | **0.7405** | **0.3168** | — |
| **DL: MobileNetV2** | **84.39%** | **0.8268** | **0.1038** | 84.39% $\rightarrow$ **94.60%** |

### Per-Class Recall Highlights:
* **Metal:** AML 75.4% $\rightarrow$ MobileNetV2 **93.44%** *(+18.0% recall gain on crushed beverage cans)*
* **Plastic:** AML 65.3% $\rightarrow$ MobileNetV2 **81.94%** *(+16.6% recall gain on crumpled bottles)*
* **Cardboard:** AML 72.1% $\rightarrow$ MobileNetV2 **88.52%**
* **Trash:** AML 60.0% $\rightarrow$ MobileNetV2 **70.00%** *(balanced loss weighting significantly lifts underrepresented classes)*

---

## Qualitative Failure Case Study

![Failure Cases](results/figures/failure_cases_aml_vs_dl.png)

*Fixed descriptors (HOG/LBP) break down on non-rigid physical deformations (crushed cans, crumpled plastics) where canonical edge gradients collapse. MobileNetV2 learns hierarchical spatial representations that remain invariant to deformation.*

---

## Interactive Notebooks

Step-by-step interactive research notebooks located in `notebooks/`:
1. [01_dataset_eda.ipynb](file:///Users/yashagarwal/Downloads/WORK/Projects_Main/waste_seg/notebooks/01_dataset_eda.ipynb): Data distributions, class imbalance, and visual sample montages.
2. [02_feature_extraction_aml.ipynb](file:///Users/yashagarwal/Downloads/WORK/Projects_Main/waste_seg/notebooks/02_feature_extraction_aml.ipynb): Visual extraction of HSV histograms, HOG contours, and LBP texture maps.
3. [03_classical_aml_models.ipynb](file:///Users/yashagarwal/Downloads/WORK/Projects_Main/waste_seg/notebooks/03_classical_aml_models.ipynb): Classical model training, cross-validation, and decision boundary analysis.
4. [04_deep_learning_mobilenetv2.ipynb](file:///Users/yashagarwal/Downloads/WORK/Projects_Main/waste_seg/notebooks/04_deep_learning_mobilenetv2.ipynb): Transfer learning setup, two-phase training, and validation tracking.
5. [05_evaluation_safety_tradeoff.ipynb](file:///Users/yashagarwal/Downloads/WORK/Projects_Main/waste_seg/notebooks/05_evaluation_safety_tradeoff.ipynb): Expected Calibration Error, confidence threshold sweeps, and failure case study.

---

## Quickstart & Local Installation

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/ishyverma/T041-AML-DL-Project.git
cd T041-AML-DL-Project
pip install -r requirements.txt
```

### 2. Run Test Suite
Verify that all feature extraction pipelines, data loaders, and model weights operate properly:
```bash
python -m unittest tests/test_pipeline.py
```

### 3. Launch the Interactive Web Application
```bash
python app.py
```
Open **[http://127.0.0.1:7860](http://127.0.0.1:7860)** in your browser to evaluate dual inference, confidence meters, and the real-time uncertainty alert.

---

## References
1. **Gary Thung and Mindy Yang (2016)**. *Classification of Trash for Recyclability Status*. Stanford University CS229 Technical Report.
2. **Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen (2018)**. *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. CVPR 2018.
3. **Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger (2017)**. *On Calibration of Modern Neural Networks*. ICML 2017.
4. **Navneet Dalal and Bill Triggs (2005)**. *Histograms of Oriented Gradients for Human Detection*. CVPR 2005.
