# Smart Waste Segregation Assistant: Project Roadmap

> **Project Title:** Everyday Waste Sorting via Manual Image Descriptors and Transfer Learning  
> **Core Dataset:** TrashNet (6 material categories: *cardboard, glass, metal, paper, plastic, trash*)  
> **Target Deliverable:** An educational, interactive photo classifier prototype providing dual AML vs. Deep Learning comparisons, calibrated confidence scores, an abstain mechanism, and practical recycling advisory notes.

---

## 1. Executive Summary & Architecture

The objective of this project is to explore the transition from **Classical Applied Machine Learning (AML)** with hand-crafted visual descriptors (Color Histograms, HOG, LBP) to **Modern Deep Learning (Transfer Learning)** using convolutional backbones (MobileNetV2, EfficientNet).

Beyond simple accuracy, this project emphasizes:
1. **Methodological Rigor:** Fair, leak-free benchmarking with deduplication and stratified splits.
2. **Failure Analysis:** Demonstrating why hand-engineered descriptors break down under lighting, background clutter, and deformation.
3. **Safety & Calibration:** Implementing low-confidence abstention to prevent harmful real-world sorting mistakes.
4. **Context Awareness:** Communicating that visual material classification $\neq$ universal disposal instructions (local municipal rules & contamination dictate reality).

```
               +-------------------------------------------+
               |             Input Waste Image             |
               +---------------------+---------------------+
                                     |
                                     v
                       +---------------------------+
                       | Preprocessing & Check:    |
                       | Resize, Deduplication,    |
                       | Quality Assessment        |
                       +-------------+-------------+
                                     |
              +----------------------+----------------------+
              |                                             |
              v                                             v
  +------------------------+                   +------------------------+
  |    AML Pipeline        |                   | Deep Learning Pipeline |
  | (Manual Descriptors)   |                   |  (Transfer Learning)   |
  +------------------------+                   +------------------------+
  | • HSV/Lab Histograms   |                   | • MobileNetV2 /        |
  | • HOG (Shape/Edges)    |                   |   EfficientNet-B0      |
  | • LBP (Texture)        |                   | • Frozen Backbone      |
  | • Feature Fusion       |                   | • Custom Dense Head    |
  | • SVM / Random Forest  |                   | • Data Augmentations   |
  +-----------+------------+                   +-----------+------------+
              |                                             |
              +----------------------+----------------------+
                                     |
                                     v
                       +---------------------------+
                       | Confidence Calibration &  |
                       |   Abstain Decision Engine |
                       +-------------+-------------+
                                     |
                                     v
                       +---------------------------+
                       | Interactive Web Prototype |
                       | (AML vs. DL Comparison,   |
                       | Probabilities & Caveats)  |
                       +---------------------------+
```

---

## 2. What We SHOULD Do (Core Requirements & MVP)

These are the non-negotiable foundations required to fulfill the project specification and deliver a rigorous comparative study.

### Phase 1: Data Audit, Cleansing & Deduplication
- [ ] **Extract & Inspect TrashNet:**
  - Unpack `dataset-resized.zip` (located in [trashnet/data](file:///Users/yashagarwal/Downloads/WORK/Projects_Main/waste_seg/trashnet/data)).
  - Validate image formats, resolutions, channel counts, and verify the 6 classes (*glass, paper, cardboard, plastic, metal, trash*).
- [ ] **Near-Duplicate Detection:**
  - Run perceptual hashing (`imagehash`: pHash/dHash) or cosine distance on deep feature embeddings to detect near-identical frames shot sequentially.
  - Remove/isolate duplicates to eliminate train-to-test data leakage.
- [ ] **Rigorous Stratified Splitting:**
  - Generate clean **Train (70%) / Validation (15%) / Test (15%)** stratified splits.
  - Save split manifests (CSV or JSON) to ensure deterministic, reproducible benchmarking across all models.

---

### Phase 2: Classical Machine Learning (AML) Baseline
- [ ] **Hand-Crafted Feature Engineering:**
  - **Color Descriptors:** Multi-channel Color Histograms in HSV or CIE-Lab color space (captures material color distributions, e.g., brown cardboard vs. clear glass).
  - **Shape & Edge Descriptors:** Histogram of Oriented Gradients (HOG) using `scikit-image` or `cv2` (captures structural silhouettes and contours).
  - **Texture Descriptors:** Local Binary Patterns (LBP) to capture micro-surface textures (crinkled plastic, smooth glass, fibrous cardboard).
- [ ] **Feature Fusion & Normalization:**
  - Concatenate features into a unified vector, followed by standard scaling (`StandardScaler` or MinMax normalization).
- [ ] **AML Model Training & Tuning:**
  - Train **Support Vector Classifier (Linear and RBF kernels)**.
  - Train **Random Forest Classifier** and **Gradient Boosting (e.g. LightGBM / XGBoost)**.
  - Tune hyper-parameters via Stratified K-Fold Cross-Validation on the training split.

---

### Phase 3: Transfer Learning Deep Learning Model
- [ ] **Backbone Selection & Setup:**
  - Implement **MobileNetV2** (lightweight, mobile-ready) and/or **EfficientNet-B0** in PyTorch or TensorFlow.
  - Initialize with ImageNet pre-trained weights.
- [ ] **Architecture Design:**
  - Freeze the convolutional feature extractor.
  - Attach a compact classification head: Global Average Pooling $\rightarrow$ Dropout ($p=0.2$ to $0.4$) $\rightarrow$ Dense / Linear layer ($6$ output logits).
- [ ] **Data Augmentation:**
  - Implement conservative transformations appropriate for waste photos: Random horizontal/vertical flip, slight rotation ($\pm 15^\circ$), mild color jitter, and random resized crop / letterboxing.
- [ ] **Training Strategy:**
  - Stage 1: Train the custom head with a standard learning rate (e.g., $10^{-3}$, AdamW).
  - Stage 2 (Optional/Fine-tuning): Unfreeze top 1–2 convolutional blocks with a low learning rate ($10^{-5}$) using cosine annealing.

---

### Phase 4: Comparative Evaluation & Rigorous Analysis
- [ ] **Standardized Metric Suite:**
  - Compute **Macro F1-Score**, **Per-Class Precision & Recall**, and **Overall Accuracy** on the unseen test set.
  - Generate annotated **Confusion Matrices** comparing AML vs. Deep Learning side-by-side.
- [ ] **Confidence Calibration & Reliability:**
  - Measure **Expected Calibration Error (ECE)** and plot **Reliability Diagrams**.
  - Compare whether the raw softmax probabilities reflect true empirical accuracy.
- [ ] **Documenting the 'Gap' & Qualitative Failure Case Study:**
  - Curate specific edge cases:
    - *Lighting variances* (harsh shadows, underexposure).
    - *Viewpoint shifts & partial occlusions*.
    - *Deformations* (crushed cans, torn paper, crumpled plastic bottles).
    - *Cluttered or non-white backgrounds*.
  - Explicitly explain why hand-crafted HOG/LBP fail on deformable items whereas hierarchical convolutional filters generalize.
- [ ] **Low-Confidence Abstain Mechanism:**
  - Implement a safety threshold (e.g., if $\max(P) < 0.60$ or entropy is high, flag: *"Low confidence / Uncertain. Please do not contaminate the recycle bin — consult local guidelines"*).

---

### Phase 5: Interactive Web Prototype & Educational UX
- [ ] **Build an Interactive Web Application:**
  - Framework: Streamlit, Gradio, or FastAPI + Modern Web UI.
- [ ] **Key UI Components:**
  - **Drag-and-Drop Photo Uploader** with sample test images provided.
  - **Dual Prediction Panel:** Shows predictions from AML (SVM/RF) vs. Deep Learning (MobileNetV2/EfficientNet) side-by-side with confidence bar meters.
  - **Visual Attention / Interpretability (Optional but recommended):** Grad-CAM heatmap showing what parts of the object the neural network focused on.
  - **Prominent Disclaimer & Contamination Warning:**
    > *"Notice: Material classification (e.g. plastic) does not guarantee recyclability. Food contamination, unwashed containers, and local municipal regulations ultimately determine disposal."*

---

## 3. What We COULD Do (Advanced Extensions & Enhancements)

These represent high-impact features and ambitious extensions that elevate the project into a comprehensive, portfolio-grade system.

### A. Advanced Computer Vision & Multi-Object Handling
1. **Object Detection for Cluttered Bins (YOLOv8 / RT-DETR):**
   - Move from single-object centered photos to multi-object scene parsing.
   - Detect and segment multiple discarded items simultaneously in a single tray or bin snapshot.
2. **Open-Vocabulary / Zero-Shot Classification (CLIP / SigLIP):**
   - Leverage OpenAI CLIP or Google SigLIP embeddings to classify rare or non-TrashNet waste classes (e.g., *electronic waste, batteries, compostable bioplastics, textiles, medical blister packs*) without retraining.
3. **Foundation Model Embeddings (DINOv2):**
   - Extract self-supervised frozen representations using Meta's DINOv2 vision transformer, comparing its linear probe accuracy against MobileNetV2 and manual descriptors.
4. **Plastic Resin Code (RIC 1–7) Recognition via OCR:**
   - Detect the small recycling symbol (chasing arrows 1 to 7: PETE, HDPE, PVC, LDPE, PP, PS, OTHER) using a crop-and-OCR pipeline.

---

### B. Multimodal LLM Reasoning & Local Policy Agents
1. **Multimodal LLM Co-Pilot (Gemini Flash / GPT-4o-mini / LLaVA):**
   - Chain the classification output to a Vision-Language Model to generate tailored prep instructions:
     - *"This is a plastic salad container. Before tossing into recycling, rinse out remaining oil/dressing, and remove the paper barcode sticker."*
2. **Geo-Location & Municipal Rule Resolver:**
   - Incorporate a municipality lookup (by zip code / city name) to reflect divergent local rules (e.g., San Francisco curbside composting vs. cities without glass collection).
3. **Contamination & Cleanliness Scoring:**
   - Train a sub-classifier or use visual heuristics to estimate food/liquid grease contamination level (Clean vs. Greasy/Dirty).

---

### C. Edge Deployment & Real-World Hardware
1. **Edge Optimization (TFLite / ONNX Runtime / CoreML):**
   - Quantize the MobileNetV2 model to INT8 (post-training quantization).
   - Benchmark inference latency and model size ($< 10$ MB, $< 15$ ms CPU latency).
2. **Smart Bin Hardware Prototype:**
   - Deploy onto a Raspberry Pi or NVIDIA Jetson Nano connected to a USB/Pi Camera and servo motors that automatically tilt into the correct bin divider.
3. **Mobile-Friendly Progressive Web App (PWA):**
   - Enable live mobile camera scanning with offline inference support using ONNX Web / WebGL in the browser.

---

### D. Data-Centric AI & Continuous Learning
1. **Dataset Expansion (TACO & ZeroWaste):**
   - Augment TrashNet with the **TACO (Trash Annotations in Context)** dataset to evaluate out-of-distribution performance in wild/outdoor settings.
2. **Active Learning Feedback Loop:**
   - Add a *"Flag Incorrect Prediction"* button in the web demo, logging disputed samples to a review queue with user-submitted correct tags.
3. **Diffusion-Based Data Synthesis:**
   - Use Stable Diffusion / ControlNet to synthesize realistic trash items placed on diverse kitchen counters, sidewalks, and cafeteria tables to stress-test model robustness.

---

### E. Environmental Impact & Gamification
1. **Eco-Metrics Calculator:**
   - Quantify environmental savings per classified item (e.g., grams of $CO_2$ avoided, liters of water saved, landfill volume diverted).
2. **Disposal Guidance Quiz / Gamified Learning:**
   - An interactive game challenging users to sort 10 tricky items (e.g., coffee cups with plastic linings, greasy pizza boxes, broken ceramics) before revealing AI explanations.

---

## 4. Suggested Implementation Phases

| Phase | Milestone | Expected Time | Key Deliverables |
|:---|:---|:---:|:---|
| **Phase 1** | Data Audit & Stratified Setup | 1–2 Days | Cleaned splits, deduplication script, EDA notebook |
| **Phase 2** | AML Baseline Pipeline | 2–3 Days | Color/HOG/LBP extractors, SVM/RF models, baseline metrics |
| **Phase 3** | Deep Learning (Transfer Learning) | 2–3 Days | MobileNetV2/EfficientNet training script, weights checkpoint |
| **Phase 4** | Rigorous Evaluation & Calibration | 2 Days | Confusion matrix, macro F1 comparison, ECE curve, failure study |
| **Phase 5** | Prototype & UI Demo | 2–3 Days | Streamlit / Gradio web app with dual predictions & abstain mode |
| **Phase 6** | Selected Advanced Enhancements | Flexible | Grad-CAM, VLM reasoning, or Edge INT8 quantization |

---

## 5. Summary Matrix: "Should Do" vs. "Could Do"

| Area | What We **SHOULD** Do (Core / Baseline) | What We **COULD** Do (Advanced / Value-Add) |
|:---|:---|:---|
| **Data** | TrashNet, deduplication, clean 70/15/15 stratified split | TACO dataset integration, synthetic lighting variations |
| **Classical ML** | HSV Histograms + HOG + LBP fused into SVM & Random Forest | Automated feature selection (PCA/mRMR), ensemble stacking |
| **Deep Learning** | MobileNetV2/EfficientNet transfer learning with frozen backbone | DINOv2 self-supervised embeddings, YOLOv8 multi-item detection |
| **Metrics** | Macro F1, Per-class Recall, Calibration Error, Failure Catalog | Adversarial robustness testing, inference latency benchmarking |
| **Safety / UX** | Softmax threshold abstention ("Low confidence / Unsure") | VLM step-by-step cleaning instructions, resin code (RIC) OCR |
| **Deployment** | Local Gradio / Streamlit upload demo with dual comparisons | On-device INT8 quantization, mobile PWA, smart-bin camera trigger |
