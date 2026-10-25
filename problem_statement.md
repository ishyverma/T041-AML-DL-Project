Project 12: Smart Waste Segregation Assistant
Title: Everyday Waste Sorting via Manual Image Descriptors and Transfer Learning
Project Overview & Abstract
Build an educational photo classifier that suggests a material class from an image, while
clearly documenting that local recycling rules and contamination determine real disposal
decisions.
The 'Hybrid' Logic: Fixed Image Descriptors to Reusable Learned Representations
AML baseline: Extract color histograms, HOG and Local Binary Pattern features; train SVM
and Random Forest classifiers.
Identify the gap: Manual descriptors can work in controlled photographs but are brittle
under lighting, viewpoint, background and object deformation changes.
DL model: Use MobileNetV2 or EfficientNet as a frozen feature extractor, then train a small
classification head with modest augmentation.
Fair comparison: Use a stratified split after duplicate checks. Report macro F1, class recall,
confidence calibration and qualitative failure cases.
Suggested Pre-reads
Thung and Yang (2016). Classification of Trash for Recyclability Status. Sandler et al.
(2018). MobileNetV2.
Datasets & Stack
Recommended starter dataset: TrashNet, beginning with its six material categories. Treat
labels as visual material classes, not universal disposal instructions.
Dataset Link: TrashNet - official six-class image dataset
Suggested stack: Python, OpenCV or scikit image, scikit learn, PyTorch or TensorFlow and
optional Gradio.
Feature Engineering & Setup Tips
Deduplicate near identical images, crop or letterbox consistently, use conservative
augmentation, balance class sampling, and add a low confidence abstain response in any
demo.
Expected Deliverable
A photo upload prototype with AML and transfer learning comparison, confidence scores and
a visible limitations note.
