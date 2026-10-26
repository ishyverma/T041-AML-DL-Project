# Empirical Research Gap & Hypotheses
> **Project Title:** The Representation Gap in Post-Consumer Waste Sorting: Fixed Descriptors vs. Hierarchical Convolutional Features

---

## 1. The Core Research Gap

Automated optical sorting in recycling infrastructure has historically oscillated between two extremes:
1. **Classical Machine Learning with Fixed Image Descriptors (AML):** Historically favored due to low computational requirements and instant inference on simple microcontrollers. Descriptors like Color Histograms, HOG, and LBP assume spatial stationarity, clean object silhouettes, and uniform diffuse reflection.
2. **Deep Convolutional Architectures:** Deliver state-of-the-art accuracy, but standard implementations operate as uncalibrated black-boxes that force a hard prediction on every input, regardless of ambiguity or contamination.

### The Specific Empirical Gap Identified:
Existing literature on TrashNet demonstrates that Deep Learning outperforms Classical ML, but **fails to answer two critical operational questions**:
1. *Why exactly do hand-crafted descriptors fail on post-consumer waste?* At what geometric or radiometric failure points do HOG and LBP break down?
2. *How can automated classifiers be made safety-critical and risk-sensitive?* When an image presents high visual ambiguity (e.g., crushed translucent bottles or soiled containers), forcing a classification introduces intolerable batch contamination in municipal recycling streams.

---

## 2. Research Questions (RQs)

- **RQ1 (Representation Gap):** Under what specific visual deformations (crumpling, crushing, specular reflections, non-uniform lighting) does the fused feature space $[ \text{HSV} \parallel \text{HOG} \parallel \text{LBP} ]$ degrade relative to hierarchical convolutional feature maps?
- **RQ2 (Calibration Disparity):** How does confidence calibration (Expected Calibration Error) differ between maximum-margin kernel methods (SVM with Platt scaling) and deep neural networks (MobileNetV2)?
- **RQ3 (Safety-Critical Abstention):** Can a post-hoc confidence rejection mechanism ($\tau = 0.60$) based on calibrated posterior probabilities improve effective retained classification accuracy above 90% without excessively reducing system throughput?

---

## 3. Testable Hypotheses

- **H1 (Geometric Non-Stationarity):** For classes with non-rigid physical deformations (`metal` soda cans and `plastic` bottles), MobileNetV2 will exhibit a statistically significant recall advantage ($>15\%$ gain) over the AML baseline, whereas for structurally rigid items (`cardboard` boxes), the gap will be substantially narrower.
- **H2 (Calibration Robustness):** Platt-scaled SVM will achieve a lower Expected Calibration Error ($\text{ECE} < 0.08$) than unregularized Random Forests ($\text{ECE} > 0.25$), rendering SVM confidence estimates more reliable for direct thresholding.
- **H3 (Pareto Improvement via Abstention):** Applying an uncertainty threshold $\tau = 0.60$ to MobileNetV2 will elevate effective accuracy on retained predictions to $>92\%$, while maintaining an operational coverage of $\ge 80\%$.
