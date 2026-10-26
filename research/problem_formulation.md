# Mathematical Problem Formulation
> **Project Title:** Selective Waste Segregation via Manual Descriptors and Calibrated Deep Networks

---

## 1. Input Space & Classical Feature Representation

Let $\mathcal{X} \subset \mathbb{R}^{H \times W \times 3}$ denote the raw RGB image domain. Let $\mathcal{Y} = \{1, 2, \dots, K\}$ denote the discrete material label space, where $K = 6$ corresponding to:
$$\mathcal{Y} = \{\text{cardboard}, \text{glass}, \text{metal}, \text{paper}, \text{plastic}, \text{trash}\}$$

### Classical Applied ML Mapping:
In the classical paradigm, a deterministic, hand-crafted feature transformation function $\phi: \mathcal{X} \rightarrow \mathbb{R}^D$ maps each high-dimensional pixel array into a fused compact descriptor vector:
$$\mathbf{z} = \phi(\mathbf{x}) = \left[ \mathbf{z}_{\text{HSV}}^\top, \, \mathbf{z}_{\text{HOG}}^\top, \, \mathbf{z}_{\text{LBP}}^\top \right]^\top \in \mathbb{R}^{1634}$$
where:
1. $\mathbf{z}_{\text{HSV}} \in \mathbb{R}^{48}$: Concatenated 16-bin normalized histograms across Hue ($H$), Saturation ($S$), and Value ($V$) channels.
2. $\mathbf{z}_{\text{HOG}} \in \mathbb{R}^{1568}$: Block-normalized ($L_2\text{-Hys}$) gradient orientations computed across $16 \times 16$ pixel spatial cells with 8 orientation bins.
3. $\mathbf{z}_{\text{LBP}} \in \mathbb{R}^{18}$: Uniform Local Binary Pattern distribution evaluating circularly symmetric neighbor differences with radius $R=2$ and sampling points $P=16$.

The classifier $f_{\text{AML}}: \mathbb{R}^D \rightarrow \Delta^{K-1}$ computes class probability distribution vectors using a Support Vector Machine with an RBF kernel $K(\mathbf{z}, \mathbf{z}') = \exp(-\gamma \|\mathbf{z} - \mathbf{z}'\|^2)$ coupled with Platt scaling.

---

## 2. Deep Transfer Learning Representation

In deep transfer learning, the representation is parameterized by a convolutional neural network $f_{\boldsymbol{\theta}}: \mathcal{X} \rightarrow \mathbb{R}^K$ (MobileNetV2):
$$\hat{\mathbf{y}} = \sigma \left( \mathbf{W}_c \, \psi_{\text{conv}}(\mathbf{x}; \boldsymbol{\theta}_{\text{backbone}}) + \mathbf{b}_c \right)$$
where $\psi_{\text{conv}}(\cdot) \in \mathbb{R}^{1280}$ represents the global average pooled feature embedding from the inverted residual bottleneck blocks, $\mathbf{W}_c \in \mathbb{R}^{K \times 1280}$ is the linear classification head, and $\sigma(\cdot)$ is the softmax function:
$$P(Y = k \mid \mathbf{x}) = \frac{\exp(z_k)}{\sum_{j=1}^K \exp(z_j)}$$

### Class-Weighted Objective:
To combat class imbalance (ratio of 4.3:1 between paper and trash), the network minimizes the weighted categorical cross-entropy loss:
$$\mathcal{L}(\boldsymbol{\theta}) = - \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K w_k \cdot \mathbb{I}(y_i = k) \cdot \log \hat{P}(Y = k \mid \mathbf{x}_i)$$
where $w_k = \frac{N}{K \cdot N_k}$ ensures equal penalty gradient mass across minority and majority classes.

---

## 3. Selective Classification with Abstention (Chow's Rule)

In high-stakes recycling segregation where error costs significantly exceed rejection costs, we formulate classification with a rejection option. 

Let $g(\mathbf{x}) \in \mathcal{Y} \cup \{\text{ABSTAIN}\}$ be the decision function:
$$g(\mathbf{x}) = \begin{cases} \arg\max_{k \in \mathcal{Y}} P(Y = k \mid \mathbf{x}), & \text{if } \max_{k} P(Y = k \mid \mathbf{x}) \ge \tau \\ \text{ABSTAIN}, & \text{if } \max_{k} P(Y = k \mid \mathbf{x}) < \tau \end{cases}$$
where $\tau \in [0, 1]$ is the safety confidence threshold.

- **Coverage:** $C(\tau) = \mathbb{E}_{\mathbf{x}} [\mathbb{I}(\max_k P(Y=k|\mathbf{x}) \ge \tau)]$
- **Selective Accuracy:** $\text{Acc}_{\text{retained}}(\tau) = \frac{\sum_{i=1}^N \mathbb{I}(y_i = \hat{y}_i) \cdot \mathbb{I}(\max_k P(Y=k|\mathbf{x}_i) \ge \tau)}{\sum_{i=1}^N \mathbb{I}(\max_k P(Y=k|\mathbf{x}_i) \ge \tau)}$

---

## 4. Expected Calibration Error (ECE)

To guarantee that confidence thresholds $\tau$ correspond to actual probability of correctness, we evaluate calibration error across $M = 10$ confidence bins:
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
where:
$$\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \mathbb{I}(\hat{y}_i = y_i), \quad \text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i$$
