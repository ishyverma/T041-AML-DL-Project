"""
Project 12: Smart Waste Segregation Assistant - Interactive Prototype
A photo upload web application featuring:
1. Dual inference: AML (Fixed Descriptors: Color+HOG+LBP -> SVM) vs. Deep Learning (MobileNetV2 Transfer Learning).
2. Side-by-side calibrated confidence scores & distribution bar chart.
3. Low-confidence abstention engine (Uncertainty thresholding to prevent contamination).
4. Educational limitations note & municipal recycling rules context.
"""

import sys
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image
import joblib
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from torchvision import transforms
import gradio as gr

from src.feature_extraction import extract_single_image_features

MODELS_DIR = REPO_ROOT / "models"
SPLITS_DIR = REPO_ROOT / "splits"
DATA_DIR = REPO_ROOT / "data" / "dataset-resized"

# 1. Load Metadata
with open(SPLITS_DIR / "classes.json", "r") as f:
    meta = json.load(f)
    CLASS_NAMES = meta["class_names"]

# 2. Load AML Model & Scaler
print("Loading AML Model (SVM) & Scaler...")
aml_scaler = joblib.load(MODELS_DIR / "aml_scaler.joblib")
aml_svm = joblib.load(MODELS_DIR / "aml_svm.joblib")

# 3. Load Deep Learning Model (MobileNetV2)
print("Loading Deep Learning Model (MobileNetV2)...")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
dl_model = mobilenet_v2(weights=None)
dl_model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(dl_model.last_channel, 6)
)
dl_model.load_state_dict(torch.load(MODELS_DIR / "mobilenetv2_best.pth", map_location=device))
dl_model.to(device)
dl_model.eval()

# Image Preprocessing for Deep Learning
dl_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

ABSTAIN_THRESHOLD = 0.60


def predict_waste(image_pil):
    if image_pil is None:
        return "Please upload an image.", {}, "Please upload an image.", {}, ""

    # Ensure RGB
    image_rgb = image_pil.convert("RGB")

    # Save to temp path for feature extractor
    temp_img_path = REPO_ROOT / "data" / ".temp_infer.jpg"
    image_rgb.save(temp_img_path)

    # --- AML Prediction (Color + HOG + LBP -> SVM) ---
    raw_features = extract_single_image_features(temp_img_path)
    scaled_features = aml_scaler.transform([raw_features])
    aml_probs = aml_svm.predict_proba(scaled_features)[0]
    aml_pred_idx = int(np.argmax(aml_probs))
    aml_pred_class = CLASS_NAMES[aml_pred_idx]
    aml_max_conf = float(aml_probs[aml_pred_idx])

    # --- Deep Learning Prediction (MobileNetV2) ---
    tensor_input = dl_transform(image_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = dl_model(tensor_input)
        dl_probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    dl_pred_idx = int(np.argmax(dl_probs))
    dl_pred_class = CLASS_NAMES[dl_pred_idx]
    dl_max_conf = float(dl_probs[dl_pred_idx])

    # Clean up temp image
    if temp_img_path.exists():
        temp_img_path.unlink()

    # Probability dictionaries for Gradio Label
    aml_prob_dict = {CLASS_NAMES[i]: float(aml_probs[i]) for i in range(len(CLASS_NAMES))}
    dl_prob_dict = {CLASS_NAMES[i]: float(dl_probs[i]) for i in range(len(CLASS_NAMES))}

    # --- Low Confidence Abstain Analysis ---
    if dl_max_conf < ABSTAIN_THRESHOLD:
        dl_status = f"ABSTAIN (Low Confidence: {dl_max_conf*100:.1f}%)"
        abstain_notice = (
            f"### Uncertainty Alert: Recommendation Abstained\n"
            f"The transfer learning model is uncertain about this item (Top confidence is **{dl_max_conf*100:.1f}%**, below the **{ABSTAIN_THRESHOLD*100:.0f}% safety threshold**).\n\n"
            f"**Action:** To avoid batch contamination, **do not dispose in a single-stream recycling bin** without verifying municipal guidelines."
        )
    else:
        dl_status = f"Confident: {dl_pred_class.upper()} ({dl_max_conf*100:.1f}%)"
        abstain_notice = (
            f"### Identified Material: **{dl_pred_class.upper()}** ({dl_max_conf*100:.1f}% confidence)\n"
            f"The neural network detected consistent visual material properties for **{dl_pred_class}**."
        )

    aml_status = f"{aml_pred_class.upper()} ({aml_max_conf*100:.1f}%)"

    return aml_status, aml_prob_dict, dl_status, dl_prob_dict, abstain_notice


# Find some representative sample images from each class for pre-loaded gallery
sample_images = []
for c in ["cardboard", "glass", "metal", "paper", "plastic", "trash"]:
    c_folder = DATA_DIR / c
    if c_folder.exists():
        samples = list(c_folder.glob("*.jpg"))
        if samples:
            sample_images.append(str(samples[0]))

# Build Gradio Interface
with gr.Blocks(title="Smart Waste Segregation Assistant") as demo:
    gr.Markdown(
        """
        # Smart Waste Segregation Assistant
        ### Everyday Waste Sorting via Manual Image Descriptors vs. Transfer Learning
        *An educational photo classifier exploring the transition from fixed image descriptors (Color + HOG + LBP) to reusable deep learning representations (MobileNetV2).*
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Upload Waste Photo (or Snap via Webcam)")
            classify_btn = gr.Button("Classify Material", variant="primary", size="lg")

            gr.Examples(
                examples=sample_images[:6],
                inputs=input_image,
                label="Pre-loaded Test Samples (Click to Test)"
            )

        with gr.Column(scale=1):
            status_box = gr.Markdown("### Upload an image and click **Classify Material** to begin.")
            
            with gr.Tabs():
                with gr.TabItem("Deep Learning (MobileNetV2)"):
                    dl_top_result = gr.Textbox(label="Transfer Learning Result", interactive=False)
                    dl_chart = gr.Label(num_top_classes=6, label="Learned Feature Probabilities")

                with gr.TabItem("Applied ML Baseline (SVM + HOG/LBP)"):
                    aml_top_result = gr.Textbox(label="AML Baseline Result", interactive=False)
                    aml_chart = gr.Label(num_top_classes=6, label="Manual Descriptor Probabilities")

    classify_btn.click(
        fn=predict_waste,
        inputs=[input_image],
        outputs=[aml_top_result, aml_chart, dl_top_result, dl_chart, status_box]
    )

    gr.Markdown(
        """
        ---
        ### Important Educational Note on Real-World Disposal
        > **Visual Material Identification != Universal Disposal Instructions.**  
        > 1. **Contamination Rules:** An item made of recyclable plastic or cardboard (like a greasy pizza box or unwashed container) **cannot be recycled** because organic residue ruins entire processing batches.  
        > 2. **Local Municipal Rules:** What is accepted in the blue bin depends strictly on your local waste management facility and municipal sorting technology. Always verify with your local guidelines before disposal.  
        > 3. **The 'Gap' Demonstrated:** Hand-crafted descriptors (HOG & LBP) easily break down on crushed cans, crumpled plastics, and dim lighting. MobileNetV2 uses hierarchical convolutional filters that generalize significantly better.
        """
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
