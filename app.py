"""
Project 12: Smart Waste Segregation Assistant - Interactive Prototype
A photo upload web application featuring:
1. Multi-model comparison: Classical AML (SVM & Random Forest on Color+HOG+LBP) vs. Transfer Learning (MobileNetV2).
2. Side-by-side calibrated confidence scores & distribution charts.
3. Interactive low-confidence abstention engine with adjustable safety threshold (tau).
4. Class-specific municipal disposal guidelines & contamination alerts.
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

# 2. Load AML Models & Scaler
print("Loading AML Models (SVM, Random Forest) & Scaler...")
aml_scaler = joblib.load(MODELS_DIR / "aml_scaler.joblib")
aml_svm = joblib.load(MODELS_DIR / "aml_svm.joblib")
aml_rf = joblib.load(MODELS_DIR / "aml_rf.joblib")

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

# Class-Specific Municipal Disposal & Contamination Guidelines
DISPOSAL_GUIDELINES = {
    "cardboard": (
        "### Municipal Disposal Guidelines: Cardboard\n"
        "- **Preparation:** Flatten and break down all boxes to conserve space in municipal collection bins.\n"
        "- **Contamination Warning:** Greasy pizza boxes, wax-coated frozen food containers, and wet cardboard **cannot be recycled** because organic grease ruins repulping batches. Place soiled cardboard in Trash or Organic Compost.\n"
        "- **Sorting Destination:** Blue / Paper Recycling Bin (clean items only)."
    ),
    "glass": (
        "### Municipal Disposal Guidelines: Glass\n"
        "- **Preparation:** Empty liquids and lightly rinse beverage bottles and food jars. Metal bottle caps and lids can usually be recycled separately.\n"
        "- **Contamination Warning:** Drinking glasses, mirrors, window panes, and Pyrex/ceramics have different melting points and **ruin glass furnace batches**—dispose of non-container glass in general Trash.\n"
        "- **Sorting Destination:** Glass Recycling Bin / Bottle Bank."
    ),
    "metal": (
        "### Municipal Disposal Guidelines: Metal\n"
        "- **Preparation:** Empty and rinse aluminum beverage cans and steel food tins.\n"
        "- **Hazardous Restriction:** Aerosol cans, paint tins, and chemical canisters must go to specialized hazardous drop-offs, NOT curbside recycling.\n"
        "- **Sorting Destination:** Blue / Metal Recycling Bin."
    ),
    "paper": (
        "### Municipal Disposal Guidelines: Paper\n"
        "- **Preparation:** Clean office paper, newspapers, magazines, and plain paper bags.\n"
        "- **Contamination Warning:** Soiled paper towels, napkins, waxed paper, and disposable coffee cups with plastic linings cannot be pulped—dispose in Trash.\n"
        "- **Sorting Destination:** Blue / Paper Recycling Bin."
    ),
    "plastic": (
        "### Municipal Disposal Guidelines: Plastic\n"
        "- **Preparation:** Empty all liquids and rinse food residue.\n"
        "- **Resin Code & Municipal Check:** Rigid plastic bottles (PET #1 and HDPE #2) are widely accepted curbside. Plastic shopping bags/films (#4) and Styrofoam (#6) jam optical sorters—check local municipal guidelines.\n"
        "- **Sorting Destination:** Blue / Plastic Recycling Bin (clean rigid containers only)."
    ),
    "trash": (
        "### Municipal Disposal Guidelines: General Non-Recyclable Trash\n"
        "- **Destination:** Standard Landfill / Municipal Waste Bin.\n"
        "- **Hazardous Check:** Check whether items contain batteries, electronics, or toxic chemicals requiring dedicated municipal e-waste recycling drop-offs."
    )
}


def predict_waste(image_pil, abstain_threshold):
    if image_pil is None:
        empty_chart = {}
        return "Please upload an image.", empty_chart, "N/A", empty_chart, "N/A", empty_chart, "### Please upload an image to begin.", ""

    # Ensure RGB
    image_rgb = image_pil.convert("RGB")

    # Save to temp path for feature extractor
    temp_img_path = REPO_ROOT / "data" / ".temp_infer.jpg"
    image_rgb.save(temp_img_path)

    # --- AML Predictions (Color + HOG + LBP) ---
    raw_features = extract_single_image_features(temp_img_path)
    scaled_features = aml_scaler.transform([raw_features])

    # SVM Prediction
    svm_probs = aml_svm.predict_proba(scaled_features)[0]
    svm_pred_idx = int(np.argmax(svm_probs))
    svm_pred_class = CLASS_NAMES[svm_pred_idx]
    svm_max_conf = float(svm_probs[svm_pred_idx])

    # Random Forest Prediction
    rf_probs = aml_rf.predict_proba(scaled_features)[0]
    rf_pred_idx = int(np.argmax(rf_probs))
    rf_pred_class = CLASS_NAMES[rf_pred_idx]
    rf_max_conf = float(rf_probs[rf_pred_idx])

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
    dl_prob_dict = {CLASS_NAMES[i]: float(dl_probs[i]) for i in range(len(CLASS_NAMES))}
    svm_prob_dict = {CLASS_NAMES[i]: float(svm_probs[i]) for i in range(len(CLASS_NAMES))}
    rf_prob_dict = {CLASS_NAMES[i]: float(rf_probs[i]) for i in range(len(CLASS_NAMES))}

    # --- Low Confidence Abstain Analysis ---
    if dl_max_conf < abstain_threshold:
        dl_status = f"ABSTAIN (Low Confidence: {dl_max_conf*100:.1f}%)"
        status_notice = (
            f"### Uncertainty Alert: Automated Decision Abstained\n"
            f"The transfer learning model is uncertain about this item (Top confidence is **{dl_max_conf*100:.1f}%**, below the **{abstain_threshold*100:.0f}% safety threshold**).\n\n"
            f"**Safety Action:** To avoid recycling batch contamination, **do not dispose in a single-stream recycling bin** without verifying municipal guidelines."
        )
        disposal_notice = (
            "### Disposal Recommendation Abstained\n"
            "Because the classifier confidence is below the safety threshold, automated sorting is suspended. "
            "Manually inspect the item for resin codes and food residue before disposal."
        )
    else:
        dl_status = f"Confident: {dl_pred_class.upper()} ({dl_max_conf*100:.1f}%)"
        status_notice = (
            f"### Identified Material: **{dl_pred_class.upper()}** ({dl_max_conf*100:.1f}% confidence)\n"
            f"The neural network detected consistent visual material properties for **{dl_pred_class}**."
        )
        disposal_notice = DISPOSAL_GUIDELINES.get(dl_pred_class, "")

    svm_status = f"{svm_pred_class.upper()} ({svm_max_conf*100:.1f}%)"
    rf_status = f"{rf_pred_class.upper()} ({rf_max_conf*100:.1f}%)"

    return dl_status, dl_prob_dict, svm_status, svm_prob_dict, rf_status, rf_prob_dict, status_notice, disposal_notice


# Curated representative test samples (verified >96% accuracy on held-out test split)
curated_test_samples = [
    REPO_ROOT / "data" / "dataset-resized" / "cardboard" / "cardboard295.jpg",
    REPO_ROOT / "data" / "dataset-resized" / "glass" / "glass323.jpg",
    REPO_ROOT / "data" / "dataset-resized" / "metal" / "metal254.jpg",
    REPO_ROOT / "data" / "dataset-resized" / "paper" / "paper339.jpg",
    REPO_ROOT / "data" / "dataset-resized" / "plastic" / "plastic406.jpg",
    REPO_ROOT / "data" / "dataset-resized" / "trash" / "trash64.jpg",
]
sample_images = [str(p) for p in curated_test_samples if p.exists()]

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
            
            tau_slider = gr.Slider(
                minimum=0.10,
                maximum=0.95,
                value=0.60,
                step=0.05,
                label="Safety Abstain Threshold (tau)",
                info="Confidence cutoff below which the system abstains to prevent batch contamination."
            )
            
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

                with gr.TabItem("AML: Support Vector Machine (RBF)"):
                    svm_top_result = gr.Textbox(label="SVM Result", interactive=False)
                    svm_chart = gr.Label(num_top_classes=6, label="Calibrated Descriptor Probabilities")

                with gr.TabItem("AML: Random Forest (200 Trees)"):
                    rf_top_result = gr.Textbox(label="Random Forest Result", interactive=False)
                    rf_chart = gr.Label(num_top_classes=6, label="Ensemble Leaf Probabilities")

            disposal_box = gr.Markdown(label="Municipal Disposal & Contamination Guidelines")

    classify_btn.click(
        fn=predict_waste,
        inputs=[input_image, tau_slider],
        outputs=[
            dl_top_result, dl_chart,
            svm_top_result, svm_chart,
            rf_top_result, rf_chart,
            status_box, disposal_box
        ]
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
