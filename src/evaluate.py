"""
Step 4: Fair Comparative Benchmark, Calibration, Failure Analysis & Abstention
Runs on the held-out test set (`splits/test.csv`) to compare AML vs. Deep Learning.
Generates:
1. Macro F1, Per-class Recall/Precision, Overall Accuracy.
2. Side-by-side Confusion Matrices -> `reports/confusion_matrices.png`.
3. Reliability Diagrams & Expected Calibration Error (ECE) -> `reports/calibration_curve.png`.
4. Qualitative Failure Mode Montage -> `reports/failure_cases_aml_vs_dl.png`.
5. Low-Confidence Abstention analysis.
"""

import sys
import json
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.feature_extraction import extract_single_image_features

SPLITS_DIR = REPO_ROOT / "splits"
MODELS_DIR = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "reports"
CACHE_PATH = REPO_ROOT / "data" / "features_aml.npz"


def compute_ece(probs, targets, n_bins=10):
    """Compute Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == targets

    ece = 0.0
    bin_confs = []
    bin_accs = []
    bin_counts = []

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        bin_counts.append(np.sum(in_bin))

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            bin_accs.append(accuracy_in_bin)
            bin_confs.append(avg_confidence_in_bin)
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.0)

    return ece, bin_confs, bin_accs, bin_boundaries


def load_dl_model(checkpoint_path: Path, device: torch.device):
    model = mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.last_channel, 6)
    )
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def run_benchmark():
    print("=" * 65)
    print("STEP 4: Fair Comparison, Calibration, Failure Analysis & Abstention")
    print("=" * 65)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SPLITS_DIR / "classes.json", "r") as f:
        meta = json.load(f)
        class_names = meta["class_names"]

    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    y_test = test_df["label"].values

    # 1. Load AML Models and Predict
    print("\n[1/5] Evaluating AML Baseline (SVM with RBF kernel)...")
    scaler = joblib.load(MODELS_DIR / "aml_scaler.joblib")
    svm_clf = joblib.load(MODELS_DIR / "aml_svm.joblib")
    rf_clf = joblib.load(MODELS_DIR / "aml_rf.joblib")

    # Load cached test features
    feat_data = np.load(CACHE_PATH)
    X_test_scaled = scaler.transform(feat_data["X_test"])

    aml_svm_probs = svm_clf.predict_proba(X_test_scaled)
    aml_svm_preds = np.argmax(aml_svm_probs, axis=1)

    aml_rf_probs = rf_clf.predict_proba(X_test_scaled)
    aml_rf_preds = np.argmax(aml_rf_probs, axis=1)

    # 2. Load Deep Learning (MobileNetV2) and Predict
    print("\n[2/5] Evaluating Deep Learning (MobileNetV2 Transfer Learning)...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dl_model = load_dl_model(MODELS_DIR / "mobilenetv2_best.pth", device)

    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dl_probs_list = []
    with torch.no_grad():
        for _, row in test_df.iterrows():
            img_path = REPO_ROOT / row["filepath"]
            with Image.open(img_path) as img:
                tensor = val_tf(img.convert("RGB")).unsqueeze(0).to(device)
            logits = dl_model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            dl_probs_list.append(probs)

    dl_probs = np.array(dl_probs_list)
    dl_preds = np.argmax(dl_probs, axis=1)

    # 3. Compute Metrics
    print("\n[3/5] Computing Quantitative Metrics on Unseen Test Split (378 items)...")
    def get_summary(y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
        return acc, f1, p, r, f

    acc_svm, f1_svm, p_svm, r_svm, _ = get_summary(y_test, aml_svm_preds)
    acc_rf, f1_rf, p_rf, r_rf, _ = get_summary(y_test, aml_rf_preds)
    acc_dl, f1_dl, p_dl, r_dl, _ = get_summary(y_test, dl_preds)

    print("\n" + "=" * 65)
    print(f"{'Model Architecture':<24} | {'Accuracy':<10} | {'Macro F1':<10} | {'ECE (Calibration)'}")
    print("-" * 65)
    ece_svm, _, _, _ = compute_ece(aml_svm_probs, y_test)
    ece_rf, _, _, _ = compute_ece(aml_rf_probs, y_test)
    ece_dl, bin_confs, bin_accs, _ = compute_ece(dl_probs, y_test)

    print(f"{'AML: Random Forest':<24} | {acc_rf*100:>6.2f}%    | {f1_rf:>8.4f}   | {ece_rf:>8.4f}")
    print(f"{'AML: SVM (RBF)':<24} | {acc_svm*100:>6.2f}%    | {f1_svm:>8.4f}   | {ece_svm:>8.4f}")
    print(f"{'DL: MobileNetV2':<24} | {acc_dl*100:>6.2f}%    | {f1_dl:>8.4f}   | {ece_dl:>8.4f}")
    print("=" * 65)

    print("\nPer-Class Recall Comparison:")
    print(f"{'Class':<12} | {'SVM Recall':<12} | {'RF Recall':<12} | {'MobileNetV2 Recall'}")
    print("-" * 65)
    for i, c in enumerate(class_names):
        print(f"{c:<12} | {r_svm[i]*100:>8.2f}%    | {r_rf[i]*100:>8.2f}%    | {r_dl[i]*100:>10.2f}%")
    print("-" * 65)

    # 4. Low-Confidence Abstention Analysis
    print("\n[4/5] Evaluating Low-Confidence Abstain Rule (threshold tau = 0.60)...")
    tau = 0.60
    svm_conf = np.max(aml_svm_probs, axis=1)
    dl_conf = np.max(dl_probs, axis=1)

    svm_abstain_mask = svm_conf < tau
    dl_abstain_mask = dl_conf < tau

    svm_retained_acc = accuracy_score(y_test[~svm_abstain_mask], aml_svm_preds[~svm_abstain_mask]) if np.sum(~svm_abstain_mask) > 0 else 0
    dl_retained_acc = accuracy_score(y_test[~dl_abstain_mask], dl_preds[~dl_abstain_mask]) if np.sum(~dl_abstain_mask) > 0 else 0

    print(f"• AML (SVM) Abstain Rate: {np.mean(svm_abstain_mask)*100:.1f}% ({np.sum(svm_abstain_mask)} items). Retained Acc: {svm_retained_acc*100:.2f}% (vs {acc_svm*100:.2f}%)")
    print(f"• DL (MobileNetV2) Abstain Rate: {np.mean(dl_abstain_mask)*100:.1f}% ({np.sum(dl_abstain_mask)} items). Retained Acc: {dl_retained_acc*100:.2f}% (vs {acc_dl*100:.2f}%)")
    print("  -> Safety Impact: Filtering out low-confidence queries prevents wrong recycling disposal!")

    # 5. Side-by-Side Confusion Matrices
    print("\n[5/5] Generating Visual Artifacts & Failure Case Study...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    cm_svm = confusion_matrix(y_test, aml_svm_preds)
    cm_dl = confusion_matrix(y_test, dl_preds)

    sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title(f"AML: SVM Confusion Matrix\n(Accuracy: {acc_svm*100:.1f}%, Macro F1: {f1_svm:.3f})", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")

    sns.heatmap(cm_dl, annot=True, fmt="d", cmap="Greens", xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title(f"DL: MobileNetV2 Confusion Matrix\n(Accuracy: {acc_dl*100:.1f}%, Macro F1: {f1_dl:.3f})", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrices.png", dpi=150)
    plt.close()
    print(f"🖼️  Saved: {REPORTS_DIR / 'confusion_matrices.png'}")

    # Reliability / Calibration Diagram
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax.plot(bin_confs, bin_accs, "s-", color="forestgreen", label=f"MobileNetV2 (ECE = {ece_dl:.4f})")
    ax.set_xlabel("Confidence (Predicted Probability)", fontsize=11)
    ax.set_ylabel("Empirical Accuracy", fontsize=11)
    ax.set_title("Reliability Diagram (Calibration Assessment)", fontsize=13, fontweight='bold')
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "calibration_curve.png", dpi=150)
    plt.close()
    print(f"🖼️  Saved: {REPORTS_DIR / 'calibration_curve.png'}")

    # Qualitative Failure Analysis: Where AML failed but DL succeeded (especially deformed/textured items)
    gap_indices = np.where((aml_svm_preds != y_test) & (dl_preds == y_test))[0]
    print(f"\nDiscovered {len(gap_indices)} failure gap cases where AML failed but DL classified correctly.")

    if len(gap_indices) >= 6:
        selected_idx = gap_indices[:6]
        fig, axes = plt.subplots(2, 3, figsize=(13, 8))
        axes = axes.flatten()

        for slot, idx in enumerate(selected_idx):
            row = test_df.iloc[idx]
            img_path = REPO_ROOT / row["filepath"]
            true_label = class_names[y_test[idx]]
            pred_aml = class_names[aml_svm_preds[idx]]
            pred_dl = class_names[dl_preds[idx]]

            with Image.open(img_path) as img:
                axes[slot].imshow(img)
            axes[slot].set_title(
                f"True: {true_label.upper()}\nAML: {pred_aml} (❌) | DL: {pred_dl} (✅)\nDL Conf: {dl_conf[idx]*100:.1f}%",
                fontsize=10, fontweight='bold',
                color="darkred" if pred_aml != true_label else "black"
            )
            axes[slot].axis("off")

        plt.suptitle("Qualitative Gap Analysis: Why Fixed Descriptors (HOG/LBP) Fail on Deformations", fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "failure_cases_aml_vs_dl.png", dpi=150)
        plt.close()
        print(f"🖼️  Saved: {REPORTS_DIR / 'failure_cases_aml_vs_dl.png'}")

    # Save complete benchmark json
    summary_data = {
        "dataset_split": "test.csv (378 items, stratified, deduplicated)",
        "metrics": {
            "aml_svm": {"accuracy": float(acc_svm), "macro_f1": float(f1_svm), "ece": float(ece_svm)},
            "aml_rf": {"accuracy": float(acc_rf), "macro_f1": float(f1_rf), "ece": float(ece_rf)},
            "dl_mobilenetv2": {"accuracy": float(acc_dl), "macro_f1": float(f1_dl), "ece": float(ece_dl)}
        },
        "abstain_tau_0_60": {
            "svm_abstain_percent": float(np.mean(svm_abstain_mask) * 100),
            "dl_abstain_percent": float(np.mean(dl_abstain_mask) * 100),
            "svm_retained_accuracy": float(svm_retained_acc),
            "dl_retained_accuracy": float(dl_retained_acc)
        }
    }
    with open(REPORTS_DIR / "final_test_benchmark.json", "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"✅ Full benchmark saved: {REPORTS_DIR / 'final_test_benchmark.json'}\n")


if __name__ == "__main__":
    run_benchmark()
