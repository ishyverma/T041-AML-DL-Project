"""
Step 2b: Train Applied ML (AML) Baseline Models (SVM & Random Forest)
Trains classifiers on extracted [Color + HOG + LBP] feature representations.
"""

import json
from pathlib import Path
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = REPO_ROOT / "data" / "features_aml.npz"
MODELS_DIR = REPO_ROOT / "models"
SPLITS_DIR = REPO_ROOT / "splits"


def train_aml():
    print("=" * 60)
    print("STEP 2b: Training AML Baseline (SVM & Random Forest)")
    print("=" * 60)

    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"Feature cache {CACHE_PATH} not found. Run src/feature_extraction.py first.")

    # 1. Load cached features
    print(f"Loading cached features from {CACHE_PATH}...")
    data = np.load(CACHE_PATH)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]

    with open(SPLITS_DIR / "classes.json", "r") as f:
        meta = json.load(f)
        class_names = meta["class_names"]

    print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)} | Feature dims: {X_train.shape[1]}")

    # 2. Fit Standard Scaler
    print("\nFitting StandardScaler on X_train...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, MODELS_DIR / "aml_scaler.joblib")

    # 3. Train Support Vector Classifier (SVM) with RBF kernel
    print("\nTraining SVM (RBF kernel, balanced weights, probability=True)...")
    svm_clf = SVC(
        C=10.0,
        kernel="rbf",
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=42
    )
    svm_clf.fit(X_train_scaled, y_train)
    val_preds_svm = svm_clf.predict(X_val_scaled)
    svm_acc = accuracy_score(y_val, val_preds_svm)
    svm_f1 = f1_score(y_val, val_preds_svm, average="macro")

    print(f"SVM Validation Accuracy: {svm_acc * 100:.2f}% | Macro F1: {svm_f1:.4f}")
    joblib.dump(svm_clf, MODELS_DIR / "aml_svm.joblib")

    # 4. Train Random Forest Classifier
    print("\nTraining Random Forest (200 trees, balanced weights)...")
    rf_clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=25,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf_clf.fit(X_train_scaled, y_train)
    val_preds_rf = rf_clf.predict(X_val_scaled)
    rf_acc = accuracy_score(y_val, val_preds_rf)
    rf_f1 = f1_score(y_val, val_preds_rf, average="macro")

    print(f"Random Forest Validation Accuracy: {rf_acc * 100:.2f}% | Macro F1: {rf_f1:.4f}")
    joblib.dump(rf_clf, MODELS_DIR / "aml_rf.joblib")

    # 5. Save validation metrics
    metrics = {
        "svm": {
            "val_accuracy": float(svm_acc),
            "val_macro_f1": float(svm_f1),
            "classification_report": classification_report(y_val, val_preds_svm, target_names=class_names, output_dict=True)
        },
        "random_forest": {
            "val_accuracy": float(rf_acc),
            "val_macro_f1": float(rf_f1),
            "classification_report": classification_report(y_val, val_preds_rf, target_names=class_names, output_dict=True)
        }
    }

    metrics_path = MODELS_DIR / "aml_val_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("Validation Comparison (AML Baseline):")
    print(f"SVM:           Accuracy = {svm_acc*100:.2f}%, Macro F1 = {svm_f1:.4f}")
    print(f"Random Forest: Accuracy = {rf_acc*100:.2f}%, Macro F1 = {rf_f1:.4f}")
    print("=" * 60)
    print(f"Models saved in: {MODELS_DIR.resolve()}\n")


if __name__ == "__main__":
    train_aml()
