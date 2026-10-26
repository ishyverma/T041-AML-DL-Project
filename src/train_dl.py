"""
Step 3: Deep Learning Transfer Learning (MobileNetV2)
Uses pre-trained MobileNetV2 with frozen backbone, class-weighted cross-entropy,
and modest fine-tuning to train a lightweight classification head.
"""

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

from src.dataset_loader import get_dataloaders

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = REPO_ROOT / "splits"
MODELS_DIR = REPO_ROOT / "models"
CHECKPOINT_PATH = MODELS_DIR / "mobilenetv2_best.pth"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_class_weights(train_csv_path: Path, num_classes: int = 6):
    df = pd.read_csv(train_csv_path)
    counts = df["label"].value_counts().sort_index().values
    total = len(df)
    # Balanced weights: total / (num_classes * count)
    weights = total / (num_classes * counts.astype(np.float32))
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model, data_loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets, _ in data_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())

    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    return acc, macro_f1, all_targets, all_preds


def train_dl(epochs_head=10, epochs_finetune=5, batch_size=32):
    print("=" * 60)
    print("STEP 3: Deep Learning Transfer Learning (MobileNetV2)")
    print("=" * 60)

    device = get_device()
    print(f"Using compute device: {device}")

    # 1. Dataloaders
    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size, num_workers=2)

    # 2. Compute balanced class weights to address trash class imbalance
    class_weights = compute_class_weights(SPLITS_DIR / "train.csv", num_classes=6).to(device)
    print(f"Balanced class weights: {[round(w, 2) for w in class_weights.cpu().numpy()]}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 3. Model Architecture: Pre-trained MobileNetV2
    print("\nLoading pre-trained MobileNetV2 weights (ImageNet)...")
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

    # Freeze convolutional feature backbone
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace classifier head
    in_features = model.last_channel  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 6)
    )
    model.to(device)

    # Stage 1: Train Head Only
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_head)

    best_val_f1 = 0.0
    print(f"\n--- Stage 1: Training Classification Head ({epochs_head} epochs) ---")

    for epoch in range(1, epochs_head + 1):
        model.train()
        running_loss = 0.0

        for images, targets, _ in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        val_acc, val_f1, _, _ = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:02d}/{epochs_head} | Train Loss: {epoch_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), CHECKPOINT_PATH)

    # Stage 2: Modest Fine-Tuning (Unfreeze Top Residual Block)
    if epochs_finetune > 0:
        print(f"\n--- Stage 2: Fine-Tuning Top Layer ({epochs_finetune} epochs, lr=1e-4) ---")
        # Load best weights so far
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        
        # Unfreeze the very last inverted residual block (features[-1])
        for param in model.features[-1].parameters():
            param.requires_grad = True

        optimizer_ft = torch.optim.AdamW([
            {"params": model.features[-1].parameters(), "lr": 1e-4},
            {"params": model.classifier.parameters(), "lr": 3e-4}
        ], weight_decay=1e-4)

        for epoch in range(1, epochs_finetune + 1):
            model.train()
            running_loss = 0.0
            for images, targets, _ in train_loader:
                images, targets = images.to(device), targets.to(device)
                optimizer_ft.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer_ft.step()
                running_loss += loss.item() * images.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            val_acc, val_f1, _, _ = evaluate(model, val_loader, device)

            print(f"FT-Epoch {epoch:02d}/{epochs_finetune} | Train Loss: {epoch_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro F1: {val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), CHECKPOINT_PATH)

    # Final Validation Evaluation
    print(f"\nLoading best checkpoint from: {CHECKPOINT_PATH}")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    final_acc, final_f1, val_targets, val_preds = evaluate(model, val_loader, device)

    with open(SPLITS_DIR / "classes.json", "r") as f:
        meta = json.load(f)
        class_names = meta["class_names"]

    val_report = classification_report(val_targets, val_preds, target_names=class_names, output_dict=True)
    with open(MODELS_DIR / "dl_val_metrics.json", "w") as f:
        json.dump({"val_accuracy": final_acc, "val_macro_f1": final_f1, "report": val_report}, f, indent=2)

    print("=" * 60)
    print(f"Final MobileNetV2 Validation Results:")
    print(f"   Accuracy: {final_acc * 100:.2f}%")
    print(f"   Macro F1: {final_f1:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    train_dl(epochs_head=10, epochs_finetune=5)
