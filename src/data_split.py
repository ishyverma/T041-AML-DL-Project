"""
Step 1: Data Deduplication & Stratified Train/Val/Test Split
Scans `data/dataset-resized`, performs perceptual hash and exact duplicate checks,
and writes locked stratified split CSVs to `splits/`.
"""

import os
import json
import hashlib
from pathlib import Path
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "dataset-resized"
SPLITS_DIR = REPO_ROOT / "splits"

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
LABEL_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

def compute_md5(file_path: Path) -> str:
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def compute_dhash(file_path: Path, hash_size: int = 8) -> str:
    """Compute difference hash using PIL (pure Python fallback if imagehash is absent)."""
    try:
        import imagehash
        with Image.open(file_path) as img:
            return str(imagehash.dhash(img))
    except Exception:
        # Pure PIL fallback for dhash
        with Image.open(file_path) as img:
            img = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
            pixels = list(img.getdata())
            diff = []
            for row in range(hash_size):
                for col in range(hash_size):
                    left = pixels[row * (hash_size + 1) + col]
                    right = pixels[row * (hash_size + 1) + col + 1]
                    diff.append(left > right)
            decimal_value = 0
            hex_string = []
            for index, value in enumerate(diff):
                if value:
                    decimal_value += 2 ** (index % 8)
                if (index % 8) == 7:
                    hex_string.append(hex(decimal_value)[2:].rjust(2, "0"))
                    decimal_value = 0
            return "".join(hex_string)

def run_split():
    print("=" * 60)
    print("STEP 1: Deduplication & Stratified Splitting")
    print("=" * 60)

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory {DATA_DIR} does not exist. Run inspect_dataset.py first.")

    # 1. Collect all images
    records = []
    print(f"Scanning images in {DATA_DIR}...")
    for class_name in CLASS_NAMES:
        class_folder = DATA_DIR / class_name
        if not class_folder.exists():
            continue
        for img_path in sorted(class_folder.glob("*.jpg")):
            records.append({
                "filepath": str(img_path.relative_to(REPO_ROOT)),
                "filename": img_path.name,
                "class_name": class_name,
                "label": LABEL_MAP[class_name]
            })

    df = pd.DataFrame(records)
    print(f"Discovered {len(df)} total images across {len(CLASS_NAMES)} classes.")

    # 2. Check for exact & perceptual duplicates
    print("\nRunning duplicate checks...")
    df["md5"] = df["filepath"].apply(lambda p: compute_md5(REPO_ROOT / p))
    df["dhash"] = df["filepath"].apply(lambda p: compute_dhash(REPO_ROOT / p))

    exact_dups = df[df.duplicated(subset=["md5"], keep=False)]
    if len(exact_dups) > 0:
        print(f"⚠️ Found {len(exact_dups)} exact duplicate entries! Deduplicating...")
        df = df.drop_duplicates(subset=["md5"], keep="first").reset_index(drop=True)
    else:
        print("✅ Zero exact MD5 duplicate files detected.")

    near_dups = df[df.duplicated(subset=["dhash"], keep=False)]
    if len(near_dups) > 0:
        print(f"⚠️ Found {len(near_dups)} near-identical perceptual duplicate images.")
        # We retain first instance to prevent train/test leakage
        df = df.drop_duplicates(subset=["dhash"], keep="first").reset_index(drop=True)
        print(f"Cleaned dataset count: {len(df)} unique images.")
    else:
        print("✅ Zero perceptual duplicate images detected.")

    # 3. Stratified 70% Train, 15% Val, 15% Test Split
    print("\nGenerating Stratified 70/15/15 Split (random_state=42)...")
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # 4. Save to splits directory
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(SPLITS_DIR / "train.csv", index=False)
    val_df.to_csv(SPLITS_DIR / "val.csv", index=False)
    test_df.to_csv(SPLITS_DIR / "test.csv", index=False)

    with open(SPLITS_DIR / "classes.json", "w") as f:
        json.dump({"class_names": CLASS_NAMES, "label_map": LABEL_MAP}, f, indent=2)

    # 5. Display Split Distribution
    print("\n" + "=" * 60)
    print(f"{'Class':<12} | {'Train (70%)':<12} | {'Val (15%)':<12} | {'Test (15%)':<12} | {'Total'}")
    print("-" * 60)
    for c in CLASS_NAMES:
        n_tr = len(train_df[train_df["class_name"] == c])
        n_va = len(val_df[val_df["class_name"] == c])
        n_te = len(test_df[test_df["class_name"] == c])
        print(f"{c:<12} | {n_tr:<12} | {n_va:<12} | {n_te:<12} | {n_tr + n_va + n_te}")
    print("-" * 60)
    print(f"{'TOTAL':<12} | {len(train_df):<12} | {len(val_df):<12} | {len(test_df):<12} | {len(df)}")
    print("=" * 60)
    print(f"✅ Split manifests written to: {SPLITS_DIR.resolve()}\n")

if __name__ == "__main__":
    run_split()
