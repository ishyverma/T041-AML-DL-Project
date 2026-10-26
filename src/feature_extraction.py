"""
Step 2a: Manual Image Descriptors Extraction (Color Histograms + HOG + LBP)
Extracts fixed hand-crafted features from images and caches feature matrices.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2gray
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = REPO_ROOT / "splits"
CACHE_PATH = REPO_ROOT / "data" / "features_aml.npz"

# Feature configuration
TARGET_SIZE = (128, 128)  # Consistent resized resolution for AML
HOG_ORIENTATIONS = 8
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)
LBP_RADIUS = 2
LBP_POINTS = 16  # Uniform LBP has (LBP_POINTS + 2) bins = 18 bins
COLOR_BINS = 16  # 16 bins per channel in HSV = 48 bins


def extract_single_image_features(img_path: Path) -> np.ndarray:
    """Extract fused [Color Histogram, HOG, LBP] feature vector from a single image."""
    with Image.open(img_path) as img:
        img_rgb = img.convert("RGB").resize(TARGET_SIZE, Image.Resampling.BILINEAR)
        img_np = np.array(img_rgb)

    # 1. Color Histogram in HSV color space
    img_hsv = img_rgb.convert("HSV")
    hsv_np = np.array(img_hsv)
    h_hist, _ = np.histogram(hsv_np[:, :, 0], bins=COLOR_BINS, range=(0, 255), density=True)
    s_hist, _ = np.histogram(hsv_np[:, :, 1], bins=COLOR_BINS, range=(0, 255), density=True)
    v_hist, _ = np.histogram(hsv_np[:, :, 2], bins=COLOR_BINS, range=(0, 255), density=True)
    color_features = np.concatenate([h_hist, s_hist, v_hist])  # Length: 48

    # 2. HOG (Histogram of Oriented Gradients) for shape contours
    img_gray = rgb2gray(img_np)
    hog_features = hog(
        img_gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm='L2-Hys',
        visualize=False
    )  # Length: (128/16 - 1) * (128/16 - 1) * (2*2) * 8 = 7 * 7 * 4 * 8 = 1568

    # 3. LBP (Local Binary Patterns) for surface texture
    img_gray_uint8 = (img_gray * 255).astype(np.uint8)
    lbp = local_binary_pattern(img_gray_uint8, P=LBP_POINTS, R=LBP_RADIUS, method="uniform")
    n_bins = LBP_POINTS + 2
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)  # Length: 18

    # Concatenate into fused feature vector
    fused_vector = np.concatenate([color_features, hog_features, lbp_hist])
    return fused_vector.astype(np.float32)


def extract_and_cache_features():
    print("=" * 60)
    print("STEP 2a: Extracting Fixed Image Descriptors (Color + HOG + LBP)")
    print("=" * 60)

    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "val.csv")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")

    datasets = {"train": train_df, "val": val_df, "test": test_df}
    cached_data = {}

    for split_name, df in datasets.items():
        print(f"\nProcessing {split_name} split ({len(df)} images)...")
        features_list = []
        labels_list = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {split_name}"):
            full_path = REPO_ROOT / row["filepath"]
            feat = extract_single_image_features(full_path)
            features_list.append(feat)
            labels_list.append(row["label"])

        cached_data[f"X_{split_name}"] = np.array(features_list)
        cached_data[f"y_{split_name}"] = np.array(labels_list)

    feature_dim = cached_data["X_train"].shape[1]
    print(f"\nTotal feature vector dimension: {feature_dim}")
    print(f"   • Color Histogram (HSV): {COLOR_BINS * 3} dims")
    print(f"   • HOG shape features:    {1568} dims")
    print(f"   • LBP texture features:  {LBP_POINTS + 2} dims")

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE_PATH, **cached_data)
    print(f"Features cached to: {CACHE_PATH.resolve()} ({CACHE_PATH.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    extract_and_cache_features()
