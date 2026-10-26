"""
Smart Waste Segregation: Dataset Inspector & Loader
Supports both:
1. Local extraction from `trashnet/data/dataset-resized.zip` (Offline & instant - already in repo)
2. Hugging Face `load_dataset("garythung/trashnet")` (via `datasets` library)
"""

import os
import sys
import zipfile
import glob
from collections import Counter
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow is required. Run: pip install pillow")
    sys.exit(1)

# Paths
REPO_ROOT = Path(__file__).resolve().parent
LOCAL_ZIP_PATH = REPO_ROOT / "trashnet" / "data" / "dataset-resized.zip"
TARGET_EXTRACT_DIR = REPO_ROOT / "data" / "dataset-resized"
VISUALIZATION_OUT = REPO_ROOT / "dataset_overview.png"


def load_from_local_zip(zip_path: Path, extract_to: Path):
    """Extract and load dataset from local zip file included in the repository."""
    if not zip_path.exists():
        raise FileNotFoundError(f"Local zip not found at {zip_path}")
    
    if not extract_to.exists() or len(list(extract_to.glob("*/*.jpg"))) == 0:
        print(f"Extracting local dataset from: {zip_path.name} -> {extract_to}...")
        extract_to.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to.parent)
        print("Extraction complete!\n")
    else:
        print(f"Dataset already extracted at: {extract_to}\n")

    # Discover images per class
    class_folders = sorted([d for d in extract_to.iterdir() if d.is_dir() and not d.name.startswith(".")])
    data_by_class = {}
    for folder in class_folders:
        images = sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg")))
        data_by_class[folder.name] = images

    return data_by_class


def load_from_huggingface():
    """Load dataset from Hugging Face hub: load_dataset('garythung/trashnet')."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Notice: Hugging Face 'datasets' library is not installed.")
        print("   To use HF directly, run: pip install datasets\n")
        return None

    print("Downloading/loading dataset from Hugging Face ('garythung/trashnet')...")
    ds = load_dataset("garythung/trashnet")
    return ds


def inspect_local_data(data_by_class):
    """Print statistical breakdown and sample image properties."""
    total_images = sum(len(imgs) for imgs in data_by_class.values())
    
    print("=" * 65)
    print(" TRASHNET DATASET OVERVIEW & CLASS DISTRIBUTION")
    print("=" * 65)
    print(f"{'Class Name':<15} | {'Count':<8} | {'Percentage':<12} | {'Sample Resolution'}")
    print("-" * 65)

    sample_images_per_class = {}
    resolution_counter = Counter()

    for cls_name, img_paths in data_by_class.items():
        count = len(img_paths)
        pct = (count / total_images) * 100 if total_images > 0 else 0
        sample_res = "N/A"
        
        if count > 0:
            sample_images_per_class[cls_name] = img_paths[0]
            with Image.open(img_paths[0]) as img:
                sample_res = f"{img.width}x{img.height} ({img.mode})"
                resolution_counter[f"{img.width}x{img.height}"] += 1

        print(f"{cls_name:<15} | {count:<8} | {pct:>6.2f}%     | {sample_res}")

    print("-" * 65)
    print(f"{'TOTAL':<15} | {total_images:<8} | 100.00%")
    print("=" * 65)
    print(f"\nImage Resolutions Observed: {dict(resolution_counter)}")
    
    return sample_images_per_class


def create_sample_montage(sample_images, output_path: Path):
    """Create a composite image showing 1 sample from each class."""
    if not sample_images:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt

        classes = list(sample_images.keys())
        cols = 3
        rows = (len(classes) + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
        axes = axes.flatten()

        for idx, cls_name in enumerate(classes):
            img_path = sample_images[cls_name]
            with Image.open(img_path) as img:
                axes[idx].imshow(img)
                axes[idx].set_title(f"Class: {cls_name}\n({img.width}x{img.height})", fontsize=12, fontweight='bold')
                axes[idx].axis("off")

        # Hide any unused subplots
        for idx in range(len(classes), len(axes)):
            axes[idx].axis("off")

        plt.suptitle("TrashNet Classes - Visual Sample Montage", fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Sample montage saved to: {output_path}")
    except Exception as e:
        print(f"Notice: Could not save montage via matplotlib ({e}).")


def main():
    print("Checking dataset availability...\n")
    
    # 1. Check local archive
    if LOCAL_ZIP_PATH.exists():
        print(f"Found local dataset archive: {LOCAL_ZIP_PATH}")
        data_by_class = load_from_local_zip(LOCAL_ZIP_PATH, TARGET_EXTRACT_DIR)
        samples = inspect_local_data(data_by_class)
        create_sample_montage(samples, VISUALIZATION_OUT)
    else:
        print(f"Local zip not found at {LOCAL_ZIP_PATH}. Trying Hugging Face...")
        hf_ds = load_from_huggingface()
        if hf_ds:
            print(f"Successfully loaded from Hugging Face:")
            print(hf_ds)

    print("\nRecommendation on 'Should I download via Hugging Face load_dataset?'")
    print("----------------------------------------------------------------------")
    print("• You ALREADY have the full dataset locally inside 'trashnet/data/dataset-resized.zip'!")
    print("• Extracting it takes ~1 second and uses 0 internet bandwidth.")
    print("• If you prefer Hugging Face's API format, you can do: pip install datasets")
    print("  and then load it with `load_dataset('garythung/trashnet')` - it is the exact same data.")


if __name__ == "__main__":
    main()
