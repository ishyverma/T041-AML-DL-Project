"""
PyTorch Dataset and DataLoader with Conservative Augmentations
"""

from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = REPO_ROOT / "splits"

# Standard ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transforms():
    """Conservative data augmentations as requested in the problem statement."""
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return train_transform, val_transform


class WasteDataset(Dataset):
    def __init__(self, csv_file: Path, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = REPO_ROOT / row["filepath"]
        label = int(row["label"])

        with Image.open(img_path) as img:
            image = img.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label, str(img_path)


def get_dataloaders(batch_size=32, num_workers=2):
    train_tf, val_tf = get_transforms()

    train_ds = WasteDataset(SPLITS_DIR / "train.csv", transform=train_tf)
    val_ds = WasteDataset(SPLITS_DIR / "val.csv", transform=val_tf)
    test_ds = WasteDataset(SPLITS_DIR / "test.csv", transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
