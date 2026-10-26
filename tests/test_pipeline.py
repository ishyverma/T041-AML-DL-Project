"""
Basic Unit & Smoke Tests for Waste Segregation Pipeline
Validates dataset loading, feature extraction dimensions, and model inference.
"""

import unittest
from pathlib import Path
import numpy as np
import torch
import joblib
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
SPLITS_DIR = REPO_ROOT / "splits"
DATA_DIR = REPO_ROOT / "data" / "dataset-resized"

from src.feature_extraction import extract_single_image_features, TARGET_SIZE


class TestWastePipeline(unittest.TestCase):

    def test_01_splits_exist(self):
        """Verify train, val, test CSV manifests and classes.json exist."""
        self.assertTrue((SPLITS_DIR / "train.csv").exists(), "train.csv missing")
        self.assertTrue((SPLITS_DIR / "val.csv").exists(), "val.csv missing")
        self.assertTrue((SPLITS_DIR / "test.csv").exists(), "test.csv missing")
        self.assertTrue((SPLITS_DIR / "classes.json").exists(), "classes.json missing")

    def test_02_feature_extraction_dimension(self):
        """Verify that extract_single_image_features returns exact 1,634-dimensional vector."""
        # Create a dummy image
        dummy_img = Image.new("RGB", (256, 256), color=(128, 128, 128))
        temp_path = REPO_ROOT / "data" / ".test_dummy.jpg"
        dummy_img.save(temp_path)

        try:
            feats = extract_single_image_features(temp_path)
            self.assertEqual(feats.shape[0], 1634, f"Expected 1634 features, got {feats.shape[0]}")
            self.assertFalse(np.isnan(feats).any(), "Extracted features contain NaN")
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_03_aml_models_load(self):
        """Verify that aml_scaler and aml_svm load and predict correctly."""
        self.assertTrue((MODELS_DIR / "aml_scaler.joblib").exists())
        self.assertTrue((MODELS_DIR / "aml_svm.joblib").exists())

        scaler = joblib.load(MODELS_DIR / "aml_scaler.joblib")
        svm = joblib.load(MODELS_DIR / "aml_svm.joblib")

        dummy_vector = np.zeros((1, 1634))
        scaled = scaler.transform(dummy_vector)
        probs = svm.predict_proba(scaled)
        self.assertEqual(probs.shape, (1, 6), f"Expected shape (1, 6), got {probs.shape}")
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=4)

    def test_04_dl_model_forward_pass(self):
        """Verify MobileNetV2 architecture and weights forward pass."""
        from torchvision.models import mobilenet_v2
        import torch.nn as nn

        self.assertTrue((MODELS_DIR / "mobilenetv2_best.pth").exists())

        model = mobilenet_v2(weights=None)
        model.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(model.last_channel, 6))
        model.load_state_dict(torch.load(MODELS_DIR / "mobilenetv2_best.pth", map_location="cpu"))
        model.eval()

        dummy_tensor = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy_tensor)
        self.assertEqual(out.shape, (1, 6), f"Expected shape (1, 6), got {out.shape}")


if __name__ == "__main__":
    unittest.main()
