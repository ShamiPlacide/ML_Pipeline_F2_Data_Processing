"""
face_predict.py -- face recognition for the CLI demo

predict_face(image_path) -> (label, confidence)
    label       member name, or "unknown" if confidence is below the threshold
    confidence  the model's top probability (0 to 1)

Put this file in src/ (or copy the function into predict.py).
It expects models/face_model.joblib, made by the notebook.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from skimage.feature import hog

# Must match the notebook settings exactly, or the features will not line up
IMG_SIZE = (128, 128)
HIST_BINS = 8

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_model.joblib"
_bundle = None  # loaded once, on first use


def _load_bundle():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def _color_histogram(img):
    arr = np.asarray(img)
    feats = []
    for c in range(3):
        counts, _ = np.histogram(arr[..., c], bins=HIST_BINS, range=(0, 256))
        feats.append(counts / counts.sum())
    return np.concatenate(feats)


def _hog_features(img):
    gray = np.asarray(ImageOps.grayscale(img))
    return hog(gray, orientations=9, pixels_per_cell=(32, 32),
               cells_per_block=(2, 2), feature_vector=True)


def _extract_features(img):
    """Same steps as the notebook's extract_features."""
    img = img.resize(IMG_SIZE)
    gray = np.asarray(ImageOps.grayscale(img), dtype=float)
    feats = {"mean_intensity": gray.mean(), "std_intensity": gray.std()}
    for i, v in enumerate(_color_histogram(img)):
        feats[f"hist_{i}"] = v
    for i, v in enumerate(_hog_features(img)):
        feats[f"hog_{i}"] = v
    return feats


def predict_face(image_path):
    bundle = _load_bundle()
    img = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")

    feats = _extract_features(img)
    # Put the features in the same column order the model was trained on
    row = pd.DataFrame([feats])[bundle["feature_cols"]]

    probs = bundle["model"].predict_proba(row)[0]
    best = int(np.argmax(probs))
    label = bundle["classes"][best]
    confidence = float(probs[best])

    if confidence < bundle["threshold"]:
        return "unknown", confidence
    return label, confidence


if __name__ == "__main__":
    import sys
    name, conf = predict_face(sys.argv[1])
    print(f"{name} (confidence {conf:.3f})")
