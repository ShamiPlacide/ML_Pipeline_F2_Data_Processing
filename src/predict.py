"""Prediction interfaces used by the CLI.

Every function returns (label, confidence):
    predict_face(image_path)        -> (member_name or "unknown", confidence in [0, 1])
    predict_voice(audio_path)       -> (member_name or "unknown", confidence in [0, 1])
    predict_product(customer_id)    -> (product_category, confidence in [0, 1])

Until Member B / Member C deliver their models, face and voice run as STUBS that read the member
name from the file name (alice_neutral.jpg -> alice, alice_yes_approve.wav -> alice, anything
else or any path containing 'unauthorized' -> unknown). A stub is used automatically while the
model file is missing. Once models/face_model.joblib / models/voice_model.joblib exist, the CLI
switches to the real function, so B and C only fill in _real_predict_face / _real_predict_voice.
"""
from pathlib import Path

import joblib
import pandas as pd

from config import (DATA_PATH, FACE_MODEL_PATH, PRODUCT_MODEL_PATH, VOICE_MODEL_PATH, MEMBERS)

_product_bundle = None
_dataset = None


def stub_status():
    """Which models are stand-ins right now (the CLI prints this so a stub is never mistaken for real)."""
    return {"face": not FACE_MODEL_PATH.exists(), "voice": not VOICE_MODEL_PATH.exists()}


# ----------------------------------------------------------------------------- stubs
def _member_from_filename(path):
    p = Path(path)
    if "unauthorized" in {part.lower() for part in p.parts} or "unauthorized" in p.stem.lower():
        return None
    prefix = p.stem.split("_")[0].lower()
    return prefix if prefix in MEMBERS else None


def _stub(path):
    member = _member_from_filename(path)
    return (member, 0.95) if member else ("unknown", 0.20)


# ----------------------------------------------------------------------------- real hooks
def _real_predict_face(image_path):
    # TODO Member B: load models/face_model.joblib, run the SAME feature extraction used in
    # training on this one image (src/image_processing.py), and return (member_name, confidence).
    # Return ("unknown", confidence) when confidence is below the threshold you chose.
    raise NotImplementedError("Member B: implement _real_predict_face in src/predict.py")


def _real_predict_voice(audio_path):
    # TODO Member C: load models/voice_model.joblib, run the SAME feature extraction used in
    # training on this one audio file (src/audio_processing.py), and return (member_name, confidence).
    # Return ("unknown", confidence) when confidence is below the threshold you chose.
    raise NotImplementedError("Member C: implement _real_predict_voice in src/predict.py")


# ----------------------------------------------------------------------------- public API
def predict_face(image_path):
    return _stub(image_path) if stub_status()["face"] else _real_predict_face(image_path)


def predict_voice(audio_path):
    return _stub(audio_path) if stub_status()["voice"] else _real_predict_voice(audio_path)


def predict_product(customer_id):
    """Category and confidence for a customer's most recent transaction (product model)."""
    global _product_bundle, _dataset
    if _product_bundle is None:
        _product_bundle = joblib.load(PRODUCT_MODEL_PATH)
        _dataset = pd.read_csv(DATA_PATH, parse_dates=["purchase_date"])
    rows = _dataset[_dataset["customer_id"] == customer_id]
    if rows.empty:
        raise ValueError(f"customer_id {customer_id} not found in {DATA_PATH.name}")
    row = rows.sort_values("purchase_date").iloc[[-1]]
    proba = _product_bundle["model"].predict_proba(row[_product_bundle["feature_names"]])[0]
    label = _product_bundle["label_encoder"].inverse_transform([proba.argmax()])[0]
    return str(label), float(proba.max())
