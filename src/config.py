"""Shared settings for the CLI (Role D)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "merged_dataset.csv"
FACE_MODEL_PATH = ROOT / "models" / "face_model.joblib"
VOICE_MODEL_PATH = ROOT / "models" / "voice_model.joblib"
PRODUCT_MODEL_PATH = ROOT / "models" / "product_model.joblib"

# TODO: replace with the real member labels, EXACTLY as the face and voice models output them
# (lowercase, same spelling in both models), and give each member one customer_id that exists
# in data/processed/merged_dataset.csv. The CLI checks the IDs when it starts.
MEMBERS = {
    "placide": 151,
    "denyse": 192,
    "nick": 114,
    "loice": 100,
}

# TODO: set from the threshold values chosen by Member B (face) and Member C (voice).
# A prediction below its threshold, or labelled "unknown", is rejected.
FACE_THRESHOLD = 0.60
VOICE_THRESHOLD = 0.60

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
AUDIO_EXTS = {".wav"}
