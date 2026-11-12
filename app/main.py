"""CLI demo: face -> voice -> product recommendation. Any failure prints 'Access Denied'.

Run from the repo root:
    python app/main.py                                   # asks for the two file paths
    python app/main.py --face PATH.jpg --voice PATH.wav  # one transaction, no prompts
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import warnings
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

import predict  # noqa: E402
from config import (AUDIO_EXTS, DATA_PATH, FACE_THRESHOLD, IMAGE_EXTS, MEMBERS,  # noqa: E402
                    VOICE_THRESHOLD)


def clean_path(raw):
    """Handle paths dragged into the terminal (quotes, backslash-escaped spaces)."""
    return raw.strip().strip("'\"").replace("\\ ", " ")


def ask_path(prompt):
    return clean_path(input(prompt))


def check_file(path, allowed_exts, kind):
    p = Path(path)
    if not p.is_file():
        return f"{kind} file not found: {path}"
    if p.suffix.lower() not in allowed_exts:
        return f"{kind} file must be one of {sorted(allowed_exts)}, got '{p.suffix}'"
    return None


def deny(reason):
    print(f"\n  Access Denied  ({reason})\n")
    return False


def run_transaction(face_path, voice_path):
    """Returns True only if face and voice both pass and a recommendation is produced."""
    # ---- Step 1: face
    print("\n[1/3] Face recognition")
    error = check_file(face_path, IMAGE_EXTS, "Image")
    if error:
        return deny(error)
    try:
        face_label, face_conf = predict.predict_face(face_path)
    except Exception as exc:
        return deny(f"face model error: {exc}")
    print(f"      predicted: {face_label}  (confidence {face_conf:.2f}, threshold {FACE_THRESHOLD:.2f})")
    if face_label == "unknown" or face_label not in MEMBERS or face_conf < FACE_THRESHOLD:
        return deny("face not recognized")
    print(f"      face recognized as {face_label}")

    # ---- Step 2: voice
    print("\n[2/3] Voice verification")
    error = check_file(voice_path, AUDIO_EXTS, "Audio")
    if error:
        return deny(error)
    try:
        voice_label, voice_conf = predict.predict_voice(voice_path)
    except Exception as exc:
        return deny(f"voice model error: {exc}")
    print(f"      predicted: {voice_label}  (confidence {voice_conf:.2f}, threshold {VOICE_THRESHOLD:.2f})")
    if voice_label == "unknown" or voice_label not in MEMBERS or voice_conf < VOICE_THRESHOLD:
        return deny("voice not verified")
    if voice_label != face_label:
        return deny(f"voice belongs to {voice_label}, but the face is {face_label}")
    print(f"      voice verified as {voice_label}")

    # ---- Step 3: product recommendation
    print("\n[3/3] Product recommendation")
    try:
        category, conf = predict.predict_product(MEMBERS[face_label])
    except Exception as exc:
        print(f"\n  Recommendation unavailable: {exc}\n")
        return False
    print(f"\n  Access Granted. Welcome, {face_label}.")
    print(f"  Recommended product category: {category}  (confidence {conf:.2f})\n")
    return True


def startup_checks():
    status = predict.stub_status()
    for name, is_stub in status.items():
        if is_stub:
            print(f"  NOTE: the {name} model is a STUB (models/{name}_model.joblib not found).")
    try:
        import pandas as pd
        ids = set(pd.read_csv(DATA_PATH, usecols=["customer_id"])["customer_id"])
        missing = {m: c for m, c in MEMBERS.items() if c not in ids}
        if missing:
            print(f"  WARNING: customer_ids not in the dataset (fix src/config.py): {missing}")
    except Exception as exc:
        print(f"  WARNING: could not read {DATA_PATH}: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Multimodal authentication + product recommendation")
    parser.add_argument("--face", help="path to a face image (.jpg/.png)")
    parser.add_argument("--voice", help="path to a voice recording (.wav)")
    args = parser.parse_args()

    print("=" * 60)
    print(" Multimodal Authentication + Product Recommendation")
    print("=" * 60)
    startup_checks()

    if args.face and args.voice:
        ok = run_transaction(clean_path(args.face), clean_path(args.voice))
        sys.exit(0 if ok else 1)

    while True:
        face = ask_path("\nFace image path: ")
        voice = ask_path("Voice audio path: ")
        run_transaction(face, voice)
        if input("Run another transaction? (y/n): ").strip().lower() != "y":
            break


if __name__ == "__main__":
    main()
