"""Train and save the voiceprint verification model (Role C).

Run from the repo root (after python src/audio_processing.py):
    python src/train_voice_model.py

Reads  data/processed/audio_features.csv
Writes models/voice_model.joblib and reports/voice_model_metrics.csv

The model identifies which team member is speaking. A voice is only *verified*
when the model's confidence reaches a threshold; otherwise the speaker is
reported as "unknown" and the transaction is denied.

Two feature sets are compared for every model:
  - "all features": everything in audio_features.csv
  - "long-term averages": only the *_mean columns (average MFCCs, roll-off,
    centroid, bandwidth, energy, zero-crossing rate, pitch). Standard
    deviations, deltas and duration describe how one particular phrase unfolds
    rather than who is speaking, so they transfer poorly to a new phrase.

Evaluation is leave-one-recording-out: a recording and all of its augmented
copies are held out together, so the model is always scored on a recording it
has never heard in any form. Non-member clips are never used for training or
for choosing the threshold; they are a separate rejection test.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "audio_features.csv"
MODEL_PATH = ROOT / "models" / "voice_model.joblib"
METRICS_PATH = ROOT / "reports" / "voice_model_metrics.csv"

META_COLS = ["member", "phrase", "augmentation", "is_original", "source_audio", "audio_path"]
BASELINE = "Baseline (class priors)"


def get_candidates():
    return {
        BASELINE: DummyClassifier(strategy="prior"),
        "Logistic Regression (C=0.1)": make_pipeline(
            StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, random_state=42
        ),
        "SVM (RBF)": make_pipeline(
            StandardScaler(), SVC(C=1.0, probability=True, random_state=42)
        ),
    }


def get_feature_sets(feature_names):
    return {
        "all features": feature_names,
        "long-term averages": [c for c in feature_names if c.endswith("_mean")],
    }


def out_of_fold_proba(model, X, y, groups):
    """Probabilities for every clip from a model that never saw its recording."""
    return cross_val_predict(
        model, X, y, groups=groups, cv=LeaveOneGroupOut(), method="predict_proba"
    )


def scores(y, proba):
    pred = proba.argmax(axis=1)
    return {
        "accuracy": accuracy_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro"),
        "log_loss": log_loss(y, proba, labels=list(range(proba.shape[1]))),
    }


def choose_threshold(y, proba):
    """Confidence cut-off that best separates right from wrong identifications.

    Uses out-of-fold predictions only. For each candidate threshold: how many
    correct identifications are accepted (TPR) and how many wrong ones are
    rejected (TNR); the threshold with the best average of the two wins.
    """
    conf, correct = proba.max(axis=1), proba.argmax(axis=1) == y
    if correct.all() or not correct.any():
        return float(np.quantile(conf, 0.10))
    candidates = np.unique(np.round(conf, 3))
    balanced = [((conf[correct] >= t).mean() + (conf[~correct] < t).mean()) / 2
                for t in candidates]
    return float(candidates[int(np.argmax(balanced))])


def main():
    df = pd.read_csv(DATA_PATH)
    feature_sets = get_feature_sets([c for c in df.columns if c not in META_COLS])
    members = df[df["member"] != "unknown"].reset_index(drop=True)
    unknown = df[df["member"] == "unknown"].reset_index(drop=True)

    le = LabelEncoder()
    y = le.fit_transform(members["member"])
    groups = members["source_audio"]
    print(f"members: {list(le.classes_)} | clips: {len(members)} from "
          f"{groups.nunique()} recordings | features: "
          f"{ {k: len(v) for k, v in feature_sets.items()} }")

    # 1. Compare models on recordings they have never heard
    # (the class-prior baseline scores 0: holding out one of a member's two
    #  recordings makes that member the minority class in every fold)
    results, oof = [], {}
    for set_name, cols in feature_sets.items():
        for name, model in get_candidates().items():
            if name == BASELINE and set_name != "all features":
                continue                      # the baseline ignores the features
            oof[name, set_name] = out_of_fold_proba(model, members[cols], y, groups)
            results.append({"model": name, "features": set_name,
                            **scores(y, oof[name, set_name])})
    results = pd.DataFrame(results)

    best = results[results["model"] != BASELINE].sort_values("log_loss").iloc[0]
    best_name, best_set = best["model"], best["features"]
    feature_names = feature_sets[best_set]
    X = members[feature_names]
    results["selected"] = (results["model"] == best_name) & (results["features"] == best_set)
    print("\nLeave-one-recording-out CV:")
    print(results.round(3).to_string(index=False))

    # 2. Confidence threshold for accepting a voice, from the same held-out predictions
    proba = oof[best_name, best_set]
    threshold = choose_threshold(y, proba)
    conf, correct = proba.max(axis=1), proba.argmax(axis=1) == y
    verified = correct & (conf >= threshold)
    false_accept_member = ~correct & (conf >= threshold)

    # 3. Final model on all member clips, then the non-member rejection test
    model = get_candidates()[best_name].fit(X, y)
    train_conf = model.predict_proba(X).max(axis=1)
    summary = {
        "threshold": threshold,
        "member_verified_rate_cv": verified.mean(),
        "member_rejected_rate_cv": (conf < threshold).mean(),
        "member_wrong_identity_accepted_rate_cv": false_accept_member.mean(),
        "member_verified_rate_training_clips": (train_conf >= threshold).mean(),
    }
    print(f"\nSelected: {best_name} on {best_set} | confidence threshold: {threshold:.3f}")
    print(f"  unseen member recordings verified correctly: {verified.mean():.0%}")
    print(f"  unseen member recordings rejected:           {(conf < threshold).mean():.0%}")
    print(f"  accepted under the wrong name:               {false_accept_member.mean():.0%}")
    print(f"  enrolled (training) clips verified:          {(train_conf >= threshold).mean():.0%}")

    if len(unknown):
        unk_proba = model.predict_proba(unknown[feature_names])
        unk_conf = unk_proba.max(axis=1)
        summary["unknown_rejected_rate"] = (unk_conf < threshold).mean()
        print(f"  non-member clips rejected:                   {(unk_conf < threshold).mean():.0%} "
              f"({len(unknown)} clips from {unknown['source_audio'].nunique()} recording(s))")
        print(pd.DataFrame({
            "clip": unknown["augmentation"], "closest_member": le.classes_[unk_proba.argmax(axis=1)],
            "confidence": unk_conf.round(3), "rejected": unk_conf < threshold,
        }).to_string(index=False))

    for key, value in summary.items():
        results[key] = np.where(results["selected"], value, np.nan)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(METRICS_PATH, index=False)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "label_encoder": le, "feature_names": feature_names,
                 "threshold": threshold, "model_name": f"{best_name}, {best_set}"}, MODEL_PATH)
    print(f"\nSaved {MODEL_PATH.relative_to(ROOT)} and {METRICS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
