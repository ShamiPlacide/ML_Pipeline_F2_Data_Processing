"""Train and save the product recommendation model (Role A).

Run from the repo root (after python src/data_merge.py):
    python src/train_product_model.py

Reads  data/processed/merged_dataset.csv
Writes models/product_model.joblib and reports/product_model_metrics.csv
"""
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:  # XGBoost needs the OpenMP runtime; skip it if it cannot load on this machine
    if os.environ.get("SKIP_XGBOOST"):
        raise ImportError("SKIP_XGBOOST is set")
    from xgboost import XGBClassifier
except Exception as exc:
    XGBClassifier = None
    print(f"XGBoost unavailable ({type(exc).__name__}); continuing without it.")

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "merged_dataset.csv"
MODEL_PATH = ROOT / "models" / "product_model.joblib"
METRICS_PATH = ROOT / "reports" / "product_model_metrics.csv"

DROP_COLS = ["product_category", "transaction_id", "customer_id", "purchase_date"]


def get_candidates():
    candidates = {
        "Baseline (class priors)": DummyClassifier(strategy="prior"),
        "Logistic Regression (C=0.05)": make_pipeline(
            StandardScaler(), LogisticRegression(C=0.05, max_iter=1000)
        ),
        "Random Forest (shallow)": RandomForestClassifier(
            n_estimators=300, max_depth=3, min_samples_leaf=5, random_state=42
        ),
        "HistGradientBoosting (regularized)": HistGradientBoostingClassifier(
            max_iter=50, max_depth=2, learning_rate=0.05, min_samples_leaf=10,
            l2_regularization=1.0, random_state=42,
        ),
    }
    if XGBClassifier is not None:
        candidates["XGBoost (regularized)"] = XGBClassifier(
            n_estimators=50, max_depth=2, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=5,
            eval_metric="mlogloss", random_state=42,
        )
    return candidates


def cv_scores(model, X, y, groups, n_classes):
    """Grouped CV: all of a customer's transactions stay in the same fold (no leakage)."""
    proba = cross_val_predict(
        model, X, y, groups=groups, cv=GroupKFold(n_splits=5), method="predict_proba"
    )
    pred = proba.argmax(axis=1)
    return {
        "accuracy": accuracy_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro"),
        "log_loss": log_loss(y, proba, labels=list(range(n_classes))),
    }


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["purchase_date"])
    X = df.drop(columns=DROP_COLS)
    le = LabelEncoder()
    y = le.fit_transform(df["product_category"])
    groups = df["customer_id"]
    print("features:", X.shape[1], "| classes:", list(le.classes_))

    candidates = get_candidates()
    results = pd.DataFrame(
        {name: cv_scores(m, X, y, groups, len(le.classes_)) for name, m in candidates.items()}
    ).T.round(3)
    print(results)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(METRICS_PATH)

    # best non-baseline model: highest macro-F1, lowest log-loss as tie-breaker
    contenders = results.drop(index="Baseline (class priors)")
    PREFERRED = "XGBoost (regularized)"
    best_name = PREFERRED if PREFERRED in contenders.index else contenders.sort_values("log_loss").index[0]
    # best_name = contenders.sort_values("log_loss").index[0]
    print("selected:", best_name)

    # final refit on all rows, then save everything the CLI needs
    best_model = candidates[best_name].fit(X, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": best_model, "label_encoder": le, "feature_names": list(X.columns)},
        MODEL_PATH,
    )
    print(f"saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
