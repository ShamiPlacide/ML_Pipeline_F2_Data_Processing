import joblib, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def predict_product(customer_id: int):
    """Returns (category, confidence) for a customer's most recent transaction."""
    bundle = joblib.load(ROOT / "models" / "product_model.joblib")
    df = pd.read_csv(ROOT / "data" / "processed" / "merged_dataset.csv", parse_dates=["purchase_date"])
    row = df[df["customer_id"] == customer_id].sort_values("purchase_date").iloc[[-1]]
    X = row[bundle["feature_names"]]
    proba = bundle["model"].predict_proba(X)[0]
    return bundle["label_encoder"].inverse_transform([proba.argmax()])[0], float(proba.max())
