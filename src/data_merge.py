"""Clean, merge and feature-engineer the customer datasets (Role A).

Run from the repo root:
    python src/data_merge.py

Reads  data/raw/customer_social_profiles.csv and data/raw/customer_transactions.csv
Writes data/processed/merged_dataset.csv
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_PATH = ROOT / "data" / "processed" / "merged_dataset.csv"

SENTIMENT_MAP = {"Positive": 1, "Neutral": 0, "Negative": -1}


def load_raw(raw_dir: Path = RAW_DIR):
    social = pd.read_csv(raw_dir / "customer_social_profiles.csv")
    trans = pd.read_csv(raw_dir / "customer_transactions.csv")
    return social, trans


def clean(social: pd.DataFrame, trans: pd.DataFrame):
    """Drop exact duplicates, make both ID columns the same integer type, parse dates."""
    social = social.drop_duplicates().copy()
    # "A178" -> 178 so it matches the integer legacy ID in transactions
    social["customer_id"] = social["customer_id_new"].str.replace("A", "", regex=False).astype(int)

    trans = trans.copy()
    trans["customer_id"] = trans["customer_id_legacy"]
    trans["purchase_date"] = pd.to_datetime(trans["purchase_date"])
    return social, trans


def aggregate_social(social: pd.DataFrame) -> pd.DataFrame:
    """Social profiles hold several rows per customer (one per platform).
    Collapse to one row per customer to avoid a many-to-many join."""
    s = social.assign(sentiment_score=social["review_sentiment"].map(SENTIMENT_MAP))
    assert s["sentiment_score"].notna().all(), "unexpected sentiment label"

    return (
        s.groupby("customer_id")
        .agg(
            n_social_profiles=("customer_id", "size"),
            avg_engagement=("engagement_score", "mean"),
            avg_purchase_interest=("purchase_interest_score", "mean"),
            avg_sentiment=("sentiment_score", "mean"),
            # on a tie, mode() returns the alphabetically first platform
            main_platform=("social_media_platform", lambda x: x.mode().iloc[0]),
        )
        .reset_index()
    )


def merge(trans: pd.DataFrame, social_agg: pd.DataFrame) -> pd.DataFrame:
    """Left join, transactions on the left: every transaction is a prediction example.
    validate='many_to_one' raises if social_agg still has repeated customer IDs."""
    merged = trans.merge(social_agg, on="customer_id", how="left", validate="many_to_one")
    assert len(merged) == len(trans), "merge changed the number of transactions"
    assert not merged["transaction_id"].duplicated().any(), "duplicated transactions after merge"
    return merged


def engineer_features(merged: pd.DataFrame) -> pd.DataFrame:
    df = merged.drop(columns=["customer_id_legacy"]).copy()

    # --- Missing values ---
    df["has_social_profile"] = df["n_social_profiles"].notna().astype(int)
    df["rating_missing"] = df["customer_rating"].isna().astype(int)
    df["customer_rating"] = df["customer_rating"].fillna(df["customer_rating"].median())
    df["n_social_profiles"] = df["n_social_profiles"].fillna(0)
    for col in ["avg_engagement", "avg_purchase_interest", "avg_sentiment"]:
        df[col] = df[col].fillna(df[col].median())
    df["main_platform"] = df["main_platform"].fillna("Unknown")

    # --- Per-customer behaviour (none of these use product_category: no target leakage) ---
    ref_date = df["purchase_date"].max() + pd.Timedelta(days=1)
    g = df.groupby("customer_id")
    df["customer_purchase_count"] = g["transaction_id"].transform("count")
    df["customer_avg_spend"] = g["purchase_amount"].transform("mean")
    df["customer_total_spend"] = g["purchase_amount"].transform("sum")
    df["recency_days"] = (ref_date - g["purchase_date"].transform("max")).dt.days
    df["amount_vs_customer_avg"] = df["purchase_amount"] / df["customer_avg_spend"]

    # --- Date features ---
    df["purchase_month"] = df["purchase_date"].dt.month
    df["purchase_dayofweek"] = df["purchase_date"].dt.day_of_week

    # --- Encode social platform ---
    return pd.get_dummies(df, columns=["main_platform"], prefix="platform", dtype=int)


def build_dataset() -> pd.DataFrame:
    social, trans = load_raw()
    social, trans = clean(social, trans)
    social_agg = aggregate_social(social)
    merged = merge(trans, social_agg)
    return engineer_features(merged)


if __name__ == "__main__":
    dataset = build_dataset()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUT_PATH, index=False)
    print(f"saved {OUT_PATH}")
    print("shape:", dataset.shape, "(expected (150, 25))")
    print("nulls left:", int(dataset.isna().sum().sum()))
