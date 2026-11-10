# Multimodal Authentication + Product Recommendation System

A user must pass **face recognition**, then **voice verification**, before the product model gives a recommendation. Failing either step prints **"Access Denied"**.

```
face image → recognized? → voice audio → verified? → product recommendation
                 │                          │
                 └── no → Access Denied ────┴── no → Access Denied
```

## Project status

| Part | Owner | Status |
|---|---|---|
| Data merge, EDA, feature engineering, product model | _Member A_ | Done |
| Image pipeline, `image_features.csv`, face model | _Member B_ | _TODO_ |
| Audio pipeline, `audio_features.csv`, voice model | _Member C_ | _TODO_ |
| CLI demo, unauthorized-attempt demo, video, report | _Member D_ | _TODO_ |

## Repository structure

```
multimodal-auth-system/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                  # original CSVs, never edited
│   └── processed/            # merged_dataset.csv, image_features.csv, audio_features.csv
├── media/
│   ├── images/               # original/, augmented/, unauthorized/
│   └── audio/                # original/, augmented/, unauthorized/
├── notebooks/
│   └── multimodal_preprocessing.ipynb   # merge validation, EDA, model comparison
├── src/
│   ├── data_merge.py             # clean, merge, feature engineering
│   ├── image_processing.py       # load, augment, extract image features
│   ├── audio_processing.py       # load, augment, extract audio features
│   ├── train_product_model.py    # product recommendation model
│   ├── train_face_model.py       # face recognition model
│   ├── train_voice_model.py      # voiceprint verification model
│   └── predict.py                # prediction interfaces used by the CLI
├── models/                   # saved .joblib models
├── app/
│   └── main.py               # CLI demo
├── reports/
│   ├── report.pdf
│   └── figures/              # plots used in the report
└── docs/
    └── contributions.md
```

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**XGBoost needs the OpenMP runtime.**
- macOS: `brew install libomp`
- macOS too old for Homebrew (e.g. 10.15): `conda install -c conda-forge llvm-openmp`, then
  `export DYLD_FALLBACK_LIBRARY_PATH="$CONDA_PREFIX/lib:$DYLD_FALLBACK_LIBRARY_PATH"`
- Windows / Linux: `pip install xgboost` is usually enough.

If XGBoost cannot load, `train_product_model.py` skips it and falls back to the best scikit-learn model.

## How to run

Run everything from the repository root.

```bash
# 1. Merge and feature-engineer the tabular data
python src/data_merge.py            # writes data/processed/merged_dataset.csv

# 2. Train the product model
python src/train_product_model.py   # writes models/product_model.joblib
                                    # and reports/product_model_metrics.csv

# 3. Image and audio pipelines, face and voice models   (TODO: add commands)

# 4. CLI demo                                            (TODO: add command)
python app/main.py
```

The notebook (`notebooks/multimodal_preprocessing.ipynb`) shows the exploration and written interpretations behind the scripts. It runs in Colab or locally.

## Interfaces between components

| Function | Returns | Used for |
|---|---|---|
| `predict_face(image_path)` | `(member_name or "unknown", confidence)` | Step 1: face recognition |
| `predict_voice(audio_path)` | `(member_name or "unknown", confidence)` | Step 2: voice verification |
| `predict_product(customer_id)` | `(product_category, confidence)` | Step 3: recommendation |

Face and voice use a confidence threshold, so strangers are rejected instead of being forced onto the nearest member. The product model returns low confidence by design (see results below), so no threshold is applied to it.

> _TODO (Members B, C, D): confirm the final function names and the member-to-`customer_id` mapping used by the CLI._

## Data and method (product model)

**Inputs:** `customer_social_profiles.csv` (155 rows) and `customer_transactions.csv` (150 rows).

1. **Cleaning.** Dropped 5 exact duplicate social rows; converted the social ID (`A178`) and the transaction ID (`151`) to a common integer `customer_id`; parsed `purchase_date`.
2. **Aggregation.** Social profiles hold several rows per customer (one per platform), so they were collapsed to one row per customer (mean scores, encoded sentiment, most frequent platform) to avoid a many-to-many join.
3. **Join.** Left join with transactions on the left, `validate="many_to_one"`. All 150 transactions are kept; 117 (78%) matched a social profile and 33 (22%) did not. An inner join would have discarded those 33 rows.
4. **Features.** `has_social_profile` and `rating_missing` flags, median imputation, per-customer purchase count / average spend / total spend / recency, date features, one-hot platform. Final dataset: 150 rows × 25 columns, no nulls.
5. **Evaluation.** Customer-grouped 5-fold cross-validation, so one customer's transactions never appear in both training and test data.

## Results (product model)

Customer-grouped 5-fold CV, produced by `src/train_product_model.py` and saved in `reports/product_model_metrics.csv`:

| Model | Accuracy | Macro-F1 | Log-loss |
|---|---|---|---|
| Baseline (class priors) | 0.193 | 0.102 | 1.637 |
| Logistic Regression (C=0.05) | 0.213 | 0.179 | 1.729 |
| Random Forest (shallow) | 0.247 | 0.207 | 1.648 |
| HistGradientBoosting (regularized) | 0.233 | 0.213 | 1.805 |
| **XGBoost (regularized), selected** | 0.240 | 0.222 | 1.665 |

**Product category is not reliably predictable from the available features.** All models perform close to the class-prior baseline (about 20% accuracy for 5 classes), consistent with the weak correlations found in the EDA. With 150 rows the standard error on accuracy is about ±3.5 points, so differences between models are within noise. This is reported as an honest result rather than tuned further. Exact scores vary slightly between environments.

> _TODO: add face and voice model results (accuracy, F1, log-loss) once trained._

## Demo

> _TODO (Member D): CLI usage, one successful transaction, one unauthorized attempt (non-member face and voice), and the video link._

- Video: _link_
- Report: `reports/report.pdf`

## Team contributions

| Member | Tasks completed | Main files |
|---|---|---|
| _Member A_ | Data cleaning, merge and validation, EDA, feature engineering, product model | `src/data_merge.py`, `src/train_product_model.py`, `notebooks/multimodal_preprocessing.ipynb`, `data/processed/merged_dataset.csv` |
| _Member B_ | _TODO_ | _TODO_ |
| _Member C_ | _TODO_ | _TODO_ |
| _Member D_ | _TODO_ | _TODO_ |

Every member also submitted their own 3 face photos (neutral, smiling, surprised) and 2 voice recordings ("Yes, approve", "Confirm transaction"). See `docs/contributions.md` for details.
