# Multimodal Authentication + Product Recommendation System

A user must pass **face recognition**, then **voice verification**, before the product model gives a recommendation. Failing either step prints **"Access Denied"**.

```
face image → recognized? → voice audio → verified? → product recommendation
                 │                          │
                 └── no → Access Denied ────┴── no → Access Denied
```

## Project status

| Part | Owner     | Status |
|---|-----------|--------|
| Data merge, EDA, feature engineering, product model | _Placide_ | Done   |
| Image pipeline, `image_features.csv`, face model | _Denyse_  | _Done_ |
| Audio pipeline, `audio_features.csv`, voice model | Nick      | Done   |
| CLI demo, unauthorized-attempt demo, video, report | _Loice_   | _Done_ |

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
│   └── audio/                # augmented/, unauthorized/
├── notebooks/
│   ├── multimodal_preprocessing.ipynb   # merge validation, EDA, model comparison
│   └── audio_pipeline.ipynb             # audio plots, augmentation, features, voice model
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

# 3. Audio pipeline and voice model
python src/audio_processing.py      # converts media/audio/*.m4a to WAV, augments, writes
                                    # data/processed/audio_features.csv and reports/figures/audio_*.png
python src/train_voice_model.py     # writes models/voice_model.joblib
                                    # and reports/voice_model_metrics.csv
python src/predict.py media/audio/Nick-approve.m4a   # quick check of predict_voice

# 3b. Image pipeline and face model                     (TODO: add commands)

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

## Data and method (voice model)

**Inputs:** two takes of each phrase ("Yes, approve", "Confirm transaction") from Denyse, Nick and Placide (12 recordings), plus one recording each from two non-members (`media/audio/Diff.m4a`, `media/audio/Unknown-2.m4a`) used as unauthorized samples.

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

## Data and method (voice model)

**Inputs:** two phone recordings per member ("Yes, approve", "Confirm transaction") for Denyse, Nick and Placide, plus one recording from a non-member (`media/audio/Diff.m4a`) used as the unauthorized sample.

1. **Cleaning.** Every recording is resampled to 16 kHz mono, trimmed of leading and trailing silence, peak-normalised and saved as `media/audio/original/<member>_<phrase>_<take>.wav` (non-members in `media/audio/unauthorized/`).
2. **Augmentation.** Six copies per clip in `media/audio/augmented/`: pitch shift (+1 / -1 semitone), time stretch (15% faster / slower), white background noise (20 dB / 10 dB SNR). 14 recordings become 98 clips.
3. **Features.** 13 MFCCs (mean and std), delta-MFCC std, spectral roll-off, centroid, bandwidth, RMS energy, zero-crossing rate, pitch (F0), voiced fraction and duration: 53 features per clip in `data/processed/audio_features.csv`. Statistics are taken over voiced frames only, so pauses and background noise do not distort them.
4. **Model.** A classifier identifies the member; the voice is only verified when its confidence reaches a threshold, otherwise `predict_voice` returns `"unknown"`.
5. **Evaluation.** Leave-one-recording-out: a recording and all its augmented copies are held out together, so the model is never tested on a recording it has heard in any form. The non-member clips are never used for training or for choosing the threshold.

Adding recordings: drop more `.m4a` files into `media/audio/` (member name first, then a phrase word, e.g. `nick_approve_3.m4a`; the number at the end is the take; anything starting with `unknown` is a non-member) and re-run the two commands.

## Results (voice model)

Leave-one-recording-out CV on 84 member clips from 12 recordings, produced by `src/train_voice_model.py` and saved in `reports/voice_model_metrics.csv`:

| Model | Features | Accuracy | Macro-F1 | Log-loss |
|---|---|---|---|---|
| Baseline (class priors) | none | 0.000 | 0.000 | 1.299 |
| **Logistic Regression (C=0.1), selected** | all 53 | 0.810 | 0.810 | 0.460 |
| Random Forest | all 53 | 0.810 | 0.806 | 0.597 |
| SVM (RBF) | all 53 | 0.786 | 0.789 | 0.522 |
| Logistic Regression (C=0.1) | 19 long-term averages | 0.738 | 0.728 | 0.554 |
| Random Forest | 19 long-term averages | 0.857 | 0.855 | 0.484 |
| SVM (RBF) | 19 long-term averages | 0.821 | 0.817 | 0.536 |

Chance is 0.33. The model with the lowest log-loss is selected; with 12 recordings the differences between models are within noise. The baseline scores 0 because holding out one of a member's four recordings makes that member the minority class in the training fold. Denyse is recognised in 26 of 28 held-out clips; most errors are between Nick and Placide, whose pitch is similar. With only one take per phrase (6 recordings) the best accuracy was 0.71, so the second takes helped every model.

With the confidence threshold (0.65, chosen on the held-out predictions; also set as `VOICE_THRESHOLD` in `src/config.py`):

| Check | Result |
|---|---|
| Held-out member clips verified under the right name | 57% |
| Held-out member clips rejected | 42% |
| Accepted under the wrong name | 1% (1 of 84) |
| The 12 member recordings (as used in the demo) | all verified, confidence 0.89 to 0.98 |
| Non-member `unknown_diff` | rejected (confidence 0.46); 4 of its 6 augmented copies also rejected |
| Non-member `unknown_2` | **accepted as Nick** (confidence 0.95), as were all its augmented copies |

**Limits.** `unknown_2` is a low voice that sits as close to Nick's recordings in feature space as Nick's own takes do; every candidate model accepts it, and an extra "unknown" class trained on the other stranger did not help. Classifier confidence says which member a voice is closest to, not how close. In the full system that person would also have to pass face recognition as Nick, because the CLI requires face and voice to agree. More non-member voices, more takes, and distance-based verification (e.g. pretrained speaker embeddings) are the next steps.

> _TODO: add face model results (accuracy, F1, log-loss) once trained._

## Demo

> _TODO (Member D): CLI usage, one successful transaction, one unauthorized attempt (non-member face and voice), and the video link._

- Video: https://youtu.be/z60imyS1AN0 
- Report: [ML_F2_C3_Group12_Final Report.pdf](https://github.com/user-attachments/files/34806888/ML_F2_C3_Group12_Final.Report.pdf)


## Team contributions
[BSE_Group_Assignments_Task_Sheet_ML_Pipeline_Formative2_Cohort_3_Team_12.xlsx](https://github.com/user-attachments/files/34806825/BSE_Group_Assignments_Task_Sheet_ML_Pipeline_Formative2_Cohort_3_Team_12.xlsx)


Every member also submitted their own 3 face photos (neutral, smiling, surprised) and 2 voice recordings ("Yes, approve", "Confirm transaction").
