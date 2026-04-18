# 💳 Credit Card Fraud Detection — Project README

> **Course Assignment** | Data mining 
> **File:** `fraud_dmsam.ipynb`  
> **Dataset:** `fraud_credit_card.csv` (~555K rows, 22 columns)  
> **Target:** `is_fraud` (1 = fraud, 0 = legitimate)

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Quick Start (Colab)](#3-quick-start-colab)
4. [Quick Start (Local)](#4-quick-start-local)
5. [Colab Crash-Survival System](#5-colab-crash-survival-system)
6. [Dataset & Features](#6-dataset--features)
7. [Methodology Summary](#7-methodology-summary)
8. [Models & Tuning](#8-models--tuning)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [Ethics & Privacy](#10-ethics--privacy)
11. [Runtime Guide](#11-runtime-guide)
12. [Known Limitations](#12-known-limitations)
13. [Dependencies](#13-dependencies)

---

## 1. Project Overview

This notebook builds a complete end-to-end ML pipeline for credit card fraud detection on a severely imbalanced dataset (~0.57% fraud).

**Learning objectives covered:**
- ✅ Handle severe class imbalance (Original / UnderSampling / SMOTENC)
- ✅ Build, tune, and compare models from 6 algorithm families
- ✅ Prevent time leakage (temporal split) and identity leakage (PII removal)
- ✅ Evaluate with PR-AUC, ROC-AUC, F1, Precision, Recall, Confusion Matrix
- ✅ Threshold optimisation (F1-optimal + cost-sensitive)
- ✅ Fairness audit (gender-stratified metrics)
- ✅ Full reproducibility with `RANDOM_SEED = 42`

---

## 2. Repository Structure

```
fraud_detection/
├── fraud_dm.ipynb            ← Main notebook (this file)
├── fraud_credit_card.csv     ← Raw dataset (download separately)
├── README.md                 ← This file
├── model_comparison.csv      ← Generated: all metrics table
├── fraud_checkpoints/        ← Generated: pickled models + metadata
│   ├── meta.json             ← Param-search cache
│   ├── splits.pkl            ← Preprocessed train/test splits
│   ├── results_list.pkl      ← Full RESULTS list
│   └── <ModelName>__<Sampler>.pkl  ← One file per trained model
└── *.png                     ← Generated plots
```

---

## 3. Quick Start (Colab)

```python
# Step 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Step 2: Upload the CSV to Drive, then set paths in §0.1:
DATA_PATH = '/content/drive/MyDrive/fraud_credit_card.csv'
CKPT_DIR  = '/content/drive/MyDrive/fraud_checkpoints'

# Step 3: Run all cells
# Runtime → Run all
```

> **⚡ Disconnect recovery:** If Colab disconnects mid-run, simply click  
> **Runtime → Run all** again. Every completed model loads from checkpoint;  
> only remaining work runs.

---

## 4. Quick Start (Local)

```bash
# 1. Clone or copy notebook + CSV to same folder
# 2. Install dependencies
pip install imbalanced-learn scikit-learn pandas numpy matplotlib seaborn

# 3. Launch Jupyter
jupyter notebook fraud_dm.ipynb

# 4. In §0.1, leave DATA_PATH = 'fraud_credit_card.csv' (default)
# 5. Run all cells (Kernel → Restart & Run All)
```

---

## 5. Colab Crash-Survival System

The notebook implements a **three-layer persistence system**:

| Layer | What is saved | When |
|---|---|---|
| **`splits.pkl`** | Preprocessed X_train, X_test, subsets, preprocessor | After §1.5 (first run only) |
| **`<Model>__<Sampler>.pkl`** | Fitted pipeline + all metrics + probability scores | Immediately after each model finishes |
| **`results_list.pkl`** | Full `RESULTS` list (all completed models) | After each model finishes |
| **`meta.json`** | Best hyperparameter from each param search | After each param search |

**How it works:**

```
§0.1: User sets CKPT_DIR to a Google Drive path
§1.1: ckpt_exists('splits') → True? → skip all of §1, reload in 2 seconds
§3.0: ckpt_load('results_list') → restore completed RESULTS
§3.x: evaluate_model() checks RESULTS + disk before fitting
      → If found: print ♻️ and return immediately
      → If not:  fit, evaluate, save, append
```

**To restart from scratch** (force rerun all models):

```python
import shutil
shutil.rmtree('fraud_checkpoints')  # delete checkpoint directory
```

---

## 6. Dataset & Features

### Raw columns (22)

| Column | Type | Usage |
|---|---|---|
| `trans_date_trans_time` | datetime | → `hour`, `dow`, `month`, `is_night`, `is_weekend` |
| `cc_num` | string | **Dropped (PII)** |
| `merchant` | category | → `merchant_freq` (frequency encoding) |
| `category` | category | **Kept** (OHE) |
| `amt` | float | **Kept** (numeric) |
| `first`, `last` | string | **Dropped (PII)** |
| `gender` | category | **Kept** (OHE) — fairness audited |
| `street`, `city`, `zip` | string | **Dropped (PII)** |
| `lat`, `long` | float | → `dist_km` (Haversine), then dropped |
| `city_pop` | int | **Kept** (numeric) |
| `job` | category | → `job_freq` (frequency encoding) |
| `dob` | string | → `age` (days since DOB), then dropped |
| `trans_num` | string | **Dropped (PII)** |
| `unix_time` | int | **Dropped** (timestamp superseded by parsed features) |
| `merch_lat`, `merch_long` | float | → `dist_km`, then dropped |
| `is_fraud` | int8 | **Target** |

### Engineered features

| Feature | Formula | Why |
|---|---|---|
| `hour` | `trans_datetime.hour` | Fraud peaks 0–5 AM |
| `dow` | `trans_datetime.dayofweek` | Weekend fraud higher in some categories |
| `month` | `trans_datetime.month` | Seasonal fraud patterns |
| `is_night` | `hour < 6 or hour >= 22` | Binary flag for highest-risk window |
| `is_weekend` | `dow >= 5` | Binary flag |
| `age` | `(trans_date - dob).days / 365.25` | Age correlates with spending norms |
| `dist_km` | Haversine(cardholder coords, merchant coords) | Unusual distance = red flag |
| `merchant_freq` | `groupby(merchant).transform('count')` | High-volume merchants = lower risk |
| `job_freq` | `groupby(job).transform('count')` | Job-level frequency signal |

---

## 7. Methodology Summary

### Train / Test Split

**Strategy:** Temporal split — first 80% of rows (sorted chronologically) → train; last 20% → test.

**Rationale:** Simulates real deployment. Model is trained on past events, evaluated on future ones. Prevents time leakage. The same `cc_num` may appear in both sets — acceptable because `cc_num` is dropped before modelling.

### Preprocessing Pipeline

```
ColumnTransformer
├── StandardScaler      → all numeric features
└── OneHotEncoder       → category, gender, state
                          (handle_unknown='ignore')
```

Wrapped in `sklearn.Pipeline` so scaling is always fit on training data only.

### Imbalance Strategies

All resampling uses `imblearn.Pipeline` — sampling fires **only on training data**, never on validation or test:

| Strategy | Implementation | Notes |
|---|---|---|
| Original | `class_weight='balanced'` | No data modification |
| UnderSampling | `RandomUnderSampler` | Applied to full train set |
| SMOTENC | `SMOTENC(k_neighbors=5)` | Applied to 30% subsample (RAM safety) |

---

## 8. Models & Tuning

| Model | Param(s) searched | Grid / Range | Method |
|---|---|---|---|
| Decision Tree | `max_depth` | {5, 10, 15, None} | Holdout |
| Logistic Regression | `C` | {0.01, 0.1, 1, 10} | Holdout |
| KNN | `n_neighbors` | {5, 11, 21, 51} | Holdout (subset) |
| SVM Linear | `C=0.1` | Fixed (literature) | — |
| SVM RBF | `C=1, gamma=scale` | Fixed (literature) | — |
| MLP | `hidden_layer_sizes` | {(64,), (128,64), (128,64,32)} | Holdout |
| Random Forest | `n_estimators`, `max_depth` | {50×10, 50×15, 100×15} | Holdout |
| Bagging | `n_estimators=30` | Fixed | — |
| GradientBoosting | `lr`, `depth`, `max_iter` | `lr=0.05, d=6, ≤200 iters` | Early stopping |
| AdaBoost | `n_estimators=100` | Fixed | — |

Each model is run with all 3 imbalance strategies → **~30 total fitted pipelines**.

---

## 9. Evaluation Metrics

### Why not accuracy?

With 0.57% fraud, a model predicting "always legitimate" achieves **99.43% accuracy** — completely useless. We use:

| Metric | Why |
|---|---|
| **PR-AUC** (primary) | Summarises precision-recall trade-off across all thresholds; not affected by class imbalance |
| **ROC-AUC** | Threshold-independent discrimination; standard benchmark |
| **F1** | Harmonic mean of precision and recall at decision threshold |
| **Precision** | Among flagged transactions, how many are actually fraud? |
| **Recall** | Among all frauds, how many did we catch? |
| **Confusion Matrix** | Absolute counts of TP/FP/TN/FN |

### Threshold Selection

Two thresholds are computed:

1. **F1-optimal:** `argmax(f1_score)` over the PR curve
2. **Cost-optimal:** `argmin(C_FP·FP + C_FN·FN)` where `C_FP=1, C_FN=10`

The **cost-optimal threshold** is recommended for deployment — it minimises total business loss given that missing a fraud costs ~10× more than a false alarm.

---

## 10. Ethics & Privacy

### PII Handling

| Data | Treatment |
|---|---|
| Names (`first`, `last`) | Dropped entirely |
| Address (`street`, `city`, `zip`) | Dropped entirely |
| Card number (`cc_num`) | Dropped entirely |
| Transaction ID (`trans_num`) | Dropped entirely |
| Date of birth (`dob`) | Converted to `age` (aggregate), raw dropped |
| Coordinates (`lat`, `long`, `merch_lat`, `merch_long`) | Converted to `dist_km` (aggregate), raw dropped |

### Fairness

- `gender` is used as a model feature.
- The notebook computes **gender-stratified Recall, FPR, Precision, F1, and ROC-AUC** at both threshold settings.
- A **disparate impact check** flags if FPR ratio (max/min across groups) exceeds 1.25.
- If disparity is detected, recommended remediation: remove `gender` entirely, or apply equalized-odds post-processing (e.g., `ThresholdOptimizer` from the Fairlearn library).

### Model Governance

- Monthly retraining recommended (fraud patterns drift).
- Feature importances satisfy regulatory explainability requirements (GDPR Art. 22, CFPB adverse-action notices).
- This is an **individual academic project** — cite any external code used.

---

## 11. Runtime Guide

### Expected runtimes (Colab T4 GPU / standard RAM)

| Section | Wall time |
|---|---|
| Data load + EDA + feature engineering | ~3–5 min |
| Decision Tree (all samplers) | ~2–4 min |
| Logistic Regression (all samplers) | ~3–5 min |
| KNN (all samplers, 10% subset) | ~2–4 min |
| SVM Linear (all samplers, 10% subset) | ~3–5 min |
| SVM RBF (original only, 10% subset) | ~2–4 min |
| MLP (all samplers) | ~8–15 min |
| Random Forest (all samplers) | ~5–10 min |
| Bagging (all samplers) | ~3–6 min |
| GradientBoosting (all samplers) | ~6–12 min |
| AdaBoost (all samplers) | ~4–8 min |
| **Total (first run)** | **~45–75 min** |
| **Total (after checkpoint reload)** | **~2–5 min** |

### Speed tips

If still too slow, add these to `§0.1`:

```python
SMOTE_FRAC   = 0.20   # reduce from 0.30
KNN_SVM_FRAC = 0.07   # reduce from 0.10
CV_STRATEGY  = 'holdout'   # already default; keep this
```

For GradientBoosting, reduce `max_iter=100` in §3.6c.

---

## 12. Known Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| KNN/SVM trained on 10% subset | Slightly pessimistic metrics vs full train | Documented; subset is stratified and representative |
| SMOTE on 30% subsample | Synthetic samples drawn from partial distribution | 30% = ~133k rows; fraud class has ~750+ samples |
| No XGBoost / LightGBM | Not in sklearn; not required by rubric | `HistGradientBoosting` is comparable in speed/performance |
| `gender` binary only in data | Non-binary identities not captured | Noted; data limitation, not a modelling choice |
| Temporal split: same card in train+test | Acceptable for time-based split; cc_num dropped | Documented in §1.5 |

---

## 13. Dependencies

```
python       >= 3.9
scikit-learn >= 1.3.0
imbalanced-learn >= 0.11.0
pandas       >= 2.0.0
numpy        >= 1.24.0
matplotlib   >= 3.7.0
seaborn      >= 0.12.0
```

Install all:

```bash
pip install imbalanced-learn scikit-learn pandas numpy matplotlib seaborn
```

---

*Notebook authored for academic submission. All code is original. External library documentation cited inline where relevant.*
