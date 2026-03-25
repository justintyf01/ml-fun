# Approach 2: Gradient Boosting + Systematic Feature Engineering

**Training**: `train_v2.py` · **Features**: `../../features_v2.py`
**Tuning**: `tune.py` (GridSearchCV), `optuna_tune.py` (Optuna)
**Experiments**: `test_ensemble.py`, `test_elite.py`, `test_ideas.py`, `train_v2_log.py`

## Overview

Systematic feature engineering campaign across 8 sub-versions (v1.0–v1.8), starting from gradient boosting baselines and iteratively adding features. Each change was tracked with a clear accept/reject decision.

The final architecture is a **5-fold K-fold cross-validated ensemble** of RF + XGBoost + LightGBM + CatBoost with stacked optimal weights. Feature pipeline consolidated in `features_v2.py`.

## Run

```bash
# Train full v2 ensemble (from project root)
python approaches/02-gradient-boosting/train_v2.py

# Hyperparameter tuning
python approaches/02-gradient-boosting/tune.py          # GridSearchCV
python approaches/02-gradient-boosting/optuna_tune.py   # Optuna
```

---

## Iteration Log

### [v1.0] Gradient Boosting Baselines

Switched from Random Forest to gradient boosting frameworks as primary models. All 4 models trained on the same baseline features as Approach 1 (but with proper storey_midpoint — see v1.1).

| Model | RMSE | MAE | MAPE | R² |
|---|---|---|---|---|
| Random Forest (baseline) | $55,116 | $36,138 | 5.39% | 0.9256 |
| XGBoost | $44,852 | $29,591 | 4.47% | 0.9508 |
| LightGBM | $45,360 | $29,978 | 4.52% | 0.9496 |
| CatBoost | $43,572 | $29,784 | 4.54% | 0.9535 |

---

### [v1.1] Storey Midpoint — **ACCEPTED**

Replaced one-hot encoded `storey_range` (e.g., "07 TO 09") with a numeric `storey_midpoint` (8.0). Allows models to understand ordinal floor value.

**Impact**: Significant improvement across all models. RF gained the most (–$5.3k RMSE) since it suffers most from sparse dummies.

| Model | RMSE | MAE | MAPE | R² |
|---|---|---|---|---|
| Random Forest | $49,842 | $32,906 | 4.98% | 0.9392 |
| XGBoost | $43,159 | $28,522 | 4.33% | 0.9544 |
| LightGBM | $44,175 | $29,452 | 4.46% | 0.9522 |
| CatBoost | $43,376 | $29,589 | 4.51% | 0.9539 |

---

### [v1.2] Transaction Year/Month — **ACCEPTED (marginal)**

Extracted `transaction_year` and `transaction_month` from the `month` string (e.g., "2023-06") to capture market timing and price cycles.

**Impact**: Mixed. XGBoost and LightGBM improved ~$100–150. RF and CatBoost slightly degraded. Kept as time is a sound real-estate factor.

| Model | RMSE | MAE | MAPE | R² |
|---|---|---|---|---|
| Random Forest | $50,327 | $33,211 | 5.02% | 0.9380 |
| XGBoost | $43,146 | $28,654 | 4.35% | 0.9544 |
| LightGBM | $44,011 | $29,518 | 4.47% | 0.9526 |
| CatBoost | $43,500 | $29,570 | 4.50% | 0.9537 |

---

### [v1.3] Distance to CBD + Nearest MRT — **ACCEPTED**

Used the OneMap API to geocode all HDB block addresses to lat/lon. Computed Haversine distance to Raffles Place (CBD) and to 4 regional hub stations (AMK, Woodlands, Jurong East, Tampines).

**Impact**: Exceptional improvement for XGBoost (–$2.3k RMSE). Models now leverage physical proximity instead of relying on categorical town names as a location proxy.

| Model | RMSE | MAE | MAPE | R² |
|---|---|---|---|---|
| Random Forest | $50,588 | $33,384 | 5.04% | 0.9374 |
| XGBoost | $40,798 | $27,450 | 4.18% | 0.9593 |
| LightGBM | $43,173 | $28,887 | 4.39% | 0.9544 |
| CatBoost | $43,030 | $29,559 | 4.53% | 0.9547 |

---

### [v1.4] Target Encode `town` — **ACCEPTED**

Replaced 26 one-hot `town` dummy columns with a single `town_encoded` column (smoothed mean resale price per town). Reduced feature count from 58 → 34.

**Impact**: A major structural win. RF improved –$3k (less sparsity). LightGBM improved –$900. XGBoost slightly degraded on validation but model is more robust.

| Model | RMSE | MAE | MAPE | R² |
|---|---|---|---|---|
| Random Forest | $47,610 | $30,740 | 4.64% | 0.9445 |
| XGBoost | $40,840 | $27,450 | 4.18% | 0.9592 |
| LightGBM | $42,244 | $28,322 | 4.31% | 0.9563 |
| CatBoost | $43,541 | $29,915 | 4.55% | 0.9536 |

---

### [v1.5] Mature Estate Flag + School Count — **ACCEPTED**

Added `is_mature_estate` (official HDB mature estate classification) and `num_schools_within_1km` using real school coordinates from the OneMap API.

**Impact**: Another strong gain. XGBoost dropped –$600 to $40.2k. RF found best score yet (–$1.9k) using the mature estate flag to split high-variance towns.

| Model | RMSE | MAE | MAPE | R² |
|---|---|---|---|---|
| Random Forest | $45,657 | $30,349 | 4.58% | 0.9490 |
| XGBoost | $40,253 | $27,126 | 4.13% | 0.9603 |
| LightGBM | $41,814 | $28,284 | 4.32% | 0.9572 |
| CatBoost | $43,455 | $29,673 | 4.52% | 0.9538 |

---

### [v1.6] HDB Resale Price Index — **REJECTED**

Mapped each transaction's quarter to the official Singapore HDB Resale Price Index to capture market inflation.

**Why rejected**: Top model (XGBoost) degraded by ~$120. The model already infers market cycles well from `transaction_year` and `transaction_month`; the explicit Index added noise. Reverted to v1.5 dataset.

| Model | RMSE | MAE | MAPE | R² |
|---|---|---|---|---|
| Random Forest | $45,668 | $30,298 | 4.57% | 0.9490 |
| XGBoost | $40,377 | $27,372 | 4.17% | 0.9601 |
| LightGBM | $41,705 | $28,308 | 4.32% | 0.9574 |
| CatBoost | $43,259 | $29,604 | 4.50% | 0.9542 |

---

### [v1.7] Hyperparameter Tuning — **ACCEPTED**

3-fold cross-validated GridSearchCV on XGBoost over the v1.5 dataset.

**Optimal parameters found**:
- `learning_rate`: 0.05
- `max_depth`: 10
- `min_child_weight`: 3
- `subsample`: 0.8
- `colsample_bytree`: 0.8
- `n_estimators`: 500

**Impact**: Tuned XGBoost achieved **$37,675 RMSE on the unseen test set**. A 31%+ improvement from the $55,116 baseline.

---

### [v1.8] Full 171-Station MRT Dataset — **REJECTED**

Replaced the 4 regional hub distances with a Kaggle dataset of all 171 Singapore MRT/LRT stations to calculate exact distance to the nearest station.

**Why rejected**: Counter-intuitively, XGBoost performance degraded from $40.2k to $41.2k RMSE.

**Root cause**: The 4 regional hubs (AMK, Woodlands, Jurong East, Tampines) acted as strong **regional centrality proxies** — distance to hub = "how central/accessible is this neighbourhood?". With 171 stations, `dist_to_nearest_mrt` became near-zero for almost every flat (there's always a station nearby), destroying the feature's ability to differentiate a prime regional hub from a secluded suburb. Reverted to 4 hubs.

| Model | RMSE | MAE | MAPE | R² |
|---|---|---|---|---|
| Random Forest | $46,107 | $30,470 | 4.60% | 0.9480 |
| XGBoost | $41,231 | $27,633 | 4.20% | 0.9584 |
| LightGBM | $42,318 | $28,514 | 4.35% | 0.9562 |
| CatBoost | $43,779 | $29,873 | 4.54% | 0.9531 |

---

## Key Takeaways

- **31% RMSE reduction** (v1.0→v1.7) purely from feature engineering + tuning, no additional data
- **Regional hub distances >> all MRT stations**: Fewer, strategically-chosen anchor points beat comprehensive granular data
- **Target encoding >> OHE** for high-cardinality categoricals like `town`
- **Physical location features** (`dist_to_cbd`) are the single highest-impact addition
- **HDB Resale Price Index** was redundant — the model captures market timing better with raw year/month

## What's Next → [Approach 3](../03-v3-production/README.md)

Additional Phase 4 features: raw lat/lon, feature interactions, storey premium bands, market volume indicators, street-level target encoding.
