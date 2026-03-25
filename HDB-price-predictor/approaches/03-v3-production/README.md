# Approach 3: v3 Feature Pipeline + Production Export (Current)

**Training**: `train_v3.py` · **Features**: `../../features_v3.py` (extends `../../features_v2.py`)
**Export**: `export_models.py` → `../../model_artifacts/`

## Overview

Extends the v2 pipeline with Phase 4 experimental features and produces the **production-ready 5-fold XGBoost ensemble** served by the FastAPI app.

Key additions over v2:
- **Raw coordinates** (`lat`, `lon`) — lets tree models learn non-linear spatial patterns directly
- **Feature interactions** — `area × storey`, `lease × area`, `lease × storey`
- **Storey premium bands** — `is_high_floor` (≥ 20F), `is_ground_floor` (≤ 3F)
- **Market volume indicators** — `town_month_volume` (transactions in same town/month), `town_year_median_price`
- **Street-level target encoding** — `street_name_encoded` with higher smoothing (min_samples=50)

Only XGBoost is exported to production (not the full ensemble) — it has the best standalone performance and simplest serving path.

## Run

```bash
# From the project root:

# Train and evaluate all models (RF + XGB + LGB + CatBoost)
python approaches/03-v3-production/train_v3.py

# Export 5-fold XGBoost models + all artifacts for API serving
python approaches/03-v3-production/export_models.py
```

`export_models.py` must be run before starting the API.

---

## Feature Set (v3)

| Category | Features |
|---|---|
| **Property** | `floor_area_sqm`, `storey_midpoint`, `remaining_lease_float` |
| **Lease decay** | `lease_below_60`, `lease_below_40`, `building_age` |
| **Time** | `transaction_year`, `transaction_month` |
| **Location** | `lat`, `lon`, `dist_to_cbd`, `dist_to_nearest_hub`, `dist_to_actual_nearest_mrt` |
| **Schools** | `num_schools_within_1km`, `dist_to_nearest_elite_school`, `elite_school_within_1km` |
| **Amenities** | `dist_to_nearest_hawker`, `num_hawkers_within_1km`, `dist_to_nearest_mall`, `num_malls_within_2km`, `dist_to_nearest_park` |
| **Market** | `is_mature_estate`, `town_month_volume`, `town_year_median_price` |
| **Interactions** | `area_x_storey`, `lease_x_area`, `lease_x_storey` |
| **Storey bands** | `is_high_floor` (≥20), `is_ground_floor` (≤3) |
| **Encoding** | `town_encoded` (smoothed TE), `street_name_encoded` (smoothed TE), OHE for flat_type/model |

Total: ~30 numeric features after encoding.

---

## Model Architecture

**K-fold cross-validation (5 folds)**:
- 85% of data used for training (5-fold CV), 15% held out as final test set
- Per-fold: target encoding is fitted on fold-train only → no leakage
- 4 base models per fold: RF, XGBoost, LightGBM, CatBoost
- OOF predictions stacked → optimal convex weights found via SLSQP

**Production export**:
- 5 XGBoost models saved (one per fold)
- Per-fold target encoding maps serialised
- OHE column names, aggregate lookups, and reference data saved
- All geocoded caches copied to `model_artifacts/caches/`

---

## Results

> From `train_v3.py` on 5-fold CV (best per-model ensemble performance)

| Model | OOF R² | Test R² | Test RMSE (approx) |
|---|---|---|---|
| XGBoost | > 0.975 | > 0.975 | ~$34–36k |
| LightGBM | > 0.97 | > 0.97 | ~$35–37k |
| CatBoost | > 0.97 | > 0.97 | ~$35–37k |
| Stacked Ensemble | > 0.98 | > 0.98 | ~$33–35k |

*XGBoost solo performance is comparable to the stacked ensemble — hence exported alone for production.*

---

## Artifacts Produced

```
model_artifacts/
├── fold_0/ … fold_4/
│   ├── xgboost.json            # Serialized XGBoost model
│   └── target_encodings.json   # town + street_name encoding maps for this fold
├── ohe_columns.json            # OHE feature column names (for reindexing at inference)
├── aggregate_lookups.json      # town_month_volume, town_year_median_price lookup tables
├── reference_data.json         # Valid towns, flat_types, flat_models, street_names (for API)
└── caches/                     # Geocoded coordinates used at inference
    ├── onemap_cache.json
    ├── school_cache.json
    ├── hawker_cache.json
    ├── mall_cache.json
    ├── park_cache.json
    └── MRT Stations.csv
```
