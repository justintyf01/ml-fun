# CLAUDE.md — HDB Price Predictor

## Project Summary

A production-grade Singapore HDB resale price prediction system using a 5-fold XGBoost ensemble served via FastAPI. The project evolved from a baseline Random Forest (RMSE ~$55,116) to a tuned XGBoost ensemble (RMSE ~$33–35k, R² > 0.975).

## Architecture

```
Training pipeline → export_models.py → model_artifacts/ → app/ (FastAPI)
```

### Folder Structure

```
HDB-price-predictor/
├── approaches/
│   ├── 01-baseline/         # regression.ipynb (initial RF exploration)
│   ├── 02-gradient-boosting/ # v1.x feature engineering; train_v2.py, tune, tests
│   └── 03-v3-production/    # Phase 4 features; train_v3.py, export_models.py
├── app/                     # FastAPI (untouched)
├── data/                    # Raw data + caches
│   ├── hdb_prices_2017.csv
│   ├── HDB_Resale_Prices.csv
│   ├── school_data.csv
│   ├── MRT Stations.csv
│   ├── Hawker Centres/
│   └── caches/              # onemap, school, hawker, mall, park JSON caches
├── model_artifacts/         # Trained models + inference artifacts (untouched)
├── scripts/                 # One-time cache building scripts
├── features_v2.py           # Shared feature pipeline (root)
├── features_v3.py           # Shared feature pipeline v3 (root, extends v2)
├── models.py                # Shared model definitions
└── utils.py                 # Haversine distance, OneMap helpers
```

### Shared Core (root — don't move these)

- **`features_v2.py`** — Full feature pipeline: geocoding lookups, distance calculations, target encoding, OHE, K-fold splits. Default paths now point to `data/` and `data/caches/`.
- **`features_v3.py`** — Imports from `features_v2`; adds Phase 4 features (lat/lon, interactions, storey bands, volume indicators)
- **`models.py`** — Model definitions: RF, XGBoost, LightGBM, CatBoost
- **`utils.py`** — Haversine distance, OneMap API helpers

### Data Flow

**Training**:
1. Load `data/hdb_prices_2017.csv` (227,425 rows)
2. `features_v3.py` engineers ~30 features using geocoded data from `data/caches/`
3. 5-fold CV: target encoding fitted on fold-train only (prevents leakage)
4. `approaches/03-v3-production/export_models.py` saves everything to `model_artifacts/`

**Inference**:
1. API receives `PredictionRequest`
2. `app/inference.py` engineers features from scratch (same logic as training)
3. Looks up lat/lon from `model_artifacts/caches/onemap_cache.json`
4. Applies per-fold target encodings from `model_artifacts/fold_i/target_encodings.json`
5. Runs each of 5 XGBoost models, returns mean + low/high range

## Running Scripts

All scripts are designed to run **from the project root**:

```bash
# Training (current)
python approaches/03-v3-production/train_v3.py

# Export for API
python approaches/03-v3-production/export_models.py

# Cache building (one-time)
python scripts/build_cache_v2.py
python scripts/build_school_cache.py
python scripts/build_amenity_caches.py

# API
uvicorn app.main:app --reload
```

## Key Design Decisions

- **Regional hub distances** (`dist_to_nearest_hub`) over all 171 MRT stations — using all 171 destroyed the regional signal; 4 strategic hubs provide better centrality proxy
- **Smoothed target encoding** for `town` (min_samples=30) and `street_name` (min_samples=50) — reduces overfitting on rare categories
- **Per-fold target encoding** — maps fitted on fold-train only, then stored per fold for inference; prevents leakage
- **XGBoost-only inference** — only XGBoost exported; best standalone performance, simplest serving path

## Model Artifacts Layout

```
model_artifacts/
├── fold_0/ … fold_4/
│   ├── xgboost.json            # Trained XGBoost model
│   └── target_encodings.json   # town/street encoding maps for this fold
├── ohe_columns.json
├── aggregate_lookups.json
├── reference_data.json
└── caches/                     # Copied from data/caches/ by export_models.py
```

## Path Convention

- `features_v2.py` defaults use `data/` prefix (e.g., `data/hdb_prices_2017.csv`)
- Scripts in `approaches/` and `scripts/` use `_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` for absolute paths
- `app/` uses `model_artifacts/` relative to process working directory
