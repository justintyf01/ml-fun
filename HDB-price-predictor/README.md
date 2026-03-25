# HDB Resale Price Predictor

Predicts Singapore HDB resale flat prices using a 5-fold XGBoost ensemble with engineered location, amenity, and market features. Served via FastAPI.

**Dataset**: 227,425 HDB resale transactions from 2017 to present (data.gov.sg)

---

## Project Progression

This project evolved from a simple notebook into a production-grade ML system through systematic feature engineering and model iteration.

| Version | Key Change | Best RMSE | Δ vs Previous |
|---|---|---|---|
| [Baseline](approaches/01-baseline/README.md) | Random Forest + OHE | $55,116 | — |
| [v1.1](approaches/02-gradient-boosting/README.md#v11-storey-midpoint--accepted) | Storey midpoint (numeric) | $43,159 | –$11,957 |
| [v1.2](approaches/02-gradient-boosting/README.md#v12-transaction-yearmonth--accepted-marginal) | Transaction year/month | $43,146 | –$13 |
| [v1.3](approaches/02-gradient-boosting/README.md#v13-distance-to-cbd--nearest-mrt--accepted) | Distance to CBD + MRT hubs | $40,798 | –$2,348 |
| [v1.4](approaches/02-gradient-boosting/README.md#v14-target-encode-town--accepted) | Target-encode `town` | $40,840 | +$42 (structural win) |
| [v1.5](approaches/02-gradient-boosting/README.md#v15-mature-estate-flag--school-count--accepted) | Mature estate + schools | $40,253 | –$545 |
| v1.6 | HDB Resale Price Index | $40,377 | +$124 ❌ rejected |
| [v1.7](approaches/02-gradient-boosting/README.md#v17-hyperparameter-tuning--accepted) | Hyperparameter tuning | **$37,675** | –$2,578 |
| v1.8 | 171-station MRT dataset | $41,231 | +$3,556 ❌ rejected |
| [v3](approaches/03-v3-production/README.md) | Phase 4: lat/lon, interactions, market volume | ~$33–35k | –$2–4k |

**Total improvement**: $55,116 → ~$33–35k RMSE — **a 36–40% reduction** through feature engineering alone.

---

## Key Findings

### What Worked
- **Physical location features** (`dist_to_cbd`) were the single highest-impact addition (+$2.3k gain in one step)
- **Numeric storey midpoint** over one-hot floor bands — models can now understand floor value ordinality
- **Target encoding for `town`** over OHE — reduced sparsity from 58 → 34 features, RF improved by $3k
- **Hyperparameter tuning** (GridSearchCV) delivered another $2.6k gain on already-tuned features

### What Didn't Work
- **HDB Resale Price Index** (v1.6) — the model already infers market cycles from raw year/month; explicit index added noise
- **171 MRT stations** (v1.8) — counter-intuitively worse than 4 regional hubs. With 171 stations, `dist_to_nearest_mrt` becomes near-zero everywhere, destroying the feature's regional centrality signal. **The 4 hubs (AMK, Woodlands, Jurong East, Tampines) are optimal anchor points** — not because they're the closest stations, but because they represent North/South/East/West centrality.

---

## Approaches

```
approaches/
├── 01-baseline/           # Random Forest + OHE — initial exploration notebook
├── 02-gradient-boosting/  # v1.0–v1.8 feature engineering campaign
└── 03-v3-production/      # Phase 4 features + production export (current)
```

---

## Project Structure

```
HDB-price-predictor/
├── approaches/
│   ├── 01-baseline/
│   │   ├── README.md
│   │   └── regression.ipynb
│   ├── 02-gradient-boosting/
│   │   ├── README.md             ← full iteration log with results tables
│   │   ├── train_v2.py           ← 5-fold ensemble training
│   │   ├── train_v2_log.py       ← log-transform target experiment
│   │   ├── tune.py               ← GridSearchCV hyperparameter search
│   │   ├── optuna_tune.py        ← Optuna hyperparameter search
│   │   ├── test_ensemble.py
│   │   ├── test_elite.py
│   │   └── test_ideas.py
│   └── 03-v3-production/
│       ├── README.md             ← feature set, architecture, results
│       ├── train_v3.py           ← production training script
│       └── export_models.py      ← export models + artifacts for API
├── app/                          ← FastAPI web application
│   ├── main.py
│   ├── inference.py
│   ├── model_loader.py
│   ├── schemas.py
│   ├── requirements.txt
│   └── Dockerfile
├── data/
│   ├── hdb_prices_2017.csv       ← training data (2017–present, 227k rows)
│   ├── HDB_Resale_Prices.csv     ← original 2024 dataset
│   ├── school_data.csv
│   ├── MRT Stations.csv
│   ├── MRT_stations.csv
│   ├── Hawker Centres/
│   └── caches/                   ← geocoded location data (built by scripts/)
│       ├── onemap_cache.json
│       ├── school_cache.json
│       ├── hawker_cache.json
│       ├── mall_cache.json
│       └── park_cache.json
├── scripts/
│   ├── build_cache.py            ← geocode all HDB addresses via OneMap
│   ├── build_cache_v2.py         ← incremental cache updater
│   ├── build_school_cache.py
│   └── build_amenity_caches.py
├── model_artifacts/              ← trained models (produced by export_models.py)
├── features_v2.py                ← shared feature pipeline (v2)
├── features_v3.py                ← shared feature pipeline (v3, extends v2)
├── models.py                     ← model definitions (RF, XGB, LGB, CatBoost)
└── utils.py                      ← Haversine distance, OneMap helpers
```

---

## Running the API

### Option 1: Local

```bash
pip install -r app/requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

API: `http://localhost:8000` · Docs: `http://localhost:8000/docs`

### Option 2: Docker

```bash
docker build -f app/Dockerfile -t hdb-predictor .
docker run -p 8000:8000 hdb-predictor
```

> **Prerequisite**: `model_artifacts/` must exist. Run `python approaches/03-v3-production/export_models.py` if it's missing.

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Server status + model count |
| `/api/options` | GET | Valid dropdown values (towns, flat types, etc.) |
| `/api/predict` | POST | Predict resale price for a flat |

**Example request** (`POST /api/predict`):
```json
{
  "town": "BISHAN",
  "flat_type": "4 ROOM",
  "flat_model": "Improved",
  "block": "123",
  "street_name": "BISHAN STREET 12",
  "storey_range": "07 TO 09",
  "floor_area_sqm": 90,
  "remaining_lease": "60 years 06 months",
  "lease_commence_date": 1990,
  "month": "2024-06"
}
```

**Example response**:
```json
{
  "predicted_price": 650000,
  "prediction_range": {"low": 638000, "high": 661000},
  "features_summary": {
    "dist_to_cbd_km": 7.2,
    "dist_to_mrt_km": 0.4,
    "remaining_lease_years": 60.5,
    "is_mature_estate": true,
    "num_schools_within_1km": 2,
    "building_age": 34
  }
}
```

---

## Training from Scratch

### 1. Build geocoded caches (one-time, ~hours for addresses)

```bash
python scripts/build_cache_v2.py        # Geocode HDB addresses via OneMap (resumable)
python scripts/build_school_cache.py    # School coordinates
python scripts/build_amenity_caches.py  # Hawker centres, malls, parks
```

### 2. Train

```bash
python approaches/03-v3-production/train_v3.py
```

### 3. Export for API

```bash
python approaches/03-v3-production/export_models.py
```

---

## Dataset

Housing & Development Board. (2021). *Resale flat prices based on registration date from Jan-2017 onwards* [Dataset]. data.gov.sg.
Retrieved February 2026 from https://data.gov.sg/datasets/d_8b84c4ee58e3cfc0ece0d773c8ca6abc/view
