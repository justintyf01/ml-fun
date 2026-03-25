# Approach 1: Baseline (Random Forest + One-Hot Encoding)

**Notebook**: `regression.ipynb`

## Overview

The initial exploration. A Random Forest model trained on 6 features with basic preprocessing — minimal feature engineering, one-hot encoding for all categoricals, no location data.

## Features Used

| Feature | Type | Notes |
|---|---|---|
| `town` | Categorical (OHE) | 26 dummy columns |
| `flat_type` | Categorical (OHE) | |
| `flat_model` | Categorical (OHE) | |
| `storey_range` | Categorical (OHE) | e.g., "07 TO 09" treated as category |
| `floor_area_sqm` | Numeric | |
| `remaining_lease_float` | Numeric | Parsed from "53 years 01 month" string |

## Model

`RandomForestRegressor`: n_estimators=100, max_depth=20, min_samples_split=5

## Results

| Model | RMSE | MAE | MAPE | R² |
|---|---|---|---|---|
| Random Forest | $55,116 | $36,138 | 5.39% | 0.9256 |

## What Was Wrong

1. **`storey_range` as OHE**: Each floor band becomes a sparse dummy; the model can't understand that higher floors are worth more
2. **`town` as OHE**: 26 binary columns for 26 towns — very sparse, poor signal/noise
3. **No location features**: No distance to CBD, MRT, or amenities
4. **No market timing**: The `month` field (transaction date) was ignored entirely

These gaps informed the v1.1–v1.8 feature engineering campaign documented in [Approach 2](../02-gradient-boosting/README.md).
