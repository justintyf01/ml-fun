# HDB Resale Price Prediction (Regression)

A supervised machine learning regression project to predict Singapore HDB resale prices from flat attributes using a Random Forest model and one-hot encoding. 

## Overview
This notebook loads an HDB resale dataset, performs light preprocessing (notably converting remaining lease into a numeric value), one-hot encodes key categorical fields, and trains a `RandomForestRegressor` to predict `resale_price`. 

## Dataset
Housing & Development Board. (2021). Resale flat prices based on registration date from Jan-2017 onwards (2026) [Dataset]. data.gov.sg. Retrieved February 9, 2026 from https://data.gov.sg/datasets/d_8b84c4ee58e3cfc0ece0d773c8ca6abc/view

## Target
- `y = resale_price` (resale price in SGD). 

## Current features
The model trains on these 6 features (a mix of categorical + numerical): 
- Categorical (one-hot encoded with `pd.get_dummies(..., drop_first=True)`): `town`, `flattype`, `flatmodel`, `storeyrange`. 
- Numerical: `floor_area_sqm`, `remaining_lease_float` (remaining lease converted from the original `remaining_lease` string). 

Notes:
- The notebook parses `remaining_lease` (e.g., “53 years 01 month”) into `remaining_lease_float` in decimal years. 
- Columns like `block`, `street_name`, `lease_commence_date`, and `month` exist in the raw dataset but are not currently used as model features. 

## Preprocessing
- One-hot encoding: `pd.get_dummies(X, columns=categoricalfeatures, drop_first=True)`. 
- Train/validation/test split: 15% test, then a validation split from the remainder (resulting in train/val/test sets). 
- Custom single-row inference: one-hot encode the input row, then `reindex` to match the training dummy columns (missing columns filled with 0). 

## Model
- Algorithm: `RandomForestRegressor`. 
- Hyperparameters used: `n_estimators=100`, `max_depth=20`, `min_samples_split=5`, `random_state=42`, `n_jobs=-1`. 

## Current performance (validation)
Reported metrics in the notebook: 
- RMSE: 55,116.07  
- MAE: 36,138.50  
- MAPE: 5.3913995788490325%  
- R²: 0.9256  

## Future improvements
Ideas to reduce error and improve robustness/realism of price predictions:

### Add location/accessibility features
- Distance to nearest MRT station (or travel time): often a strong driver of resale value; can be computed from flat coordinates + MRT coordinates.  
- Distance to CBD / major employment centers; distance to town center.  
- Nearby amenities counts within radius (schools, malls, hawker centres, parks).  

### Add stronger time features
- Use `month` (transaction month/year) explicitly as a feature to capture price trends and market cycles (instead of letting the model treat all years as exchangeable). 

### Add richer property descriptors
- Use `lease_commence_date` (and/or flat age) directly; remaining lease already captures this partially, but “age” can interact with flat model and location. 
- Include `block`/`streetname` carefully (high-cardinality): consider target encoding or frequency encoding instead of one-hot to avoid huge sparse matrices. 

### Improve modeling approach
- Try gradient boosting models (e.g., LightGBM/XGBoost/CatBoost) which often outperform Random Forests on tabular problems with mixed feature types.
- Use cross-validation and systematic hyperparameter tuning to reduce variance and avoid overfitting to a single split.

### Production-readiness
- Replace ad-hoc `get_dummies` with a persisted preprocessing pipeline so training and inference transformations are guaranteed identical (e.g., `ColumnTransformer` + `OneHotEncoder(handle_unknown="ignore")`).
- Add input validation (category casing, storey range format, remaining lease parsing) and logging for inference.