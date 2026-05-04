"""
HDB Resale Price Prediction — Gradient Boosting Models
=======================================================
Compares XGBoost, LightGBM, and CatBoost against the existing
RandomForest baseline using the same data & splits.
Usage:
    pip install xgboost lightgbm catboost
    python gradient_boosting_models.py
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
import requests
import math
import time
from features import haversine_distance

# ──────────────────────────────────────────────────────────────
# 1. Load & preprocess
# ──────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(_ROOT, "data", "HDB_Resale_Prices.csv"))
# Parse remaining_lease → float
df["years"] = df["remaining_lease"].str.extract(r"(\d+) years").astype(float).fillna(0)
df["months"] = df["remaining_lease"].str.extract(r"(\d+) month").astype(float).fillna(0)
df["remaining_lease_float"] = df["years"] + (df["months"] / 12)
# ---------------------------------------------------------
# [v1.1] Feature: Storey Midpoint
# Eg. Converts "04 TO 06" into a continuous float 5.0
# ---------------------------------------------------------
df["storey_midpoint"] = df["storey_range"].str.split(" TO ", expand=True).astype(float).mean(axis=1)

# ---------------------------------------------------------
# [v1.2] Feature: Time Features
# Extracts 'transaction_year' and 'transaction_month'
# ---------------------------------------------------------
df["transaction_year"] = df["month"].str.extract(r"(\d{4})").astype(int)
df["transaction_month"] = df["month"].str.extract(r"-(\d{2})").astype(int)

# ---------------------------------------------------------
# [v1.3] Feature: Location Features (OneMap API)
# Distances to CBD and nearest MRT
# ---------------------------------------------------------
# ---------------------------------------------------------
# [v1.3] Feature: Location Features (OneMap API)
# Distances to CBD and nearest MRT
# ---------------------------------------------------------
# Note: Real coordinates are fetched using the standalone `build_cache.py` script.
df['full_address'] = df['block'] + " " + df['street_name']

import json
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
cache_file = os.path.join(_ROOT, "data", "caches", "onemap_cache.json")
address_to_coords = {}
if os.path.exists(cache_file):
    with open(cache_file, "r") as f:
        address_to_coords = json.load(f)
else:
    raise FileNotFoundError("onemap_cache.json not found! Run `build_cache.py` first.")

# Map real latitude and longitude from the cache
df['lat'] = df['full_address'].map(lambda x: address_to_coords.get(x, (1.35, 103.8))[0])
df['lon'] = df['full_address'].map(lambda x: address_to_coords.get(x, (1.35, 103.8))[1])

# CBD (Raffles Place MRT)
CBD_COORD = (1.283933, 103.851463)
df['dist_to_cbd'] = df.apply(lambda row: haversine_distance(row['lat'], row['lon'], CBD_COORD[0], CBD_COORD[1]), axis=1)

# MRTs
# [v1.8 REJECTED] - Using 171 local stations destroyed the 'Regional Centrality' 
# signal provided by the 4 major mock hubs, degrading performance by $1k. Reverting.
# try:
#     mrt_df = pd.read_csv('MRT Stations.csv')
#     mrt_stations = list(zip(mrt_df['Latitude'], mrt_df['Longitude']))
#     print(f"Loaded {len(mrt_stations)} MRT/LRT stations from Kaggle dataset for distance calculation.")
# except FileNotFoundError:
#     print("Warning: MRT Stations.csv not found. Falling back to 4 mock stations.")
#     mrt_stations = [
#         (1.369933, 103.84958),  # AMK
#         (1.436058, 103.786053), # Woodlands
#         (1.333152, 103.742286), # Jurong East
#         (1.353266, 103.945143), # Tampines
#     ]

mrt_stations = [
    (1.369933, 103.84958),  # AMK
    (1.436058, 103.786053), # Woodlands
    (1.333152, 103.742286), # Jurong East
    (1.353266, 103.945143), # Tampines
]

# Calculate absolute nearest station for every flat
df['dist_to_nearest_mrt'] = df.apply(lambda row: min([haversine_distance(row['lat'], row['lon'], m[0], m[1]) for m in mrt_stations]), axis=1)

# ---------------------------------------------------------
# [v1.5] Feature: Estate & School Additions
# is_mature_estate, num_schools_within_1km
# ---------------------------------------------------------

# Official HDB Mature Estates
mature_estates = [
    "ANG MO KIO", "BEDOK", "BISHAN", "BUKIT MERAH", "BUKIT TIMAH", 
    "CLEMENTI", "GEYLANG", "KALLANG/WHAMPOA", "MARINE PARADE", 
    "PASIR RIS", "QUEENSTOWN", "SERANGOON", "TAMPINES", "TOA PAYOH"
]
df['is_mature_estate'] = df['town'].isin(mature_estates).astype(int)

import json
import os

# Load actual school coordinates from cache
if os.path.exists(os.path.join(_ROOT, "data", "caches", "school_cache.json")):
    with open(os.path.join(_ROOT, "data", "caches", "school_cache.json"), "r") as f:
        school_coords_data = json.load(f)
        school_points = [(s["lat"], s["lon"]) for s in school_coords_data if s["lat"] is not None]
        
        # Elite Schools (GEP)
        gep_schools = ["ANGLO-CHINESE SCHOOL (PRIMARY)", "CATHOLIC HIGH SCHOOL", "HENRY PARK PRIMARY SCHOOL", "NAN HUA PRIMARY SCHOOL", "NANYANG PRIMARY SCHOOL", "RAFFLES GIRLS' PRIMARY SCHOOL", "ROSYTH SCHOOL", "ST. HILDA'S PRIMARY SCHOOL", "TAO NAN SCHOOL"]
        elite_school_points = [(s["lat"], s["lon"]) for s in school_coords_data if s["lat"] is not None and s["name"] in gep_schools]
else:
    raise FileNotFoundError("data/caches/school_cache.json not found! Run the standalone script to build the school cache first.")

def count_schools_within_1km(lat, lon):
    return sum(1 for s_lat, s_lon in school_points if haversine_distance(lat, lon, s_lat, s_lon) <= 1000)

df['num_schools_within_1km'] = df.apply(lambda row: count_schools_within_1km(row['lat'], row['lon']), axis=1)

# [v1.9] Feature: Elite School Premium
if len(elite_school_points) > 0:
    df['dist_to_nearest_elite_school'] = df.apply(lambda row: min([haversine_distance(row['lat'], row['lon'], m[0], m[1]) for m in elite_school_points]), axis=1)
    df['elite_school_within_1km'] = (df['dist_to_nearest_elite_school'] <= 1000).astype(int)
else:
    df['dist_to_nearest_elite_school'] = 99999
    df['elite_school_within_1km'] = 0


# ---------------------------------------------------------
# [v1.6] Feature: HDB Resale Price Index (RPI)
# Normalizes macroeconomic inflation cycles
# ---------------------------------------------------------
# Calculate the quarter (1-4) from the month (1-12)
# df['transaction_quarter'] = df['transaction_year'].astype(str) + "-Q" + np.ceil(df['transaction_month'] / 3).astype(int).astype(str)

# Official HDB RPI values (Sample mapping for the years covered)
# In production, this would be loaded from a public data API
# Base Q1 2009 = 100
# hdb_rpi_map = {
#     # 2024
#     "2024-Q1": 183.5,
#     "2024-Q2": 187.9,
#     "2024-Q3": 192.6,
#     "2024-Q4": 193.3,
#     # 2025 (Estimates based on Q4 2024 for notebook data)
#     "2025-Q1": 194.0, 
#     "2025-Q2": 194.5,
#     "2025-Q3": 195.0,
#     "2025-Q4": 195.5,
# }
# Fallback to the latest known index if missing
# default_rpi = 193.3 
# df['hdb_rpi'] = df['transaction_quarter'].map(hdb_rpi_map).fillna(default_rpi)


# Feature definitions
feature_columns = [
    "town", "flat_type", "storey_midpoint", "floor_area_sqm", "flat_model", 
    "remaining_lease_float", "transaction_year", "transaction_month",
    "dist_to_cbd", "dist_to_nearest_mrt", "is_mature_estate", "num_schools_within_1km",
    "dist_to_nearest_elite_school", "elite_school_within_1km"
]
    # "hdb_rpi" # [v1.6] REJECTED - Worsened RMSE


# Removed town from one-hot encoding list
categorical_features = ["flat_type", "flat_model"]

y = df["resale_price"]
X = df[feature_columns]
# ──────────────────────────────────────────────────────────────
# 2. One-hot encoded splits  (for RF / XGBoost / LightGBM)
# ──────────────────────────────────────────────────────────────
X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)
X_temp, X_test, y_temp, y_test = train_test_split(
    X_encoded, y, test_size=0.15, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42
)

# ---------------------------------------------------------
# [v1.4] Feature: Target Encode 'town'
# ---------------------------------------------------------
# Calculate town means ONLY on training set to prevent leakage
town_mean_prices = y_train.groupby(X_train['town']).mean()
global_mean_price = y_train.mean()

# Map to all splits
X_train['town_encoded'] = X_train['town'].map(town_mean_prices).fillna(global_mean_price)
X_val['town_encoded']   = X_val['town'].map(town_mean_prices).fillna(global_mean_price)
X_test['town_encoded']  = X_test['town'].map(town_mean_prices).fillna(global_mean_price)

# Drop original string column
X_train.drop(columns=['town'], inplace=True)
X_val.drop(columns=['town'], inplace=True)
X_test.drop(columns=['town'], inplace=True)

print(f"Training set:   {X_train.shape[0]:,} samples  ({X_train.shape[1]} features)")
print(f"Validation set: {X_val.shape[0]:,} samples")
print(f"Test set:       {X_test.shape[0]:,} samples")
# ──────────────────────────────────────────────────────────────
# 3. Raw-categorical splits  (for CatBoost — no one-hot)
# ──────────────────────────────────────────────────────────────
X_raw = df[feature_columns].copy()
X_raw_temp, X_raw_test, y_temp2, y_test2 = train_test_split(
    X_raw, y, test_size=0.15, random_state=42
)
X_raw_train, X_raw_val, y_raw_train, y_raw_val = train_test_split(
    X_raw_temp, y_temp2, test_size=0.176, random_state=42
)

# Apply target encoding to raw splits too
X_raw_train['town_encoded'] = X_raw_train['town'].map(town_mean_prices).fillna(global_mean_price)
X_raw_val['town_encoded']   = X_raw_val['town'].map(town_mean_prices).fillna(global_mean_price)
X_raw_test['town_encoded']  = X_raw_test['town'].map(town_mean_prices).fillna(global_mean_price)

X_raw_train.drop(columns=['town'], inplace=True)
X_raw_val.drop(columns=['town'], inplace=True)
X_raw_test.drop(columns=['town'], inplace=True)

cat_feature_indices = [X_raw_train.columns.get_loc(c) for c in categorical_features]
# ──────────────────────────────────────────────────────────────
# Helper: evaluate & print
# ──────────────────────────────────────────────────────────────
def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2   = r2_score(y_true, y_pred)
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  RMSE:   ${rmse:>12,.2f}")
    print(f"  MAE:    ${mae:>12,.2f}")
    print(f"  MAPE:    {mape:>11.4f}%")
    print(f"  R²:      {r2:>11.4f}")
    return {"model": name, "rmse": rmse, "mae": mae, "mape": mape, "r2": r2}
results = []
# ──────────────────────────────────────────────────────────────
# 4a. Random Forest  (baseline — same as notebook)
# ──────────────────────────────────────────────────────────────
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1,
)
rf_model.fit(X_train, y_train)
results.append(evaluate("Random Forest (baseline)", y_val, rf_model.predict(X_val)))
# ──────────────────────────────────────────────────────────────
# 4b. XGBoost
# ──────────────────────────────────────────────────────────────
import xgboost as xgb
xgb_model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,            # L1 regularisation
    reg_lambda=1.0,           # L2 regularisation
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50, # stop if val loss doesn't improve for 50 rounds
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)
print(f"\n  XGBoost best iteration: {xgb_model.best_iteration}")
results.append(evaluate("XGBoost", y_val, xgb_model.predict(X_val)))
# ──────────────────────────────────────────────────────────────
# 4c. LightGBM
# ──────────────────────────────────────────────────────────────
import lightgbm as lgb
lgb_model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=63,            # must be < 2^max_depth
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    verbosity=-1,             # suppress warnings
)
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
        lgb.early_stopping(50),   # stop if val loss doesn't improve for 50 rounds
        lgb.log_evaluation(0),    # suppress per-round logging
    ],
)
print(f"\n  LightGBM best iteration: {lgb_model.best_iteration_}")
results.append(evaluate("LightGBM", y_val, lgb_model.predict(X_val)))
# ──────────────────────────────────────────────────────────────
# 4d. CatBoost  (uses raw categoricals — no one-hot encoding!)
# ──────────────────────────────────────────────────────────────
from catboost import CatBoostRegressor
cb_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3,
    subsample=0.8,
    random_seed=42,
    early_stopping_rounds=50,
    verbose=0,                # silent training
)
cb_model.fit(
    X_raw_train, y_raw_train,
    eval_set=(X_raw_val, y_raw_val),
    cat_features=cat_feature_indices,
)
print(f"\n  CatBoost best iteration: {cb_model.get_best_iteration()}")
results.append(evaluate("CatBoost", y_raw_val, cb_model.predict(X_raw_val)))

# ──────────────────────────────────────────────────────────────
# 4e. Weighted Ensemble (XGB + LGBM + CB)
# ──────────────────────────────────────────────────────────────
# We weight XGB heavier since it had the best independent performance
ensemble_predict = (0.5 * xgb_model.predict(X_val)) + (0.3 * lgb_model.predict(X_val)) + (0.2 * cb_model.predict(X_raw_val))
results.append(evaluate("Weighted Ensemble", y_val, ensemble_predict))

# ──────────────────────────────────────────────────────────────
# 5. Summary comparison table
# ──────────────────────────────────────────────────────────────
print("\n")
print("=" * 70)
print("  MODEL COMPARISON SUMMARY  (validation set)")
print("=" * 70)
print(f"  {'Model':<28} {'RMSE':>12} {'MAE':>12} {'MAPE %':>10} {'R²':>8}")
print("-" * 70)
for r in results:
    print(
        f"  {r['model']:<28} "
        f"${r['rmse']:>11,.0f} "
        f"${r['mae']:>11,.0f} "
        f"{r['mape']:>9.2f}% "
        f"{r['r2']:>7.4f}"
    )
print("=" * 70)