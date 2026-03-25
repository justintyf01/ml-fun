"""
HDB Resale Price Prediction — Enhanced Feature Engineering (v2)
================================================================
Self-contained feature pipeline with Phase 1 improvements:
  1a. Supports expanded 2017+ dataset
  1b. Actual nearest MRT distance (additive, keeps regional hubs)
  1c. Street-name target encoding (smoothed)
  1d. Non-linear lease decay + building age

Imports from existing utils.py for haversine_distance.
"""
import pandas as pd
import numpy as np
import json
import os

from utils import haversine_distance


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

CBD_COORD = (1.283933, 103.851463)  # Raffles Place MRT

REGIONAL_HUBS = [
    (1.369933, 103.84958),   # AMK (North)
    (1.436058, 103.786053),  # Woodlands (North-West)
    (1.333152, 103.742286),  # Jurong East (West)
    (1.353266, 103.945143),  # Tampines (East)
]

MATURE_ESTATES = [
    "ANG MO KIO", "BEDOK", "BISHAN", "BUKIT MERAH", "BUKIT TIMAH",
    "CLEMENTI", "GEYLANG", "KALLANG/WHAMPOA", "MARINE PARADE",
    "PASIR RIS", "QUEENSTOWN", "SERANGOON", "TAMPINES", "TOA PAYOH"
]

GEP_SCHOOLS = [
    "ANGLO-CHINESE SCHOOL (PRIMARY)", "CATHOLIC HIGH SCHOOL",
    "HENRY PARK PRIMARY SCHOOL", "NAN HUA PRIMARY SCHOOL",
    "NANYANG PRIMARY SCHOOL", "RAFFLES GIRLS' PRIMARY SCHOOL",
    "ROSYTH SCHOOL", "ST. HILDA'S PRIMARY SCHOOL", "TAO NAN SCHOOL"
]


# ──────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────

def load_data(csv_path="data/hdb_prices_2017.csv", fallback_path="data/HDB_Resale_Prices.csv"):
    """Load the HDB resale dataset, preferring the expanded version."""
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Loaded expanded dataset: {len(df):,} rows from {csv_path}")
    elif os.path.exists(fallback_path):
        df = pd.read_csv(fallback_path)
        print(f"Loaded fallback dataset: {len(df):,} rows from {fallback_path}")
    else:
        raise FileNotFoundError(f"No dataset found at {csv_path} or {fallback_path}")
    return df


# ──────────────────────────────────────────────────────────────
# Feature engineering
# ──────────────────────────────────────────────────────────────

def add_base_features(df):
    """Parse remaining_lease, storey, and time features."""
    # Remaining lease → float
    df["years"] = df["remaining_lease"].str.extract(r"(\d+) years").astype(float).fillna(0)
    df["months"] = df["remaining_lease"].str.extract(r"(\d+) month").astype(float).fillna(0)
    df["remaining_lease_float"] = df["years"] + (df["months"] / 12)

    # Storey midpoint
    df["storey_midpoint"] = (
        df["storey_range"].str.split(" TO ", expand=True).astype(float).mean(axis=1)
    )

    # Time features
    df["transaction_year"] = df["month"].str.extract(r"(\d{4})").astype(int)
    df["transaction_month"] = df["month"].str.extract(r"-(\d{2})").astype(int)

    # [v2 - Phase 1d] Non-linear lease decay
    df["lease_below_60"] = np.maximum(0, 60 - df["remaining_lease_float"])
    df["lease_below_40"] = np.maximum(0, 40 - df["remaining_lease_float"])

    # [v2 - Phase 1d] Building age
    df["building_age"] = df["transaction_year"] - df["lease_commence_date"]

    df.drop(columns=["years", "months"], inplace=True)
    return df


def add_location_features(df, cache_path="data/caches/onemap_cache.json"):
    """Add geocoded location features: CBD distance, regional hub distance, actual MRT distance."""
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"{cache_path} not found! Run build_cache.py first.")

    with open(cache_path, "r") as f:
        address_to_coords = json.load(f)

    df["full_address"] = df["block"] + " " + df["street_name"]

    # Map coordinates (fallback to Singapore centre)
    df["lat"] = df["full_address"].map(lambda x: address_to_coords.get(x, [1.35, 103.8])[0])
    df["lon"] = df["full_address"].map(lambda x: address_to_coords.get(x, [1.35, 103.8])[1])

    # Distance to CBD
    df["dist_to_cbd"] = df.apply(
        lambda r: haversine_distance(r["lat"], r["lon"], CBD_COORD[0], CBD_COORD[1]), axis=1
    )

    # Distance to nearest regional hub (existing v1.3 feature)
    df["dist_to_nearest_hub"] = df.apply(
        lambda r: min(haversine_distance(r["lat"], r["lon"], h[0], h[1]) for h in REGIONAL_HUBS),
        axis=1,
    )

    # [v2 - Phase 1b] Distance to ACTUAL nearest MRT station (additive)
    mrt_path = "data/MRT Stations.csv"
    if os.path.exists(mrt_path):
        mrt_df = pd.read_csv(mrt_path)
        mrt_coords = list(zip(mrt_df["Latitude"], mrt_df["Longitude"]))
        print(f"  Loaded {len(mrt_coords)} MRT/LRT stations for actual nearest distance")
        df["dist_to_actual_nearest_mrt"] = df.apply(
            lambda r: min(haversine_distance(r["lat"], r["lon"], m[0], m[1]) for m in mrt_coords),
            axis=1,
        )
    else:
        print("  WARNING: MRT Stations.csv not found. Skipping actual MRT distance.")
        df["dist_to_actual_nearest_mrt"] = df["dist_to_nearest_hub"]

    return df


def _load_point_cache(path):
    """Load a JSON cache of [{name, lat, lon}, ...] and return [(lat, lon), ...]."""
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found. Skipping.")
        return []
    with open(path) as f:
        data = json.load(f)
    return [(d["lat"], d["lon"]) for d in data if d.get("lat") is not None]


def _nearest_distance(lat, lon, points):
    """Distance (m) to nearest point in list."""
    if not points:
        return 99999.0
    return min(haversine_distance(lat, lon, p[0], p[1]) for p in points)


def _count_within(lat, lon, points, radius_m):
    """Count points within radius_m metres."""
    return sum(1 for p in points if haversine_distance(lat, lon, p[0], p[1]) <= radius_m)


def add_amenity_features(df, school_cache_path="data/caches/school_cache.json"):
    """Add estate maturity, school, hawker, mall, and park proximity features."""
    df["is_mature_estate"] = df["town"].isin(MATURE_ESTATES).astype(int)

    # ── Schools ──────────────────────────────────────────────────
    if not os.path.exists(school_cache_path):
        raise FileNotFoundError(f"{school_cache_path} not found! Run build_school_cache.py first.")

    with open(school_cache_path, "r") as f:
        school_data = json.load(f)

    school_points = [(s["lat"], s["lon"]) for s in school_data if s["lat"] is not None]
    elite_points = [
        (s["lat"], s["lon"]) for s in school_data
        if s["lat"] is not None and s["name"] in GEP_SCHOOLS
    ]

    df["num_schools_within_1km"] = df.apply(
        lambda r: _count_within(r["lat"], r["lon"], school_points, 1000), axis=1
    )

    if elite_points:
        df["dist_to_nearest_elite_school"] = df.apply(
            lambda r: _nearest_distance(r["lat"], r["lon"], elite_points), axis=1,
        )
        df["elite_school_within_1km"] = (df["dist_to_nearest_elite_school"] <= 1000).astype(int)
    else:
        df["dist_to_nearest_elite_school"] = 99999
        df["elite_school_within_1km"] = 0

    # ── Hawker centres (Phase 2a) ────────────────────────────────
    hawker_points = _load_point_cache("data/caches/hawker_cache.json")
    if hawker_points:
        print(f"  Loaded {len(hawker_points)} hawker centres")
        df["dist_to_nearest_hawker"] = df.apply(
            lambda r: _nearest_distance(r["lat"], r["lon"], hawker_points), axis=1,
        )
        df["num_hawkers_within_1km"] = df.apply(
            lambda r: _count_within(r["lat"], r["lon"], hawker_points, 1000), axis=1,
        )

    # ── Shopping malls (Phase 2b) ────────────────────────────────
    mall_points = _load_point_cache("data/caches/mall_cache.json")
    if mall_points:
        print(f"  Loaded {len(mall_points)} shopping malls")
        df["dist_to_nearest_mall"] = df.apply(
            lambda r: _nearest_distance(r["lat"], r["lon"], mall_points), axis=1,
        )
        df["num_malls_within_2km"] = df.apply(
            lambda r: _count_within(r["lat"], r["lon"], mall_points, 2000), axis=1,
        )

    # ── Parks (Phase 2c) ─────────────────────────────────────────
    park_points = _load_point_cache("data/caches/park_cache.json")
    if park_points:
        print(f"  Loaded {len(park_points)} parks")
        df["dist_to_nearest_park"] = df.apply(
            lambda r: _nearest_distance(r["lat"], r["lon"], park_points), axis=1,
        )

    return df


# ──────────────────────────────────────────────────────────────
# Target encoding (with smoothing)
# ──────────────────────────────────────────────────────────────

def smoothed_target_encode(train_series, y_train, val_series=None, test_series=None, min_samples=30):
    """
    Smoothed target encoding: blends category mean with global mean
    based on sample count. Prevents overfitting for low-count categories.

    smoothed_mean = (count * cat_mean + min_samples * global_mean) / (count + min_samples)
    """
    global_mean = y_train.mean()
    stats = y_train.groupby(train_series).agg(["mean", "count"])
    smoothed = (stats["count"] * stats["mean"] + min_samples * global_mean) / (stats["count"] + min_samples)

    result = {"train": train_series.map(smoothed).fillna(global_mean)}
    if val_series is not None:
        result["val"] = val_series.map(smoothed).fillna(global_mean)
    if test_series is not None:
        result["test"] = test_series.map(smoothed).fillna(global_mean)
    return result


# ──────────────────────────────────────────────────────────────
# Splitting and encoding
# ──────────────────────────────────────────────────────────────

FEATURE_COLUMNS = [
    # Original features
    "town", "flat_type", "storey_midpoint", "floor_area_sqm", "flat_model",
    "remaining_lease_float", "transaction_year", "transaction_month",
    # Location features (v1.3 + v2)
    "dist_to_cbd", "dist_to_nearest_hub", "dist_to_actual_nearest_mrt",
    # Amenity features (v1.5)
    "is_mature_estate", "num_schools_within_1km",
    "dist_to_nearest_elite_school", "elite_school_within_1km",
    # Amenity features (Phase 2)
    "dist_to_nearest_hawker", "num_hawkers_within_1km",
    "dist_to_nearest_mall", "num_malls_within_2km",
    "dist_to_nearest_park",
    # Lease decay features (v2 - Phase 1d)
    "lease_below_60", "lease_below_40", "building_age",
    # Street name for target encoding (v2 - Phase 1c)
    "street_name",
]

CATEGORICAL_FEATURES = ["flat_type", "flat_model"]
TARGET_ENCODE_FEATURES = ["town", "street_name"]


def prepare_data(df, test_size=0.15, val_size=0.176, random_state=42):
    """
    Split data, apply target encoding and one-hot encoding.
    Returns encoded splits for tree models and raw splits for CatBoost.
    """
    from sklearn.model_selection import train_test_split

    y = df["resale_price"]
    X = df[FEATURE_COLUMNS].copy()

    # Split: 70% train, 15% val, 15% test (same as original)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size, random_state=random_state)

    print(f"Training:   {X_train.shape[0]:,} samples")
    print(f"Validation: {X_val.shape[0]:,} samples")
    print(f"Test:       {X_test.shape[0]:,} samples")

    # --- Target encoding for town and street_name ---
    for col in TARGET_ENCODE_FEATURES:
        min_samples = 30 if col == "town" else 50  # more smoothing for street_name (higher cardinality)
        encoded = smoothed_target_encode(
            X_train[col], y_train, X_val[col], X_test[col], min_samples=min_samples
        )
        X_train[f"{col}_encoded"] = encoded["train"]
        X_val[f"{col}_encoded"] = encoded["val"]
        X_test[f"{col}_encoded"] = encoded["test"]

    # Save raw copies for CatBoost before dropping
    X_train_raw = X_train.copy()
    X_val_raw = X_val.copy()
    X_test_raw = X_test.copy()

    # Drop original string columns (keep encoded versions)
    drop_cols = TARGET_ENCODE_FEATURES
    X_train.drop(columns=drop_cols, inplace=True)
    X_val.drop(columns=drop_cols, inplace=True)
    X_test.drop(columns=drop_cols, inplace=True)

    # One-hot encode for XGBoost/LightGBM
    X_train_ohe = pd.get_dummies(X_train, columns=CATEGORICAL_FEATURES, drop_first=True)
    X_val_ohe = pd.get_dummies(X_val, columns=CATEGORICAL_FEATURES, drop_first=True)
    X_test_ohe = pd.get_dummies(X_test, columns=CATEGORICAL_FEATURES, drop_first=True)

    # Align columns (in case val/test has missing categories)
    X_val_ohe = X_val_ohe.reindex(columns=X_train_ohe.columns, fill_value=0)
    X_test_ohe = X_test_ohe.reindex(columns=X_train_ohe.columns, fill_value=0)

    # CatBoost raw splits (keep categoricals as-is, drop target-encode originals, add encoded)
    for col in TARGET_ENCODE_FEATURES:
        X_train_raw[f"{col}_encoded"] = X_train[f"{col}_encoded"]
        X_val_raw[f"{col}_encoded"] = X_val[f"{col}_encoded"]
        X_test_raw[f"{col}_encoded"] = X_test[f"{col}_encoded"]
    X_train_raw.drop(columns=TARGET_ENCODE_FEATURES, inplace=True)
    X_val_raw.drop(columns=TARGET_ENCODE_FEATURES, inplace=True)
    X_test_raw.drop(columns=TARGET_ENCODE_FEATURES, inplace=True)

    cat_indices = [X_train_raw.columns.get_loc(c) for c in CATEGORICAL_FEATURES]

    print(f"Features (OHE): {X_train_ohe.shape[1]}")
    print(f"Features (raw): {X_train_raw.shape[1]}")

    return {
        # One-hot encoded splits (for RF, XGBoost, LightGBM)
        "X_train": X_train_ohe, "X_val": X_val_ohe, "X_test": X_test_ohe,
        # Raw splits (for CatBoost)
        "X_train_raw": X_train_raw, "X_val_raw": X_val_raw, "X_test_raw": X_test_raw,
        "cat_indices": cat_indices,
        # Targets
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
    }


def prepare_data_kfold(df, test_size=0.15, random_state=42):
    """
    Split data into trainval (85%) and test (15%) for K-fold CV.
    Returns raw features (no encoding) — encoding happens per fold.
    """
    from sklearn.model_selection import train_test_split

    y = df["resale_price"]
    X = df[FEATURE_COLUMNS].copy()

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Train+Val:  {X_trainval.shape[0]:,} samples")
    print(f"Test:       {X_test.shape[0]:,} samples")

    return {
        "X_trainval": X_trainval,
        "y_trainval": y_trainval,
        "X_test": X_test,
        "y_test": y_test,
    }


def encode_fold(X_train, y_train, X_val, X_test=None):
    """
    Apply target encoding and one-hot encoding for a single fold.
    Returns OHE splits (for RF/XGB/LGBM) and raw splits (for CatBoost).
    """
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy() if X_test is not None else None

    # Target encoding fitted on fold-train only
    for col in TARGET_ENCODE_FEATURES:
        min_samples = 30 if col == "town" else 50
        encoded = smoothed_target_encode(
            X_train[col], y_train, X_val[col],
            X_test[col] if X_test is not None else None,
            min_samples=min_samples,
        )
        X_train[f"{col}_encoded"] = encoded["train"]
        X_val[f"{col}_encoded"] = encoded["val"]
        if X_test is not None:
            X_test[f"{col}_encoded"] = encoded["test"]

    # Raw copies for CatBoost (before dropping string cols)
    X_train_raw = X_train.drop(columns=TARGET_ENCODE_FEATURES)
    X_val_raw = X_val.drop(columns=TARGET_ENCODE_FEATURES)
    X_test_raw = X_test.drop(columns=TARGET_ENCODE_FEATURES) if X_test is not None else None

    cat_indices = [X_train_raw.columns.get_loc(c) for c in CATEGORICAL_FEATURES]

    # OHE for gradient boosting
    X_train.drop(columns=TARGET_ENCODE_FEATURES, inplace=True)
    X_val.drop(columns=TARGET_ENCODE_FEATURES, inplace=True)
    if X_test is not None:
        X_test.drop(columns=TARGET_ENCODE_FEATURES, inplace=True)

    X_train_ohe = pd.get_dummies(X_train, columns=CATEGORICAL_FEATURES, drop_first=True)
    X_val_ohe = pd.get_dummies(X_val, columns=CATEGORICAL_FEATURES, drop_first=True)
    X_val_ohe = X_val_ohe.reindex(columns=X_train_ohe.columns, fill_value=0)

    X_test_ohe = None
    if X_test is not None:
        X_test_ohe = pd.get_dummies(X_test, columns=CATEGORICAL_FEATURES, drop_first=True)
        X_test_ohe = X_test_ohe.reindex(columns=X_train_ohe.columns, fill_value=0)

    return {
        "X_train": X_train_ohe, "X_val": X_val_ohe, "X_test": X_test_ohe,
        "X_train_raw": X_train_raw, "X_val_raw": X_val_raw, "X_test_raw": X_test_raw,
        "cat_indices": cat_indices,
    }


# ──────────────────────────────────────────────────────────────
# Full pipeline
# ──────────────────────────────────────────────────────────────

def build_features(csv_path="data/hdb_prices_2017.csv", fallback_path="data/HDB_Resale_Prices.csv"):
    """Run the full feature engineering pipeline. Returns splits ready for training."""
    print("=" * 60)
    print("  Feature Engineering Pipeline v2")
    print("=" * 60)

    print("\n[1/4] Loading data...")
    df = load_data(csv_path, fallback_path)

    print("\n[2/4] Adding base features (storey, time, lease decay)...")
    df = add_base_features(df)

    print("\n[3/4] Adding location features (CBD, MRT hubs, actual MRT)...")
    df = add_location_features(df)

    print("\n[4/4] Adding amenity features (schools, estate maturity)...")
    df = add_amenity_features(df)

    print(f"\nFinal dataframe: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print("\nPreparing train/val/test splits...")
    splits = prepare_data(df)

    return splits


def build_features_kfold(csv_path="data/hdb_prices_2017.csv", fallback_path="data/HDB_Resale_Prices.csv"):
    """Run the feature pipeline and return trainval/test for K-fold CV."""
    print("=" * 60)
    print("  Feature Engineering Pipeline v2 (K-Fold)")
    print("=" * 60)

    print("\n[1/4] Loading data...")
    df = load_data(csv_path, fallback_path)

    print("\n[2/4] Adding base features (storey, time, lease decay)...")
    df = add_base_features(df)

    print("\n[3/4] Adding location features (CBD, MRT hubs, actual MRT)...")
    df = add_location_features(df)

    print("\n[4/4] Adding amenity features (schools, estate maturity)...")
    df = add_amenity_features(df)

    print(f"\nFinal dataframe: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print("\nPreparing trainval/test split...")
    return prepare_data_kfold(df)


if __name__ == "__main__":
    splits = build_features()
    print("\nFeature columns (OHE):")
    for col in splits["X_train"].columns:
        print(f"  {col}")
