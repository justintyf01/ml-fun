"""
HDB Resale Price Prediction — Feature Engineering Pipeline
==========================================================
Single feature pipeline used by train.py and export.py.

Feature groups
--------------
Base:     storey midpoint, remaining lease, time, lease decay, building age
Location: dist to CBD, regional hubs, actual nearest MRT, lat/lon
Amenity:  estate maturity, schools, hawker centres, shopping malls, parks
Phase 4:  feature interactions, storey bands, market heat indicators

Run from project root: python features.py  (prints the feature column list)
"""
import math
import json
import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

_ROOT = os.path.dirname(os.path.abspath(__file__))


# ── Geometry ──────────────────────────────────────────────────────────────────

def haversine_distance(lat1, lon1, lat2, lon2):
    """Straight-line distance in metres between two GPS coordinates."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Constants ─────────────────────────────────────────────────────────────────

CBD_COORD = (1.283933, 103.851463)  # Raffles Place MRT

# 4 regional hubs instead of all 171 MRT stations — preserves regional
# centrality signal (using all 171 degraded RMSE by ~$1k in experiments)
REGIONAL_HUBS = [
    (1.369933, 103.84958),   # Ang Mo Kio  (North)
    (1.436058, 103.786053),  # Woodlands   (North-West)
    (1.333152, 103.742286),  # Jurong East (West)
    (1.353266, 103.945143),  # Tampines    (East)
]

MATURE_ESTATES = [
    "ANG MO KIO", "BEDOK", "BISHAN", "BUKIT MERAH", "BUKIT TIMAH",
    "CLEMENTI", "GEYLANG", "KALLANG/WHAMPOA", "MARINE PARADE",
    "PASIR RIS", "QUEENSTOWN", "SERANGOON", "TAMPINES", "TOA PAYOH",
]

# GEP feeder schools — proximity commands a measurable price premium
GEP_SCHOOLS = [
    "ANGLO-CHINESE SCHOOL (PRIMARY)", "CATHOLIC HIGH SCHOOL",
    "HENRY PARK PRIMARY SCHOOL", "NAN HUA PRIMARY SCHOOL",
    "NANYANG PRIMARY SCHOOL", "RAFFLES GIRLS' PRIMARY SCHOOL",
    "ROSYTH SCHOOL", "ST. HILDA'S PRIMARY SCHOOL", "TAO NAN SCHOOL",
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(csv_path=None, fallback_path=None):
    """Load the HDB resale dataset, preferring the expanded 2017+ version."""
    if csv_path is None:
        csv_path = os.path.join(_ROOT, "data", "hdb_prices_2017.csv")
    if fallback_path is None:
        fallback_path = os.path.join(_ROOT, "data", "HDB_Resale_Prices.csv")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df):,} rows from {os.path.basename(csv_path)}")
    elif os.path.exists(fallback_path):
        df = pd.read_csv(fallback_path)
        print(f"  Loaded {len(df):,} rows from {os.path.basename(fallback_path)}")
    else:
        raise FileNotFoundError(f"No dataset found at {csv_path} or {fallback_path}")
    return df


# ── Feature engineering ───────────────────────────────────────────────────────

def add_base_features(df):
    """
    Parse storey range and remaining lease strings; add time, decay, and age features.

    New columns: storey_midpoint, remaining_lease_float, transaction_year,
    transaction_month, lease_below_60, lease_below_40, building_age
    """
    df["years"] = df["remaining_lease"].str.extract(r"(\d+) years").astype(float).fillna(0)
    df["months"] = df["remaining_lease"].str.extract(r"(\d+) month").astype(float).fillna(0)
    df["remaining_lease_float"] = df["years"] + (df["months"] / 12)

    df["storey_midpoint"] = (
        df["storey_range"].str.split(" TO ", expand=True).astype(float).mean(axis=1)
    )

    df["transaction_year"] = df["month"].str.extract(r"(\d{4})").astype(int)
    df["transaction_month"] = df["month"].str.extract(r"-(\d{2})").astype(int)

    # Non-linear decay: extra penalty for leases falling below 60 or 40 years
    df["lease_below_60"] = np.maximum(0, 60 - df["remaining_lease_float"])
    df["lease_below_40"] = np.maximum(0, 40 - df["remaining_lease_float"])

    df["building_age"] = df["transaction_year"] - df["lease_commence_date"]

    df.drop(columns=["years", "months"], inplace=True)
    return df


def add_location_features(df, cache_path=None):
    """
    Geocode addresses via the OneMap cache and compute distance-based features.

    New columns: lat, lon, dist_to_cbd, dist_to_nearest_hub,
    dist_to_actual_nearest_mrt
    """
    if cache_path is None:
        cache_path = os.path.join(_ROOT, "data", "caches", "onemap_cache.json")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"{cache_path} not found. Run scripts/build_cache_v2.py first.")

    with open(cache_path) as f:
        address_to_coords = json.load(f)

    df["full_address"] = df["block"] + " " + df["street_name"]
    df["lat"] = df["full_address"].map(lambda x: address_to_coords.get(x, [1.35, 103.8])[0])
    df["lon"] = df["full_address"].map(lambda x: address_to_coords.get(x, [1.35, 103.8])[1])

    df["dist_to_cbd"] = df.apply(
        lambda r: haversine_distance(r["lat"], r["lon"], CBD_COORD[0], CBD_COORD[1]), axis=1
    )
    df["dist_to_nearest_hub"] = df.apply(
        lambda r: min(haversine_distance(r["lat"], r["lon"], h[0], h[1]) for h in REGIONAL_HUBS),
        axis=1,
    )

    mrt_path = os.path.join(_ROOT, "data", "MRT Stations.csv")
    if os.path.exists(mrt_path):
        mrt_df = pd.read_csv(mrt_path)
        mrt_coords = list(zip(mrt_df["Latitude"], mrt_df["Longitude"]))
        print(f"  Loaded {len(mrt_coords)} MRT/LRT stations")
        df["dist_to_actual_nearest_mrt"] = df.apply(
            lambda r: min(haversine_distance(r["lat"], r["lon"], m[0], m[1]) for m in mrt_coords),
            axis=1,
        )
    else:
        print("  WARNING: MRT Stations.csv not found. Using hub distance as fallback.")
        df["dist_to_actual_nearest_mrt"] = df["dist_to_nearest_hub"]

    return df


def _load_point_cache(path):
    """Load a [{lat, lon, ...}] JSON cache and return [(lat, lon), ...]."""
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found. Skipping.")
        return []
    with open(path) as f:
        data = json.load(f)
    return [(d["lat"], d["lon"]) for d in data if d.get("lat") is not None]


def _nearest_distance(lat, lon, points):
    """Distance in metres to the nearest point. Returns 99999 if list is empty."""
    if not points:
        return 99999.0
    return min(haversine_distance(lat, lon, p[0], p[1]) for p in points)


def _count_within(lat, lon, points, radius_m):
    """Count how many points fall within radius_m metres of (lat, lon)."""
    return sum(1 for p in points if haversine_distance(lat, lon, p[0], p[1]) <= radius_m)


def add_amenity_features(df, school_cache_path=None):
    """
    Add estate maturity and proximity features for schools, hawkers, malls, parks.

    New columns: is_mature_estate, num_schools_within_1km,
    dist_to_nearest_elite_school, elite_school_within_1km,
    dist_to_nearest_hawker, num_hawkers_within_1km,
    dist_to_nearest_mall, num_malls_within_2km, dist_to_nearest_park
    """
    if school_cache_path is None:
        school_cache_path = os.path.join(_ROOT, "data", "caches", "school_cache.json")
    if not os.path.exists(school_cache_path):
        raise FileNotFoundError(
            f"{school_cache_path} not found. Run scripts/build_school_cache.py first."
        )

    df["is_mature_estate"] = df["town"].isin(MATURE_ESTATES).astype(int)

    with open(school_cache_path) as f:
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
            lambda r: _nearest_distance(r["lat"], r["lon"], elite_points), axis=1
        )
        df["elite_school_within_1km"] = (df["dist_to_nearest_elite_school"] <= 1000).astype(int)
    else:
        df["dist_to_nearest_elite_school"] = 99999
        df["elite_school_within_1km"] = 0

    hawker_points = _load_point_cache(os.path.join(_ROOT, "data", "caches", "hawker_cache.json"))
    if hawker_points:
        print(f"  Loaded {len(hawker_points)} hawker centres")
        df["dist_to_nearest_hawker"] = df.apply(
            lambda r: _nearest_distance(r["lat"], r["lon"], hawker_points), axis=1
        )
        df["num_hawkers_within_1km"] = df.apply(
            lambda r: _count_within(r["lat"], r["lon"], hawker_points, 1000), axis=1
        )

    mall_points = _load_point_cache(os.path.join(_ROOT, "data", "caches", "mall_cache.json"))
    if mall_points:
        print(f"  Loaded {len(mall_points)} shopping malls")
        df["dist_to_nearest_mall"] = df.apply(
            lambda r: _nearest_distance(r["lat"], r["lon"], mall_points), axis=1
        )
        df["num_malls_within_2km"] = df.apply(
            lambda r: _count_within(r["lat"], r["lon"], mall_points, 2000), axis=1
        )

    park_points = _load_point_cache(os.path.join(_ROOT, "data", "caches", "park_cache.json"))
    if park_points:
        print(f"  Loaded {len(park_points)} parks")
        df["dist_to_nearest_park"] = df.apply(
            lambda r: _nearest_distance(r["lat"], r["lon"], park_points), axis=1
        )

    return df


def add_phase4_features(df):
    """
    Add Phase 4 interaction and market indicator features.

    New columns: area_x_storey, lease_x_area, lease_x_storey,
    is_high_floor, is_ground_floor, town_month_volume, town_year_median_price
    """
    df["area_x_storey"] = df["floor_area_sqm"] * df["storey_midpoint"]
    df["lease_x_area"] = df["remaining_lease_float"] * df["floor_area_sqm"]
    df["lease_x_storey"] = df["remaining_lease_float"] * df["storey_midpoint"]

    df["is_high_floor"] = (df["storey_midpoint"] >= 20).astype(int)
    df["is_ground_floor"] = (df["storey_midpoint"] <= 3).astype(int)

    # Transaction count in same town/month — proxy for market heat
    df["town_month_volume"] = df.groupby(["town", "month"])["resale_price"].transform("count")

    # Median price in same town/year — proxy for local price trend
    df["town_year_median_price"] = df.groupby(
        ["town", "transaction_year"]
    )["resale_price"].transform("median")

    return df


# ── Feature column list ───────────────────────────────────────────────────────

FEATURE_COLUMNS = [
    # Property basics
    "town", "flat_type", "storey_midpoint", "floor_area_sqm", "flat_model",
    "remaining_lease_float", "transaction_year", "transaction_month",
    # Location
    "dist_to_cbd", "dist_to_nearest_hub", "dist_to_actual_nearest_mrt",
    # Amenities
    "is_mature_estate", "num_schools_within_1km",
    "dist_to_nearest_elite_school", "elite_school_within_1km",
    "dist_to_nearest_hawker", "num_hawkers_within_1km",
    "dist_to_nearest_mall", "num_malls_within_2km",
    "dist_to_nearest_park",
    # Lease decay
    "lease_below_60", "lease_below_40", "building_age",
    # Phase 4: spatial coordinates
    "lat", "lon",
    # Phase 4: interactions
    "area_x_storey", "lease_x_area", "lease_x_storey",
    # Phase 4: storey bands
    "is_high_floor", "is_ground_floor",
    # Phase 4: market indicators
    "town_month_volume", "town_year_median_price",
    # Target-encoded (kept last — string cols dropped before model input)
    "street_name",
]

CATEGORICAL_FEATURES = ["flat_type", "flat_model"]    # OHE'd before training
TARGET_ENCODE_FEATURES = ["town", "street_name"]       # smoothed target encoding


# ── Encoding ──────────────────────────────────────────────────────────────────

def smoothed_target_encode(train_series, y_train, val_series=None, test_series=None, min_samples=30):
    """
    Smoothed (Bayesian) target encoding.

    Blends the per-category mean toward the global mean based on sample count:
        encoded = (n * cat_mean + min_samples * global_mean) / (n + min_samples)

    Fitted on train_series only; val/test use the same mapping with unseen
    categories falling back to global_mean.
    """
    global_mean = y_train.mean()
    stats = y_train.groupby(train_series).agg(["mean", "count"])
    smoothed = (
        (stats["count"] * stats["mean"] + min_samples * global_mean)
        / (stats["count"] + min_samples)
    )
    result = {"train": train_series.map(smoothed).fillna(global_mean)}
    if val_series is not None:
        result["val"] = val_series.map(smoothed).fillna(global_mean)
    if test_series is not None:
        result["test"] = test_series.map(smoothed).fillna(global_mean)
    return result


def encode_fold(X_train, y_train, X_val, X_test=None):
    """
    Apply per-fold target encoding then one-hot encoding.

    Target encoding is fitted on X_train only (prevents leakage into val/test).
    Smoothing: min_samples=30 for town, 50 for street_name (higher cardinality).

    Returns dict with keys: X_train, X_val, X_test (OHE-encoded DataFrames).
    """
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy() if X_test is not None else None

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

    return {"X_train": X_train_ohe, "X_val": X_val_ohe, "X_test": X_test_ohe}


# ── Full pipeline ─────────────────────────────────────────────────────────────

def build_features_kfold(csv_path=None, fallback_path=None):
    """
    Run the full feature engineering pipeline and return trainval/test splits.

    Split: 85% trainval, 15% test. Encoding is not applied here — it is
    done per fold inside encode_fold() to prevent target leakage.

    Returns dict: X_trainval, y_trainval, X_test, y_test
    """
    print("=" * 60)
    print("  Feature Engineering Pipeline")
    print("=" * 60)

    print("\n[1/5] Loading data...")
    df = load_data(csv_path, fallback_path)

    print("\n[2/5] Adding base features...")
    df = add_base_features(df)

    print("\n[3/5] Adding location features...")
    df = add_location_features(df)

    print("\n[4/5] Adding amenity features...")
    df = add_amenity_features(df)

    print("\n[5/5] Adding Phase 4 features...")
    df = add_phase4_features(df)

    print(f"\nFinal dataframe: {df.shape[0]:,} rows, {df.shape[1]} columns")

    y = df["resale_price"]
    X = df[FEATURE_COLUMNS].copy()
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    print(f"Train+Val:  {len(X_trainval):,} samples")
    print(f"Test:       {len(X_test):,} samples")
    print(f"Features:   {len(FEATURE_COLUMNS)}")

    return {
        "X_trainval": X_trainval,
        "y_trainval": y_trainval,
        "X_test": X_test,
        "y_test": y_test,
    }


if __name__ == "__main__":
    print("Feature columns:")
    for i, col in enumerate(FEATURE_COLUMNS, 1):
        print(f"  {i:2d}. {col}")
