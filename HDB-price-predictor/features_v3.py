"""
HDB Resale Price Prediction — Feature Engineering v3 (Phase 4)
================================================================
Extends v2 with experimental features:
  - Raw lat/lon coordinates
  - Price per sqm (as a derived feature for the model to learn from)
  - Feature interactions (floor_area * storey, lease * floor_area)
  - Transaction volume per town/month (market heat indicator)
  - Storey premium bands
"""
import pandas as pd
import numpy as np
import json
import os

_ROOT = os.path.dirname(os.path.abspath(__file__))

from features_v2 import (
    load_data, add_base_features, add_location_features, add_amenity_features,
    smoothed_target_encode, FEATURE_COLUMNS as V2_FEATURE_COLUMNS,
    CATEGORICAL_FEATURES, TARGET_ENCODE_FEATURES,
)


# ──────────────────────────────────────────────────────────────
# Phase 4 features
# ──────────────────────────────────────────────────────────────

def add_phase4_features(df):
    """Add experimental Phase 4 features."""

    # ── Raw lat/lon (lets tree models learn spatial patterns) ─────
    # Already computed in add_location_features, just need to include

    # ── Feature interactions ─────────────────────────────────────
    df["area_x_storey"] = df["floor_area_sqm"] * df["storey_midpoint"]
    df["lease_x_area"] = df["remaining_lease_float"] * df["floor_area_sqm"]
    df["lease_x_storey"] = df["remaining_lease_float"] * df["storey_midpoint"]

    # ── Storey premium bands ─────────────────────────────────────
    df["is_high_floor"] = (df["storey_midpoint"] >= 20).astype(int)
    df["is_ground_floor"] = (df["storey_midpoint"] <= 3).astype(int)

    # ── Transaction volume (market heat) ─────────────────────────
    # Number of transactions in same town during same month
    df["town_month_volume"] = df.groupby(["town", "month"])["resale_price"].transform("count")

    # ── Year-over-year town price trend ──────────────────────────
    town_year_median = df.groupby(["town", "transaction_year"])["resale_price"].transform("median")
    df["town_year_median_price"] = town_year_median

    return df


# Extend feature columns
FEATURE_COLUMNS_V3 = V2_FEATURE_COLUMNS.copy()
# Insert lat/lon before street_name (which is last)
FEATURE_COLUMNS_V3.remove("street_name")
FEATURE_COLUMNS_V3.extend([
    "lat", "lon",
    "area_x_storey", "lease_x_area", "lease_x_storey",
    "is_high_floor", "is_ground_floor",
    "town_month_volume", "town_year_median_price",
    "street_name",
])


def prepare_data_kfold_v3(df, test_size=0.15, random_state=42):
    """Split data for K-fold CV with v3 features."""
    from sklearn.model_selection import train_test_split

    y = df["resale_price"]
    X = df[FEATURE_COLUMNS_V3].copy()

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Train+Val:  {X_trainval.shape[0]:,} samples")
    print(f"Test:       {X_test.shape[0]:,} samples")
    print(f"Features:   {X_trainval.shape[1]}")

    return {
        "X_trainval": X_trainval,
        "y_trainval": y_trainval,
        "X_test": X_test,
        "y_test": y_test,
    }


def build_features_kfold_v3(csv_path=None, fallback_path=None):
    if csv_path is None:
        csv_path = os.path.join(_ROOT, "data", "hdb_prices_2017.csv")
    if fallback_path is None:
        fallback_path = os.path.join(_ROOT, "data", "HDB_Resale_Prices.csv")
    """Run v3 feature pipeline and return trainval/test for K-fold CV."""
    print("=" * 60)
    print("  Feature Engineering Pipeline v3 (Phase 4)")
    print("=" * 60)

    print("\n[1/5] Loading data...")
    df = load_data(csv_path, fallback_path)

    print("\n[2/5] Adding base features...")
    df = add_base_features(df)

    print("\n[3/5] Adding location features...")
    df = add_location_features(df)

    print("\n[4/5] Adding amenity features...")
    df = add_amenity_features(df)

    print("\n[5/5] Adding Phase 4 features (interactions, volume, lat/lon)...")
    df = add_phase4_features(df)

    print(f"\nFinal dataframe: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print("\nPreparing trainval/test split...")
    return prepare_data_kfold_v3(df)


if __name__ == "__main__":
    data = build_features_kfold_v3()
    print("\nFeature columns:")
    for col in FEATURE_COLUMNS_V3:
        print(f"  {col}")
