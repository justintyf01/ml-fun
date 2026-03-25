"""
Export trained XGBoost models and all artifacts needed for inference.
============================================================
Runs the v3 feature pipeline, trains 5-fold XGBoost models,
and saves everything into model_artifacts/.

Run from the project root: python approaches/03-v3-production/export_models.py
"""
import sys
import json
import os
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score

from features_v3 import FEATURE_COLUMNS_V3, add_phase4_features
from features_v2 import (
    load_data, add_base_features, add_location_features, add_amenity_features,
    smoothed_target_encode, CATEGORICAL_FEATURES, TARGET_ENCODE_FEATURES,
)

ARTIFACTS_DIR = os.path.join(_ROOT, "model_artifacts")
N_FOLDS = 5


def extract_target_encoding(train_series, y_train, min_samples=30):
    """Fit smoothed target encoding and return the mapping dict + global mean."""
    global_mean = float(y_train.mean())
    stats = y_train.groupby(train_series).agg(["mean", "count"])
    smoothed = (stats["count"] * stats["mean"] + min_samples * global_mean) / (stats["count"] + min_samples)
    return {str(k): float(v) for k, v in smoothed.items()}, global_mean


def extract_aggregate_lookups(df):
    """Pre-compute town_month_volume and town_year_median_price lookup tables."""
    volume = df.groupby(["town", "month"]).size().reset_index(name="volume")
    volume_lookup = {
        f"{row['town']}|{row['month']}": int(row["volume"])
        for _, row in volume.iterrows()
    }

    median = df.groupby(["town", "transaction_year"])["resale_price"].median().reset_index()
    median_lookup = {
        f"{row['town']}|{int(row['transaction_year'])}": float(row["resale_price"])
        for _, row in median.iterrows()
    }

    # Fallbacks
    global_volume = float(df.groupby(["town", "month"]).size().mean())
    global_median = float(df["resale_price"].median())

    return {
        "town_month_volume": volume_lookup,
        "town_year_median_price": median_lookup,
        "fallback_volume": global_volume,
        "fallback_median": global_median,
    }


def extract_reference_data(df):
    """Extract valid values for dropdowns."""
    return {
        "towns": sorted(df["town"].unique().tolist()),
        "flat_types": sorted(df["flat_type"].unique().tolist()),
        "flat_models": sorted(df["flat_model"].unique().tolist()),
        "street_names": sorted(df["street_name"].unique().tolist()),
    }


def main():
    print("=" * 60)
    print("  Exporting Model Artifacts")
    print("=" * 60)

    # ── Build features ──────────────────────────────────────────
    print("\n[1/6] Loading and engineering features...")
    df = load_data()
    df = add_base_features(df)
    df = add_location_features(df)
    df = add_amenity_features(df)

    # Save reference data before phase4 (which adds derived columns)
    reference_data = extract_reference_data(df)

    # Save aggregate lookups before phase4 modifies df
    print("\n[2/6] Extracting aggregate lookup tables...")
    # Need transaction_year for median lookup
    aggregate_lookups = extract_aggregate_lookups(df)

    df = add_phase4_features(df)

    # ── Prepare splits ──────────────────────────────────────────
    print("\n[3/6] Splitting data...")
    y = df["resale_price"]
    X = df[FEATURE_COLUMNS_V3].copy()
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    print(f"  Train+Val: {len(X_trainval):,}, Test: {len(X_test):,}")

    # ── Create artifacts directory ──────────────────────────────
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # ── Train and save per fold ─────────────────────────────────
    print("\n[4/6] Training XGBoost models (5-fold CV)...")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    test_preds = np.zeros(len(X_test))

    for fold_i, (train_idx, val_idx) in enumerate(kf.split(X_trainval)):
        print(f"\n  Fold {fold_i + 1}/{N_FOLDS}")
        fold_dir = os.path.join(ARTIFACTS_DIR, f"fold_{fold_i}")
        os.makedirs(fold_dir, exist_ok=True)

        X_fold_train = X_trainval.iloc[train_idx]
        X_fold_val = X_trainval.iloc[val_idx]
        y_fold_train = y_trainval.iloc[train_idx]
        y_fold_val = y_trainval.iloc[val_idx]

        # ── Extract and save target encoding maps ───────────────
        encoding_maps = {}
        X_fold_train = X_fold_train.copy()
        X_fold_val = X_fold_val.copy()
        X_fold_test = X_test.copy()

        for col in TARGET_ENCODE_FEATURES:
            min_samples = 30 if col == "town" else 50
            te_map, global_mean = extract_target_encoding(
                X_fold_train[col], y_fold_train, min_samples
            )
            encoding_maps[col] = {"map": te_map, "global_mean": global_mean}

            # Apply encoding
            X_fold_train[f"{col}_encoded"] = X_fold_train[col].map(
                lambda x, m=te_map, g=global_mean: m.get(str(x), g)
            )
            X_fold_val[f"{col}_encoded"] = X_fold_val[col].map(
                lambda x, m=te_map, g=global_mean: m.get(str(x), g)
            )
            X_fold_test[f"{col}_encoded"] = X_fold_test[col].map(
                lambda x, m=te_map, g=global_mean: m.get(str(x), g)
            )

        with open(os.path.join(fold_dir, "target_encodings.json"), "w") as f:
            json.dump(encoding_maps, f)

        # ── Drop original string columns, OHE ──────────────────
        for ds in [X_fold_train, X_fold_val, X_fold_test]:
            ds.drop(columns=TARGET_ENCODE_FEATURES, inplace=True)

        X_train_ohe = pd.get_dummies(X_fold_train, columns=CATEGORICAL_FEATURES, drop_first=True)
        X_val_ohe = pd.get_dummies(X_fold_val, columns=CATEGORICAL_FEATURES, drop_first=True)
        X_test_ohe = pd.get_dummies(X_fold_test, columns=CATEGORICAL_FEATURES, drop_first=True)

        X_val_ohe = X_val_ohe.reindex(columns=X_train_ohe.columns, fill_value=0)
        X_test_ohe = X_test_ohe.reindex(columns=X_train_ohe.columns, fill_value=0)

        # Save OHE columns (first fold only — should be identical across folds)
        if fold_i == 0:
            with open(os.path.join(ARTIFACTS_DIR, "ohe_columns.json"), "w") as f:
                json.dump(X_train_ohe.columns.tolist(), f)

        # ── Train XGBoost ───────────────────────────────────────
        model = xgb.XGBRegressor(
            n_estimators=2000, learning_rate=0.05, max_depth=8,
            min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42,
            n_jobs=-1, early_stopping_rounds=50,
        )
        model.fit(X_train_ohe, y_fold_train,
                  eval_set=[(X_val_ohe, y_fold_val)], verbose=False)

        val_pred = model.predict(X_val_ohe)
        r2 = r2_score(y_fold_val, val_pred)
        print(f"    R²: {r2:.4f}")

        test_preds += model.predict(X_test_ohe) / N_FOLDS

        # Save model
        model.save_model(os.path.join(fold_dir, "xgboost.json"))

    # ── Test set evaluation ─────────────────────────────────────
    test_r2 = r2_score(y_test, test_preds)
    print(f"\n  Test R²: {test_r2:.4f}")

    # ── Save global artifacts ───────────────────────────────────
    print("\n[5/6] Saving global artifacts...")

    with open(os.path.join(ARTIFACTS_DIR, "aggregate_lookups.json"), "w") as f:
        json.dump(aggregate_lookups, f)

    with open(os.path.join(ARTIFACTS_DIR, "reference_data.json"), "w") as f:
        json.dump(reference_data, f)

    # ── Copy caches ─────────────────────────────────────────────
    print("\n[6/6] Copying cache files...")
    caches_dir = os.path.join(ARTIFACTS_DIR, "caches")
    os.makedirs(caches_dir, exist_ok=True)

    cache_sources = [
        (os.path.join(_ROOT, "data", "caches", "onemap_cache.json"), "onemap_cache.json"),
        (os.path.join(_ROOT, "data", "caches", "school_cache.json"), "school_cache.json"),
        (os.path.join(_ROOT, "data", "caches", "hawker_cache.json"), "hawker_cache.json"),
        (os.path.join(_ROOT, "data", "caches", "mall_cache.json"), "mall_cache.json"),
        (os.path.join(_ROOT, "data", "caches", "park_cache.json"), "park_cache.json"),
        (os.path.join(_ROOT, "data", "MRT Stations.csv"), "MRT Stations.csv"),
    ]
    for src, dest_name in cache_sources:
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(caches_dir, dest_name))
            print(f"  Copied {dest_name}")
        else:
            print(f"  WARNING: {src} not found!")

    print(f"\nAll artifacts saved to {ARTIFACTS_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
