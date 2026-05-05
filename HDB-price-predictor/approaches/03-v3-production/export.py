"""
Export trained XGBoost models and all artifacts needed for inference.
=====================================================================
Runs the full feature pipeline, trains 5-fold XGBoost models with early
stopping, and saves all artifacts to model_artifacts/.

Artifacts produced
------------------
model_artifacts/
  fold_0..4/
    xgboost.json          — trained XGBoost model
    target_encodings.json — per-fold town/street encoding maps
  ohe_columns.json        — OHE column order (from fold 0)
  aggregate_lookups.json  — town/month volume + town/year median price
  reference_data.json     — valid dropdown values for the API
  caches/                 — copied from data/caches/

Run from project root:
    python approaches/03-v3-production/export.py
"""
import sys
import json
import os
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score

from features import (
    load_data, add_base_features, add_location_features, add_amenity_features,
    add_phase4_features, smoothed_target_encode,
    FEATURE_COLUMNS, CATEGORICAL_FEATURES, TARGET_ENCODE_FEATURES,
)

ARTIFACTS_DIR = os.path.join(_ROOT, "model_artifacts")
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
N_FOLDS = 5


def extract_target_encoding(train_series, y_train, min_samples=30):
    """
    Fit smoothed target encoding on a single categorical column.

    Blends the per-category mean toward the global mean when sample counts
    are low (Bayesian smoothing):
        encoded = (n * cat_mean + min_samples * global_mean) / (n + min_samples)

    Returns (mapping_dict, global_mean).
    """
    global_mean = float(y_train.mean())
    stats = y_train.groupby(train_series).agg(["mean", "count"])
    smoothed = (
        (stats["count"] * stats["mean"] + min_samples * global_mean)
        / (stats["count"] + min_samples)
    )
    return {str(k): float(v) for k, v in smoothed.items()}, global_mean


def extract_aggregate_lookups(df):
    """
    Pre-compute lookup tables for Phase 4 market indicator features.

    Returns a dict with:
      town_month_volume       — {town|YYYY-MM: int}   transaction count
      town_year_median_price  — {town|YYYY: float}    median resale price
      fallback_volume         — global mean volume (unseen town/month combos)
      fallback_median         — global median price  (unseen town/year combos)
    """
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

    return {
        "town_month_volume": volume_lookup,
        "town_year_median_price": median_lookup,
        "fallback_volume": float(df.groupby(["town", "month"]).size().mean()),
        "fallback_median": float(df["resale_price"].median()),
    }


def extract_reference_data(df):
    """Extract sorted unique values for API dropdown fields."""
    return {
        "towns": sorted(df["town"].unique().tolist()),
        "flat_types": sorted(df["flat_type"].unique().tolist()),
        "flat_models": sorted(df["flat_model"].unique().tolist()),
        "street_names": sorted(df["street_name"].unique().tolist()),
    }


def extract_block_lookup(df):
    """
    Build a per-block lookup table for the frontend auto-fill.

    Keyed by "BLOCK STREET_NAME" (uppercase). Each entry contains:
      town              — HDB town name
      lease_commence_date — year the 99-year lease started
      floor_areas       — {flat_type: median_sqm} from training data
      flat_models       — {flat_type: most_common_model} from training data
    """
    result = {}
    for (block, street), grp in df.groupby(["block", "street_name"]):
        key = f"{block} {street}"
        floor_areas, flat_models = {}, {}
        for flat_type, ft_grp in grp.groupby("flat_type"):
            floor_areas[flat_type] = round(float(ft_grp["floor_area_sqm"].median()), 1)
            mode = ft_grp["flat_model"].mode()
            flat_models[flat_type] = mode.iloc[0] if len(mode) else "Model A"
        result[key] = {
            "town": str(grp["town"].iloc[0]),
            "lease_commence_date": int(grp["lease_commence_date"].iloc[0]),
            "floor_areas": floor_areas,
            "flat_models": flat_models,
        }
    return result


def extract_street_to_town(df):
    """
    Build a street_name → town mapping for town inference when a block
    is not found in the block_lookup (e.g. recently built blocks).
    """
    return (
        df.groupby("street_name")["town"]
        .agg(lambda x: x.mode().iloc[0])
        .to_dict()
    )


def plot_feature_importance(fold_importances, save_path, top_n=20):
    """
    Save a bar chart of mean XGBoost feature importance (gain) across folds.

    fold_importances: list of {feature: gain} dicts, one per fold.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = pd.DataFrame(fold_importances).fillna(0).mean(axis=0)
    df = df.sort_values(ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(df.index, df.values, color="steelblue")
    ax.set_xlabel("Mean Gain")
    ax.set_title(f"Feature Importance — Top {top_n} (mean gain across 5 folds)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


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

    # Extract lookups before Phase 4 adds derived columns
    reference_data = extract_reference_data(df)

    print("\n[2/7] Extracting aggregate lookup tables...")
    aggregate_lookups = extract_aggregate_lookups(df)

    print("\n[3/7] Building block lookup for frontend auto-fill...")
    block_lookup = extract_block_lookup(df)
    street_to_town = extract_street_to_town(df)
    print(f"  {len(block_lookup):,} blocks indexed")

    df = add_phase4_features(df)

    # ── Prepare splits ──────────────────────────────────────────
    print("\n[4/7] Splitting data...")
    y = df["resale_price"]
    X = df[FEATURE_COLUMNS].copy()
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    print(f"  Train+Val: {len(X_trainval):,}, Test: {len(X_test):,}")

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # ── Train and save per fold ─────────────────────────────────
    print("\n[5/7] Training XGBoost models (5-fold CV)...")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    test_preds = np.zeros(len(X_test))
    fold_importances = []

    for fold_i, (train_idx, val_idx) in enumerate(kf.split(X_trainval)):
        print(f"\n  Fold {fold_i + 1}/{N_FOLDS}")
        fold_dir = os.path.join(ARTIFACTS_DIR, f"fold_{fold_i}")
        os.makedirs(fold_dir, exist_ok=True)

        X_fold_train = X_trainval.iloc[train_idx].copy()
        X_fold_val = X_trainval.iloc[val_idx].copy()
        y_fold_train = y_trainval.iloc[train_idx]
        y_fold_val = y_trainval.iloc[val_idx]
        X_fold_test = X_test.copy()

        # ── Per-fold target encoding ────────────────────────────
        # Fitted on fold-train only to prevent leakage into val/test
        encoding_maps = {}
        for col in TARGET_ENCODE_FEATURES:
            min_samples = 30 if col == "town" else 50
            te_map, global_mean = extract_target_encoding(
                X_fold_train[col], y_fold_train, min_samples
            )
            encoding_maps[col] = {"map": te_map, "global_mean": global_mean}

            for ds in [X_fold_train, X_fold_val, X_fold_test]:
                ds[f"{col}_encoded"] = ds[col].map(
                    lambda x, m=te_map, g=global_mean: m.get(str(x), g)
                )

        with open(os.path.join(fold_dir, "target_encodings.json"), "w") as f:
            json.dump(encoding_maps, f)

        # ── Drop string columns, apply OHE ─────────────────────
        for ds in [X_fold_train, X_fold_val, X_fold_test]:
            ds.drop(columns=TARGET_ENCODE_FEATURES, inplace=True)

        X_train_ohe = pd.get_dummies(X_fold_train, columns=CATEGORICAL_FEATURES, drop_first=True)
        X_val_ohe = pd.get_dummies(X_fold_val, columns=CATEGORICAL_FEATURES, drop_first=True)
        X_test_ohe = pd.get_dummies(X_fold_test, columns=CATEGORICAL_FEATURES, drop_first=True)

        X_val_ohe = X_val_ohe.reindex(columns=X_train_ohe.columns, fill_value=0)
        X_test_ohe = X_test_ohe.reindex(columns=X_train_ohe.columns, fill_value=0)

        # OHE columns are identical across folds; save once from fold 0
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
        print(f"    Best round: {model.best_iteration}  R²: {r2:.4f}")

        fold_importances.append(model.get_booster().get_score(importance_type="gain"))
        test_preds += model.predict(X_test_ohe) / N_FOLDS

        model.get_booster().save_model(os.path.join(fold_dir, "xgboost.ubj"))

    # ── Ensemble test evaluation ────────────────────────────────
    test_r2 = r2_score(y_test, test_preds)
    test_rmse = np.sqrt(np.mean((y_test.values - test_preds) ** 2))
    print(f"\n  Ensemble test R²: {test_r2:.4f}  RMSE: ${test_rmse:,.0f}")

    # ── Save global artifacts ───────────────────────────────────
    print("\n[6/7] Saving global artifacts...")
    with open(os.path.join(ARTIFACTS_DIR, "aggregate_lookups.json"), "w") as f:
        json.dump(aggregate_lookups, f)
    with open(os.path.join(ARTIFACTS_DIR, "reference_data.json"), "w") as f:
        json.dump(reference_data, f)
    with open(os.path.join(ARTIFACTS_DIR, "block_lookup.json"), "w") as f:
        json.dump(block_lookup, f)
    with open(os.path.join(ARTIFACTS_DIR, "street_to_town.json"), "w") as f:
        json.dump(street_to_town, f)

    # ── Copy caches ─────────────────────────────────────────────
    print("\n[7/7] Copying cache files...")
    caches_dir = os.path.join(ARTIFACTS_DIR, "caches")
    os.makedirs(caches_dir, exist_ok=True)

    cache_sources = [
        (os.path.join(_ROOT, "data", "caches", "onemap_cache.json"), "onemap_cache.json"),
        (os.path.join(_ROOT, "data", "caches", "school_cache.json"), "school_cache.json"),
        (os.path.join(_ROOT, "data", "caches", "hawker_cache.json"), "hawker_cache.json"),
        (os.path.join(_ROOT, "data", "caches", "mall_cache.json"),   "mall_cache.json"),
        (os.path.join(_ROOT, "data", "caches", "park_cache.json"),   "park_cache.json"),
        (os.path.join(_ROOT, "data", "MRT Stations.csv"),            "MRT Stations.csv"),
    ]
    for src, dest_name in cache_sources:
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(caches_dir, dest_name))
            print(f"  Copied {dest_name}")
        else:
            print(f"  WARNING: {src} not found!")

    # ── Feature importance plot ─────────────────────────────────
    plot_feature_importance(
        fold_importances,
        os.path.join(PLOTS_DIR, "feature_importance_export.png"),
    )

    print(f"\nAll artifacts saved to {ARTIFACTS_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
