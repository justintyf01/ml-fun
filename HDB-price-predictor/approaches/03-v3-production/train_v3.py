"""
HDB Resale Price Prediction — Training Script v3 (Phase 4)
============================================================
Uses v3 features (lat/lon, interactions, volume, storey bands).
K-Fold CV with OOF stacking.

Run from the project root: python approaches/03-v3-production/train_v3.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import minimize
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    mean_absolute_percentage_error, r2_score,
)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

from features_v3 import build_features_kfold_v3
from features_v2 import encode_fold


def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)
    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    print(f"  RMSE:   ${rmse:>12,.2f}")
    print(f"  MAE:    ${mae:>12,.2f}")
    print(f"  MAPE:    {mape:>11.4f}%")
    print(f"  R²:      {r2:>11.4f}")
    return {"model": name, "rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


MODEL_NAMES = ["RF", "XGBoost", "LightGBM", "CatBoost"]


def make_models():
    return [
        RandomForestRegressor(
            n_estimators=200, max_depth=25, min_samples_split=5,
            random_state=42, n_jobs=-1,
        ),
        xgb.XGBRegressor(
            n_estimators=2000, learning_rate=0.05, max_depth=8,
            min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42,
            n_jobs=-1, early_stopping_rounds=50,
        ),
        lgb.LGBMRegressor(
            n_estimators=2000, learning_rate=0.05, max_depth=8,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, random_state=42,
            n_jobs=-1, verbosity=-1,
        ),
        CatBoostRegressor(
            iterations=2000, learning_rate=0.05, depth=8,
            l2_leaf_reg=3, subsample=0.8, random_seed=42,
            early_stopping_rounds=50, verbose=0,
        ),
    ]


def fit_model(model, name, X_train, y_train, X_val, y_val,
              X_train_raw=None, X_val_raw=None, cat_indices=None):
    if name == "CatBoost":
        model.fit(X_train_raw, y_train, eval_set=(X_val_raw, y_val), cat_features=cat_indices)
    elif name == "XGBoost":
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elif name == "LightGBM":
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    else:
        model.fit(X_train, y_train)
    return model


def predict_model(model, name, X_ohe, X_raw=None):
    if name == "CatBoost":
        return model.predict(X_raw)
    return model.predict(X_ohe)


def main():
    N_FOLDS = 5
    data = build_features_kfold_v3()

    X_trainval = data["X_trainval"]
    y_trainval = data["y_trainval"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    n_trainval = len(X_trainval)
    n_test = len(X_test)
    n_models = len(MODEL_NAMES)

    oof_preds = np.zeros((n_trainval, n_models))
    test_preds = np.zeros((n_test, n_models))

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    for fold_i, (train_idx, val_idx) in enumerate(kf.split(X_trainval)):
        print(f"\n  FOLD {fold_i + 1} / {N_FOLDS}")

        X_fold_train = X_trainval.iloc[train_idx]
        X_fold_val = X_trainval.iloc[val_idx]
        y_fold_train = y_trainval.iloc[train_idx]
        y_fold_val = y_trainval.iloc[val_idx]

        fold_data = encode_fold(X_fold_train, y_fold_train, X_fold_val, X_test)
        models = make_models()

        for m_i, (model, name) in enumerate(zip(models, MODEL_NAMES)):
            model = fit_model(
                model, name,
                fold_data["X_train"], y_fold_train,
                fold_data["X_val"], y_fold_val,
                fold_data["X_train_raw"], fold_data["X_val_raw"],
                fold_data["cat_indices"],
            )

            val_pred = predict_model(model, name, fold_data["X_val"], fold_data["X_val_raw"])
            oof_preds[val_idx, m_i] = val_pred
            test_pred = predict_model(model, name, fold_data["X_test"], fold_data["X_test_raw"])
            test_preds[:, m_i] += test_pred / N_FOLDS

            fold_r2 = r2_score(y_fold_val, val_pred)
            print(f"    {name} fold {fold_i + 1} R²: {fold_r2:.4f}")

    # Individual model results
    print(f"\n\n{'=' * 75}")
    print("  V3 FEATURES — INDIVIDUAL MODEL RESULTS")
    print(f"{'=' * 75}")

    results = []
    for m_i, name in enumerate(MODEL_NAMES):
        oof_r2 = evaluate(f"{name} (OOF)", y_trainval, oof_preds[:, m_i])
        test_r2 = evaluate(f"{name} (test)", y_test, test_preds[:, m_i])
        results.append((name, oof_r2, test_r2))

    # Ensemble
    def mse_objective(w):
        return np.mean((y_trainval.values - oof_preds @ w) ** 2)

    result = minimize(
        mse_objective,
        x0=np.ones(n_models) / n_models,
        bounds=[(0, 1)] * n_models,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
        method="SLSQP",
    )
    weights = result.x
    print(f"\n  Optimal weights: {dict(zip(MODEL_NAMES, weights.round(4)))}")

    ens_oof = oof_preds @ weights
    ens_test = test_preds @ weights
    oof_ens = evaluate("Ensemble (OOF)", y_trainval, ens_oof)
    test_ens = evaluate("Ensemble (test)", y_test, ens_test)

    # Summary
    for label, idx in [("OOF", 1), ("TEST", 2)]:
        print(f"\n{'=' * 75}")
        print(f"  V3 FEATURES — {label}")
        print(f"{'=' * 75}")
        print(f"  {'Model':<30} {'RMSE':>12} {'MAE':>12} {'MAPE %':>10} {'R²':>8}")
        print("-" * 75)
        for name, oof_r, test_r in results:
            r = oof_r if idx == 1 else test_r
            print(f"  {name:<30} ${r['rmse']:>11,.0f} ${r['mae']:>11,.0f} {r['mape']:>9.2f}% {r['r2']:>7.4f}")
        r = oof_ens if idx == 1 else test_ens
        print(f"  {'Ensemble':<30} ${r['rmse']:>11,.0f} ${r['mae']:>11,.0f} {r['mape']:>9.2f}% {r['r2']:>7.4f}")
        print("=" * 75)


if __name__ == "__main__":
    main()
