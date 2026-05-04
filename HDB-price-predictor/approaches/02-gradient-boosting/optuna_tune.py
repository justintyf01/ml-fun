"""
HDB Resale Price Prediction — Optuna Hyperparameter Tuning
============================================================
Tunes XGBoost, LightGBM, and CatBoost individually using K-Fold CV.
Then trains final models with best params and evaluates ensemble.

Run from the project root: python approaches/02-gradient-boosting/optuna_tune.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import optuna
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from scipy.optimize import minimize
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

from features import build_features_kfold, encode_fold, TARGET_ENCODE_FEATURES, CATEGORICAL_FEATURES

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ──────────────────────────────────────────────────────────────
# Evaluation helper
# ──────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────
# Optuna objective functions
# ──────────────────────────────────────────────────────────────

def xgb_objective(trial, X_trainval, y_trainval):
    params = {
        "n_estimators": 3000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 5, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0, 5.0),
        "random_state": 42,
        "n_jobs": -1,
        "early_stopping_rounds": 50,
    }

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in kf.split(X_trainval):
        X_tr, X_va = X_trainval.iloc[train_idx], X_trainval.iloc[val_idx]
        y_tr, y_va = y_trainval.iloc[train_idx], y_trainval.iloc[val_idx]

        fold_data = encode_fold(X_tr, y_tr, X_va)
        model = xgb.XGBRegressor(**params)
        model.fit(fold_data["X_train"], y_tr, eval_set=[(fold_data["X_val"], y_va)], verbose=False)
        pred = model.predict(fold_data["X_val"])
        scores.append(r2_score(y_va, pred))

    return np.mean(scores)


def lgb_objective(trial, X_trainval, y_trainval):
    params = {
        "n_estimators": 3000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 5, 12),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in kf.split(X_trainval):
        X_tr, X_va = X_trainval.iloc[train_idx], X_trainval.iloc[val_idx]
        y_tr, y_va = y_trainval.iloc[train_idx], y_trainval.iloc[val_idx]

        fold_data = encode_fold(X_tr, y_tr, X_va)
        model = lgb.LGBMRegressor(**params)
        model.fit(fold_data["X_train"], y_tr, eval_set=[(fold_data["X_val"], y_va)],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        pred = model.predict(fold_data["X_val"])
        scores.append(r2_score(y_va, pred))

    return np.mean(scores)


def catboost_objective(trial, X_trainval, y_trainval):
    params = {
        "iterations": 3000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "depth": trial.suggest_int("depth", 5, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "random_strength": trial.suggest_float("random_strength", 0.1, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
        "random_seed": 42,
        "early_stopping_rounds": 50,
        "verbose": 0,
    }

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in kf.split(X_trainval):
        X_tr, X_va = X_trainval.iloc[train_idx], X_trainval.iloc[val_idx]
        y_tr, y_va = y_trainval.iloc[train_idx], y_trainval.iloc[val_idx]

        fold_data = encode_fold(X_tr, y_tr, X_va)
        model = CatBoostRegressor(**params)
        model.fit(fold_data["X_train_raw"], y_tr,
                  eval_set=(fold_data["X_val_raw"], y_va),
                  cat_features=fold_data["cat_indices"])
        pred = model.predict(fold_data["X_val_raw"])
        scores.append(r2_score(y_va, pred))

    return np.mean(scores)


# ──────────────────────────────────────────────────────────────
# Final training with best params
# ──────────────────────────────────────────────────────────────

def train_final(best_params, X_trainval, y_trainval, X_test, y_test):
    """Train all models with best params using 5-fold CV and evaluate."""
    N_FOLDS = 5
    MODEL_NAMES = ["RF", "XGBoost", "LightGBM", "CatBoost"]
    n_trainval = len(X_trainval)
    n_test = len(X_test)
    n_models = len(MODEL_NAMES)

    oof_preds = np.zeros((n_trainval, n_models))
    test_preds = np.zeros((n_test, n_models))

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    for fold_i, (train_idx, val_idx) in enumerate(kf.split(X_trainval)):
        print(f"\n  FOLD {fold_i + 1} / {N_FOLDS}")

        X_tr = X_trainval.iloc[train_idx]
        X_va = X_trainval.iloc[val_idx]
        y_tr = y_trainval.iloc[train_idx]
        y_va = y_trainval.iloc[val_idx]

        fold_data = encode_fold(X_tr, y_tr, X_va, X_test)

        models = [
            RandomForestRegressor(
                n_estimators=300, max_depth=25, min_samples_split=5,
                random_state=42, n_jobs=-1,
            ),
            xgb.XGBRegressor(
                **best_params["xgb"], n_estimators=3000,
                random_state=42, n_jobs=-1, early_stopping_rounds=50,
            ),
            lgb.LGBMRegressor(
                **best_params["lgb"], n_estimators=3000,
                random_state=42, n_jobs=-1, verbosity=-1,
            ),
            CatBoostRegressor(
                **best_params["catboost"], iterations=3000,
                random_seed=42, early_stopping_rounds=50, verbose=0,
            ),
        ]

        for m_i, (model, name) in enumerate(zip(models, MODEL_NAMES)):
            if name == "CatBoost":
                model.fit(fold_data["X_train_raw"], y_tr,
                          eval_set=(fold_data["X_val_raw"], y_va),
                          cat_features=fold_data["cat_indices"])
                oof_preds[val_idx, m_i] = model.predict(fold_data["X_val_raw"])
                test_preds[:, m_i] += model.predict(fold_data["X_test_raw"]) / N_FOLDS
            elif name == "XGBoost":
                model.fit(fold_data["X_train"], y_tr,
                          eval_set=[(fold_data["X_val"], y_va)], verbose=False)
                oof_preds[val_idx, m_i] = model.predict(fold_data["X_val"])
                test_preds[:, m_i] += model.predict(fold_data["X_test"]) / N_FOLDS
            elif name == "LightGBM":
                model.fit(fold_data["X_train"], y_tr, eval_set=[(fold_data["X_val"], y_va)],
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
                oof_preds[val_idx, m_i] = model.predict(fold_data["X_val"])
                test_preds[:, m_i] += model.predict(fold_data["X_test"]) / N_FOLDS
            else:
                model.fit(fold_data["X_train"], y_tr)
                oof_preds[val_idx, m_i] = model.predict(fold_data["X_val"])
                test_preds[:, m_i] += model.predict(fold_data["X_test"]) / N_FOLDS

            fold_r2 = r2_score(y_va, oof_preds[val_idx, m_i])
            print(f"    {name} fold {fold_i + 1} R²: {fold_r2:.4f}")

    # Individual model results
    print(f"\n\n{'=' * 75}")
    print("  INDIVIDUAL MODEL RESULTS (Optuna-Tuned)")
    print(f"{'=' * 75}")

    results = []
    for m_i, name in enumerate(MODEL_NAMES):
        oof_r2 = evaluate(f"{name} (OOF)", y_trainval, oof_preds[:, m_i])
        test_r2 = evaluate(f"{name} (test)", y_test, test_preds[:, m_i])
        results.append((name, oof_r2, test_r2))

    # Constrained ensemble weights
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
        print(f"  MODEL COMPARISON — {label}")
        print(f"{'=' * 75}")
        print(f"  {'Model':<30} {'RMSE':>12} {'MAE':>12} {'MAPE %':>10} {'R²':>8}")
        print("-" * 75)
        for name, oof_r, test_r in results:
            r = oof_r if idx == 1 else test_r
            print(f"  {name:<30} ${r['rmse']:>11,.0f} ${r['mae']:>11,.0f} {r['mape']:>9.2f}% {r['r2']:>7.4f}")
        r = oof_ens if idx == 1 else test_ens
        print(f"  {'Ensemble':<30} ${r['rmse']:>11,.0f} ${r['mae']:>11,.0f} {r['mape']:>9.2f}% {r['r2']:>7.4f}")
        print("=" * 75)


def main():
    N_TRIALS = 50

    print("=" * 60)
    print("  Optuna Hyperparameter Tuning")
    print("=" * 60)

    data = build_features_kfold()
    X_trainval = data["X_trainval"]
    y_trainval = data["y_trainval"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    best_params = {}

    # ── Tune XGBoost ─────────────────────────────────────────────
    print(f"\n{'#' * 60}")
    print(f"  Tuning XGBoost ({N_TRIALS} trials)")
    print(f"{'#' * 60}")
    study_xgb = optuna.create_study(direction="maximize")
    study_xgb.optimize(lambda t: xgb_objective(t, X_trainval, y_trainval), n_trials=N_TRIALS)
    best_params["xgb"] = study_xgb.best_params
    print(f"  Best XGBoost R²: {study_xgb.best_value:.4f}")
    print(f"  Best params: {study_xgb.best_params}")

    # ── Tune LightGBM ───────────────────────────────────────────
    print(f"\n{'#' * 60}")
    print(f"  Tuning LightGBM ({N_TRIALS} trials)")
    print(f"{'#' * 60}")
    study_lgb = optuna.create_study(direction="maximize")
    study_lgb.optimize(lambda t: lgb_objective(t, X_trainval, y_trainval), n_trials=N_TRIALS)
    best_params["lgb"] = study_lgb.best_params
    print(f"  Best LightGBM R²: {study_lgb.best_value:.4f}")
    print(f"  Best params: {study_lgb.best_params}")

    # ── Tune CatBoost ───────────────────────────────────────────
    print(f"\n{'#' * 60}")
    print(f"  Tuning CatBoost ({N_TRIALS} trials)")
    print(f"{'#' * 60}")
    study_cb = optuna.create_study(direction="maximize")
    study_cb.optimize(lambda t: catboost_objective(t, X_trainval, y_trainval), n_trials=N_TRIALS)
    best_params["catboost"] = study_cb.best_params
    print(f"  Best CatBoost R²: {study_cb.best_value:.4f}")
    print(f"  Best params: {study_cb.best_params}")

    # ── Final training with best params ──────────────────────────
    print(f"\n\n{'#' * 60}")
    print("  FINAL TRAINING WITH OPTUNA-TUNED PARAMS")
    print(f"{'#' * 60}")
    train_final(best_params, X_trainval, y_trainval, X_test, y_test)


if __name__ == "__main__":
    main()
