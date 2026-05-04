"""
HDB Resale Price Prediction — Training Script
=============================================
Evaluates XGBoost with 5-fold CV using the full feature pipeline.
Outputs per-fold metrics and saves diagnostic plots to plots/.

Run from project root:
    python approaches/03-v3-production/train.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    mean_absolute_percentage_error, r2_score,
)

from features import build_features_kfold, encode_fold

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")

# XGBoost hyperparameters
XGB_PARAMS = dict(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=8,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
)


# ── Metrics ───────────────────────────────────────────────────────────────────

def evaluate(y_true, y_pred):
    """Compute RMSE, MAE, MAPE, and R²."""
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred) * 100,
        "r2": r2_score(y_true, y_pred),
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_learning_curves(fold_evals, save_path):
    """
    Plot train and validation RMSE per boosting round for each fold.

    fold_evals: list of dicts returned by model.evals_result() — one per fold.
                Expected keys: "validation_0" (train), "validation_1" (val).
    """
    n = len(fold_evals)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for fold_i, (ax, evals) in enumerate(zip(axes, fold_evals)):
        train_rmse = evals["validation_0"]["rmse"]
        val_rmse = evals["validation_1"]["rmse"]
        best_round = int(np.argmin(val_rmse))

        ax.plot(train_rmse, label="Train", linewidth=1)
        ax.plot(val_rmse, label="Val", linewidth=1)
        ax.axvline(best_round, color="gray", linestyle="--", alpha=0.6,
                   label=f"Best: {best_round}")
        ax.set_title(f"Fold {fold_i + 1}")
        ax.set_xlabel("Boosting Round")
        if fold_i == 0:
            ax.set_ylabel("RMSE ($)")
        ax.legend(fontsize=7)

    fig.suptitle("XGBoost Learning Curves — 5-Fold CV", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_feature_importance(fold_importances, save_path, top_n=20):
    """
    Bar chart of mean feature importance (gain) across all folds.

    fold_importances: list of {feature_name: gain} dicts — one per fold.
    """
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


def plot_oof_scatter(y_true, y_pred, save_path):
    """Scatter plot of OOF predictions vs actual prices."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.05, s=4, color="steelblue")
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lim, lim, "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual Price ($)")
    ax.set_ylabel("Predicted Price ($)")
    ax.set_title("OOF Predictions vs Actuals")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_residuals(y_true, y_pred, save_path):
    """Residuals (actual − predicted) vs predicted price."""
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, alpha=0.05, s=4, color="steelblue")
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted Price ($)")
    ax.set_ylabel("Residual ($)")
    ax.set_title("Residuals vs Predicted")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ── Training loop ─────────────────────────────────────────────────────────────

def main():
    N_FOLDS = 5
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Build feature set (85% trainval / 15% test, no encoding yet)
    data = build_features_kfold()
    X_trainval = data["X_trainval"]
    y_trainval = data["y_trainval"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    oof_preds = np.zeros(len(X_trainval))
    test_preds = np.zeros(len(X_test))
    fold_evals = []
    fold_importances = []
    fold_metrics = []

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    for fold_i, (train_idx, val_idx) in enumerate(kf.split(X_trainval)):
        print(f"\n  Fold {fold_i + 1}/{N_FOLDS}")

        X_fold_train = X_trainval.iloc[train_idx]
        X_fold_val = X_trainval.iloc[val_idx]
        y_fold_train = y_trainval.iloc[train_idx]
        y_fold_val = y_trainval.iloc[val_idx]

        # Per-fold target encoding + OHE (prevents data leakage)
        fold_data = encode_fold(X_fold_train, y_fold_train, X_fold_val, X_test)

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(
            fold_data["X_train"], y_fold_train,
            # Train set first so early stopping monitors validation_1 (val)
            eval_set=[
                (fold_data["X_train"], y_fold_train),
                (fold_data["X_val"], y_fold_val),
            ],
            verbose=False,
        )

        fold_evals.append(model.evals_result())
        fold_importances.append(model.get_booster().get_score(importance_type="gain"))

        val_pred = model.predict(fold_data["X_val"])
        oof_preds[val_idx] = val_pred
        test_preds += model.predict(fold_data["X_test"]) / N_FOLDS

        m = evaluate(y_fold_val, val_pred)
        fold_metrics.append(m)
        print(f"    Best round: {model.best_iteration}  RMSE: ${m['rmse']:,.0f}  R²: {m['r2']:.4f}")

    # ── Results ───────────────────────────────────────────────────────────────
    oof_metrics = evaluate(y_trainval.values, oof_preds)
    test_metrics = evaluate(y_test.values, test_preds)

    print(f"\n{'=' * 62}")
    print("  Per-Fold Validation Metrics")
    print(f"{'=' * 62}")
    print(f"  {'Fold':<8} {'RMSE':>12} {'MAE':>12} {'MAPE %':>10} {'R²':>8}")
    print("-" * 62)
    for i, m in enumerate(fold_metrics):
        print(f"  {i+1:<8} ${m['rmse']:>11,.0f} ${m['mae']:>11,.0f} {m['mape']:>9.2f}% {m['r2']:>7.4f}")

    print(f"\n{'=' * 62}")
    print("  Final Evaluation")
    print(f"{'=' * 62}")
    for label, m in [("OOF (trainval)", oof_metrics), ("Test", test_metrics)]:
        print(f"\n  {label}:")
        print(f"    RMSE:  ${m['rmse']:>12,.2f}")
        print(f"    MAE:   ${m['mae']:>12,.2f}")
        print(f"    MAPE:   {m['mape']:>11.4f}%")
        print(f"    R²:     {m['r2']:>11.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print(f"\n  Saving plots to {PLOTS_DIR}/")
    plot_learning_curves(
        fold_evals,
        os.path.join(PLOTS_DIR, "learning_curves.png"),
    )
    plot_feature_importance(
        fold_importances,
        os.path.join(PLOTS_DIR, "feature_importance.png"),
    )
    plot_oof_scatter(
        y_trainval.values, oof_preds,
        os.path.join(PLOTS_DIR, "oof_scatter.png"),
    )
    plot_residuals(
        y_trainval.values, oof_preds,
        os.path.join(PLOTS_DIR, "residuals.png"),
    )


if __name__ == "__main__":
    main()
