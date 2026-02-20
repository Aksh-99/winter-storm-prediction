# src/train.py
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    precision_recall_curve,
)
from xgboost import XGBClassifier
import shap

DATA_PATH = "data/dataset_flash_freeze.csv"
MODEL_PATH = "models/xgb_flash_freeze.joblib"

SHAP_PNG = "reports/shap_summary.png"
PR_CURVE_PNG = "reports/pr_curve.png"
RISK_TIME_PNG = "reports/risk_over_time.png"


def time_split(df: pd.DataFrame, split_date: str):
    split_dt = pd.to_datetime(split_date)
    train = df[df["date"] < split_dt].copy()
    test = df[df["date"] >= split_dt].copy()
    return train, test


def pick_best_threshold_by_f1(y_true: np.ndarray, proba: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = (2 * prec * rec) / (prec + rec + 1e-12)
    best_idx = int(np.nanargmax(f1))
    if best_idx == 0:
        return 0.5
    return float(thr[best_idx - 1])


def plot_pr_curve(y_true: np.ndarray, proba: np.ndarray, out_path: str):
    prec, rec, _ = precision_recall_curve(y_true, proba)
    ap = average_precision_score(y_true, proba)
    baseline = float(np.mean(y_true))

    plt.figure()
    plt.plot(rec, prec, label=f"PR curve (AP={ap:.3f})")
    plt.axhline(baseline, linestyle="--", label=f"Baseline (pos rate={baseline:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_risk_over_time(dates: pd.Series, y_true: np.ndarray, proba: np.ndarray, threshold: float, out_path: str):
    order = np.argsort(dates.values)
    d = dates.values[order]
    y = y_true[order]
    p = proba[order]

    plt.figure(figsize=(12, 4))
    plt.plot(d, p, linewidth=1.5, label="Predicted risk (probability)")

    # Mark true events (FLASH_FREEZE=1) on the same probability line
    idx_pos = np.where(y == 1)[0]
    plt.scatter(d[idx_pos], p[idx_pos], s=18, label="Actual flash-freeze day")

    # Show chosen threshold
    plt.axhline(threshold, linestyle="--", label=f"Alert threshold={threshold:.3f}")

    plt.ylim(-0.02, 1.02)
    plt.xlabel("Date")
    plt.ylabel("Risk probability")
    plt.title("Risk Probability Over Time (Test Period)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing {DATA_PATH}. Run build_dataset.py first.")

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    # Features
    drop_cols = {"date", "FLASH_FREEZE"}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # Time split
    split_date = "2023-01-01"
    train_df, test_df = time_split(df, split_date)

    X_train = train_df[feature_cols]
    y_train = train_df["FLASH_FREEZE"].astype(int)
    X_test = test_df[feature_cols]
    y_test = test_df["FLASH_FREEZE"].astype(int)

    print(f"Train positives: {y_train.sum()} / {len(y_train)} ({y_train.mean():.3%})")
    print(f"Test  positives: {y_test.sum()} / {len(y_test)} ({y_test.mean():.3%})")

    # Imbalance weight
    pos = float(y_train.sum())
    neg = float(len(y_train) - y_train.sum())
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0
    print(f"scale_pos_weight: {scale_pos_weight:.3f}\n")

    # Model
    model = XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        n_jobs=4,
    )
    model.fit(X_train, y_train)

    # Probabilities
    proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, proba) if y_test.nunique() > 1 else float("nan")
    ap = average_precision_score(y_test, proba) if y_test.nunique() > 1 else float("nan")
    print(f"Test ROC-AUC: {auc:.4f}")
    print(f"Test PR-AUC (Avg Precision): {ap:.4f}\n")

    # Threshold tuning
    best_thr = pick_best_threshold_by_f1(y_test.values, proba)
    preds = (proba >= best_thr).astype(int)

    print(f"Best threshold by F1: {best_thr:.3f}\n")
    print(classification_report(y_test, preds, digits=4))

    # Save model bundle
    os.makedirs("models", exist_ok=True)
    joblib.dump(
        {"model": model, "feature_cols": feature_cols, "split_date": split_date, "threshold": best_thr},
        MODEL_PATH,
    )
    print(f"Saved model: {MODEL_PATH}")

    # Reports folder
    os.makedirs("reports", exist_ok=True)

    # ---- PLOTS YOU ASKED FOR ----
    plot_pr_curve(y_test.values, proba, PR_CURVE_PNG)
    print(f"Saved PR curve: {PR_CURVE_PNG}")

    plot_risk_over_time(test_df["date"], y_test.values, proba, best_thr, RISK_TIME_PNG)
    print(f"Saved risk-over-time plot: {RISK_TIME_PNG}")

    # ---- READABLE FEATURE NAMES (for SHAP plot) ----
    READABLE_NAMES = {
        "PRCP": "Precipitation Today",
        "PRCP_L1": "Precipitation Yesterday",
        "PRCP_3D_SUM": "Precipitation (Last 3 Days)",
        "SNOW": "Snowfall Today",
        "SNOW_L1": "Snowfall Yesterday",
        "SNOW_3D_SUM": "Snowfall (Last 3 Days)",
        "TMAX": "Max Temp Today",
        "TMIN": "Min Temp Today",
        "TAVG": "Avg Temp Today",
        "TMAX_L1": "Max Temp Yesterday",
        "TMIN_L1": "Min Temp Yesterday",
        "TAVG_L1": "Avg Temp Yesterday",
        "TEMP_RANGE": "Temp Swing Today",
        "TEMP_RANGE_L1": "Temp Swing Yesterday",
        "TEMP_RANGE_3D_MEAN": "Temp Swing (3-Day Avg)",
        "TAVG_3D_MEAN": "Avg Temp (3-Day Avg)",
        "DOY": "Day of Year",
        "DOY_SIN": "Seasonality",
        "DOY_COS": "Seasonality",
        "DOW": "Day of Week",
    }

    # SHAP
    explainer = shap.TreeExplainer(model)
    X_shap = X_test.copy()
    if len(X_shap) > 1500:
        X_shap = X_shap.sample(1500, random_state=42)

    shap_values = explainer.shap_values(X_shap)
    X_shap_readable = X_shap.rename(columns=READABLE_NAMES)

    plt.figure()
    shap.summary_plot(shap_values, X_shap_readable, show=False)
    plt.tight_layout()
    plt.savefig(SHAP_PNG, dpi=200)
    plt.close()
    print(f"Saved SHAP summary: {SHAP_PNG}")


if __name__ == "__main__":
    main()