import os
import joblib
import pandas as pd

DATA_PATH = "data/dataset_flash_freeze.csv"
MODEL_PATH = "models/xgb_flash_freeze.joblib"

def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Missing model. Run train.py first.")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Missing dataset. Run build_dataset.py first.")

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date")

    # Predict on most recent row
    row = df.iloc[-1]
    X = row[feature_cols].to_frame().T

    # ✅ Force every feature to numeric (fixes object dtype)
    X = X.apply(pd.to_numeric, errors="coerce")

    # Optional safety: if anything became NaN, fill or raise
    if X.isna().any().any():
        # simplest: fill with 0 (or you can use training medians)
        X = X.fillna(0)

    p = model.predict_proba(X)[0, 1]
    print(f"Date: {row['date'].date()}")
    print(f"Flash-freeze risk probability: {p:.3f}")

if __name__ == "__main__":
    main()