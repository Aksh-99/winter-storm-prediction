# src/build_dataset.py
import os
import numpy as np
import pandas as pd

IN_PATH = "data/noaa_central_park_daily.csv"
OUT_PATH = "data/dataset_flash_freeze.csv"

WINTER_MONTHS = {11, 12, 1, 2, 3}

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()

    # Ensure expected columns exist
    for col in ["TMAX", "TMIN", "PRCP", "SNOW"]:
        if col not in df.columns:
            df[col] = np.nan

    # Fill precip/snow missing with 0 (common for station feeds)
    df["PRCP"] = df["PRCP"].fillna(0.0)
    df["SNOW"] = df["SNOW"].fillna(0.0)

    # Derived features
    df["TAVG"] = (df["TMAX"] + df["TMIN"]) / 2.0
    df["TEMP_RANGE"] = df["TMAX"] - df["TMIN"]

    # Lags (yesterday)
    for col in ["TMAX", "TMIN", "TAVG", "PRCP", "SNOW", "TEMP_RANGE"]:
        df[f"{col}_L1"] = df[col].shift(1)

    # Rolling windows (3-day)
    df["PRCP_3D_SUM"] = df["PRCP"].rolling(3).sum()
    df["SNOW_3D_SUM"] = df["SNOW"].rolling(3).sum()
    df["TAVG_3D_MEAN"] = df["TAVG"].rolling(3).mean()
    df["TEMP_RANGE_3D_MEAN"] = df["TEMP_RANGE"].rolling(3).mean()

    # Calendar features
    df["DOW"] = df["date"].dt.dayofweek
    df["MONTH"] = df["date"].dt.month
    df["DOY"] = df["date"].dt.dayofyear

    # Cyclical seasonality encoding
    df["DOY_SIN"] = np.sin(2 * np.pi * df["DOY"] / 365.25)
    df["DOY_COS"] = np.cos(2 * np.pi * df["DOY"] / 365.25)

    return df

def add_label_next_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict *tomorrow's* flash-freeze risk using *today's* conditions.
    Label definition (simple proxy):
      - PRCP today > 0
      - Tomorrow TMAX > 1°C
      - Tomorrow TMIN < -1°C
    """
    df["TMAX_TMR"] = df["TMAX"].shift(-1)
    df["TMIN_TMR"] = df["TMIN"].shift(-1)

    df["FLASH_FREEZE"] = (
        (df["PRCP"] > 0.0) &
        (df["TMAX_TMR"] > 1.0) &
        (df["TMIN_TMR"] < -1.0)
    ).astype(int)

    # Remove helper columns so they cannot leak into features
    df = df.drop(columns=["TMAX_TMR", "TMIN_TMR"], errors="ignore")
    return df

def main():
    if not os.path.exists(IN_PATH):
        raise FileNotFoundError(f"Missing {IN_PATH}. Run fetch script first.")

    df = pd.read_csv(IN_PATH, parse_dates=["date"])

    # Keep winter months only (optional, but makes the dataset more focused)
    df = df[df["date"].dt.month.isin(WINTER_MONTHS)].copy()

    df = add_features(df)
    df = add_label_next_day(df)

    # Drop rows missing essential temperatures (needed for features and label shift)
    df = df.dropna(subset=["TMAX", "TMIN"]).copy()

    # The last row will have a shifted label based on missing "tomorrow" temps
    df = df.dropna(subset=["FLASH_FREEZE"]).copy()

    os.makedirs("data", exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    rate = df["FLASH_FREEZE"].mean()
    print(f"Saved: {OUT_PATH} | rows={len(df):,} | flash_freeze_rate={rate:.3%}")
    print(df[["date", "TMAX", "TMIN", "PRCP", "SNOW", "FLASH_FREEZE"]].head())

if __name__ == "__main__":
    main()