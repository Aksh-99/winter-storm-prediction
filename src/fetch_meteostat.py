import os
import requests
import pandas as pd

# Central Park approx
LAT, LON = 40.7812, -73.9665

def fetch_daily(start: str, end: str) -> pd.DataFrame:
    """
    Fetch daily historical data from Open-Meteo for NYC.
    Returns columns: date, TMAX, TMIN, PRCP, SNOW
    Units: °C, mm, cm (snowfall)
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start,
        "end_date": end,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "snowfall_sum",
        ],
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
        "timezone": "America/New_York",
    }

    r = requests.get(url, params=params, timeout=60)
    if not r.ok:
        print("Status:", r.status_code)
        print("URL:", r.url)
        print("Body:", r.text[:1200])
        r.raise_for_status()

    data = r.json()
    daily = data.get("daily", {})
    df = pd.DataFrame({
        "date": daily.get("time", []),
        "TMAX": daily.get("temperature_2m_max", []),
        "TMIN": daily.get("temperature_2m_min", []),
        "PRCP": daily.get("precipitation_sum", []),
        "SNOW": daily.get("snowfall_sum", []),
    })
    df["date"] = pd.to_datetime(df["date"])
    return df

def main():
    os.makedirs("data", exist_ok=True)

    start = "2010-01-01"
    end = "2025-12-31"

    print(f"Fetching Open-Meteo daily: {start} -> {end} ...")
    df = fetch_daily(start, end)

    out_path = "data/noaa_central_park_daily.csv"  # keep downstream unchanged
    df.to_csv(out_path, index=False)

    print(f"Saved: {out_path} | rows: {len(df):,}")
    print(df.head())

if __name__ == "__main__":
    main()