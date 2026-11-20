# src/preprocess.py
"""
Clean AQICN long-format data, compute AQI, add time & weather features,
and persist the wide feature table used for training.
"""

import pandas as pd
import numpy as np

WEATHER_COLS = ["temperature", "humidity", "wind_speed", "pressure"]
POLLUTANT_COLS = ["pm25", "pm10", "no2", "so2", "o3", "co"]


def pivot_pollutants(df: pd.DataFrame) -> pd.DataFrame:
    """Wide format: one row per (city, timestamp)."""
    wide = (
        df.pivot_table(
            index=["city", "timestamp"],
            columns="pollutant",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
    )
    wide.columns.name = None
    return wide


def compute_aqi(df: pd.DataFrame) -> pd.DataFrame:
    """Simplified CPCB AQI from PM2.5 and PM10."""
    def sub_index_pm25(x):
        if pd.isna(x): return np.nan
        if x <= 30:  return x * 50 / 30
        if x <= 60:  return 50 + (x - 30) * 50 / 30
        if x <= 90:  return 100 + (x - 60) * 100 / 30
        if x <= 120: return 200 + (x - 90) * 100 / 30
        if x <= 250: return 300 + (x - 120) * 100 / 130
        return 400 + (x - 250) * 100 / 100

    def sub_index_pm10(x):
        if pd.isna(x): return np.nan
        if x <= 50:  return x
        if x <= 100: return 50 + (x - 50)
        if x <= 250: return 100 + (x - 100) * 100 / 150
        if x <= 350: return 200 + (x - 250) * 100 / 100
        return 300 + (x - 350) * 100 / 80

    if "pm25" in df.columns:
        df["aqi_pm25"] = df["pm25"].apply(sub_index_pm25)
    if "pm10" in df.columns:
        df["aqi_pm10"] = df["pm10"].apply(sub_index_pm10)

    aqi_cols = [c for c in ["aqi_pm25", "aqi_pm10"] if c in df.columns]
    df["AQI"] = df[aqi_cols].max(axis=1)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["hour"]   = df["timestamp"].dt.hour
    df["day"]    = df["timestamp"].dt.day
    df["month"]  = df["timestamp"].dt.month
    df["dow"]    = df["timestamp"].dt.dayofweek
    df["season"] = df["month"].map({
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring",  4: "Spring", 5: "Spring",
        6: "Summer",  7: "Summer", 8: "Summer",
        9: "Monsoon", 10: "Monsoon", 11: "Monsoon",
    })
    return df


def build_features(raw_path="data/raw/air_quality.parquet",
                   out_path="data/processed/features.parquet") -> pd.DataFrame:
    df = pd.read_parquet(raw_path)

    # 1. Pivot long -> wide
    df = pivot_pollutants(df)

    # 2. Guarantee expected columns exist
    for col in POLLUTANT_COLS + WEATHER_COLS:
        if col not in df.columns:
            df[col] = np.nan

    # 3. Sort + forward-fill weather per city
    df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)
    df[WEATHER_COLS] = (
        df.groupby("city")[WEATHER_COLS]
          .transform(lambda s: s.ffill().bfill())
    )

    # 4. Compute AQI
    df = compute_aqi(df)

    # 5. Time features
    df = add_time_features(df)

    # 6. Drop rows without the pollutants the model needs
    df = df.dropna(subset=["pm25", "pm10"])

    df.to_parquet(out_path, index=False)
    return df


if __name__ == "__main__":
    out = build_features()
    print(f"Built features: {out.shape[0]} rows, {out.shape[1]} cols")
    print("Columns:", out.columns.tolist())
    print(out.head())