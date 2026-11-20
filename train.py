# src/train.py
"""
Train an XGBoost regressor to predict AQI from pollutants + weather.
Persists model to models/xgb_aqi.pkl.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

FEATURES = [
    "pm25", "pm10", "no2", "so2", "o3",
    "hour", "day", "month", "dow",
    "temperature", "humidity", "wind_speed",
]

MODEL_PATH = Path("models/xgb_aqi.pkl")


def train():
    df = pd.read_parquet("data/processed/features.parquet").dropna(subset=["AQI"])

    # Guard: make sure every feature column exists and isn't all-NaN
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features in training data: {missing}")
    all_nan = [f for f in FEATURES if df[f].isna().all()]
    if all_nan:
        raise ValueError(f"These features are entirely NaN: {all_nan}")

    df = df.sort_values("timestamp").reset_index(drop=True)
    X = df[FEATURES]
    y = df["AQI"]

    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    for tr, te in tscv.split(X):
        model = XGBRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            objective="reg:squarederror", n_jobs=-1,
        )
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[te])
        scores.append({
            "mae": mean_absolute_error(y.iloc[te], pred),
            "r2":  r2_score(y.iloc[te], pred),
        })
    print("CV scores:\n", pd.DataFrame(scores).mean())

    final = XGBRegressor(
        n_estimators=600, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror", n_jobs=-1,
    )
    final.fit(X, y)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(final, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    # Feature importance
    import matplotlib.pyplot as plt
    imp = pd.Series(final.feature_importances_, index=FEATURES).sort_values()
    imp.plot.barh(figsize=(8, 6), title="XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig("models/feature_importance.png")
    print("Saved feature_importance.png")


if __name__ == "__main__":
    train()