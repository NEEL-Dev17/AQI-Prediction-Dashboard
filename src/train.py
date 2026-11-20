import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

FEATURES = ["pm25", "pm10", "no2", "so2", "o3",
            "hour", "day", "month", "dow",
            "temperature", "humidity", "wind_speed"]
MODEL_PATH = Path("models/xgb_aqi.pkl")


def train():
    df = pd.read_parquet("data/processed/features.parquet").dropna(subset=["AQI"])
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    df = df.sort_values("timestamp").reset_index(drop=True)
    X, y = df[FEATURES], df["AQI"]
    n = len(df)
    n_splits = max(2, min(5, n - 1))   # never more folds than samples allow
    tscv = TimeSeriesSplit(n_splits=n_splits)
    print(f"Using n_splits={n_splits} on {n} samples")
    scores = []
    for tr, te in tscv.split(X):
        m = XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6,
                         subsample=0.8, colsample_bytree=0.8, n_jobs=-1)
        m.fit(X.iloc[tr], y.iloc[tr])
        pred = m.predict(X.iloc[te])
        scores.append({"mae": mean_absolute_error(y.iloc[te], pred),
                       "r2": r2_score(y.iloc[te], pred)})
    print("CV scores:\n", pd.DataFrame(scores).mean())
    final = XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6,
                         subsample=0.8, colsample_bytree=0.8, n_jobs=-1)
    final.fit(X, y)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(final, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    train()