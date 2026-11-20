import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

FEATURES = ["pm25", "pm10", "no2", "so2", "o3",
            "hour", "day", "month", "dow",
            "temperature", "humidity", "wind_speed"]

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
METRICS_FILE = MODEL_DIR / "model_metrics.csv"


def train():
    print("Loading data...")
    df = pd.read_parquet("data/processed/features.parquet").dropna(subset=["AQI"])

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    df = df.sort_values("timestamp").reset_index(drop=True)
    X, y = df[FEATURES], df["AQI"]

    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    for tr, te in tscv.split(X):
        m = XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6,
                         subsample=0.8, colsample_bytree=0.8, n_jobs=-1)
        m.fit(X.iloc[tr], y.iloc[tr])
        pred = m.predict(X.iloc[te])
        scores.append({
            "mae": mean_absolute_error(y.iloc[te], pred),
            "r2":  r2_score(y.iloc[te], pred),
        })

    cv_results = pd.DataFrame(scores).mean()
    mae = round(float(cv_results["mae"]), 2)
    r2  = round(float(cv_results["r2"]), 4)
    print(f"\nCV scores -> MAE: {mae}, R2: {r2}")

    print("Training final model on all data...")
    final = XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6,
                         subsample=0.8, colsample_bytree=0.8, n_jobs=-1)
    final.fit(X, y)

    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")
    versioned_path = MODEL_DIR / f"xgb_aqi_{timestamp}.pkl"
    latest_path = MODEL_DIR / "xgb_aqi.pkl"

    joblib.dump(final, versioned_path)
    joblib.dump(final, latest_path)

    print(f"Saved versioned model: {versioned_path}")
    print(f"Updated production model: {latest_path}")

    new_row = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_version": versioned_path.name,
        "mae": mae,
        "r2": r2,
        "rows_trained": len(df),
        "cities": df["city"].nunique()
    }])

    if METRICS_FILE.exists():
        history = pd.read_csv(METRICS_FILE)
        history = pd.concat([history, new_row], ignore_index=True)
    else:
        history = new_row

    history.to_csv(METRICS_FILE, index=False)
    print(f"Logged metrics to: {METRICS_FILE}")


if __name__ == "__main__":
    train()
