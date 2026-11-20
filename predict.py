# src/predict.py
"""
Inference utilities for the Air Quality Prediction project.
"""

import json
import argparse
from pathlib import Path

import joblib
import pandas as pd

FEATURES = [
    "pm25", "pm10", "no2", "so2", "o3",
    "hour", "day", "month", "dow",
    "temperature", "humidity", "wind_speed",
]

MODEL_PATH = Path("models/xgb_aqi.pkl")
_model = None


def _get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run `python -m src.train` first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def aqi_category(aqi: float) -> str:
    if aqi <= 50:  return "Good"
    if aqi <= 100: return "Moderate"
    if aqi <= 200: return "Poor"
    if aqi <= 300: return "Very Poor"
    if aqi <= 400: return "Severe"
    return "Hazardous"


def _ensure_features(record: dict) -> pd.DataFrame:
    defaults = {
        "hour": 12, "day": 1, "month": 1, "dow": 0,
        "temperature": 25.0, "humidity": 60.0, "wind_speed": 3.0,
    }
    row = {**defaults, **record}
    missing = [f for f in FEATURES if row.get(f) is None]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    return pd.DataFrame([[row[f] for f in FEATURES]], columns=FEATURES)


def predict_aqi(record: dict) -> dict:
    X = _ensure_features(record)
    pred = float(_get_model().predict(X)[0])
    lo, hi = pred * 0.9, pred * 1.1
    return {
        "aqi": round(pred, 1),
        "category": aqi_category(pred),
        "confidence_low": round(lo, 1),
        "confidence_high": round(hi, 1),
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")
    out = df.copy()
    out["predicted_aqi"] = _get_model().predict(out[FEATURES]).round(1)
    out["category"] = out["predicted_aqi"].apply(aqi_category)
    return out


def _cli():
    parser = argparse.ArgumentParser(description="Predict AQI from features.")
    parser.add_argument("--input", required=True, help='JSON string, e.g. \'{"pm25":80,...}\'')
    args = parser.parse_args()
    print(json.dumps(predict_aqi(json.loads(args.input)), indent=2))


if __name__ == "__main__":
    _cli()