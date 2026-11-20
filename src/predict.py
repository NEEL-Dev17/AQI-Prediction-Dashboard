import json
import argparse
from pathlib import Path
import joblib
import pandas as pd

FEATURES = ["pm25", "pm10", "no2", "so2", "o3",
            "hour", "day", "month", "dow",
            "temperature", "humidity", "wind_speed"]
MODEL_PATH = Path("models/xgb_aqi.pkl")
_model = None


def _get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def aqi_category(aqi: float) -> str:
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Moderate"
    if aqi <= 200: return "Poor"
    if aqi <= 300: return "Very Poor"
    if aqi <= 400: return "Severe"
    return "Hazardous"


def _ensure_features(record: dict) -> pd.DataFrame:
    defaults = {"hour": 12, "day": 1, "month": 1, "dow": 0,
                "temperature": 25.0, "humidity": 60.0, "wind_speed": 3.0}
    row = {**defaults, **record}
    missing = [f for f in FEATURES if row.get(f) is None]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    return pd.DataFrame([[row[f] for f in FEATURES]], columns=FEATURES)


def predict_aqi(record: dict) -> dict:
    X = _ensure_features(record)
    pred = float(_get_model().predict(X)[0])
    return {"aqi": round(pred, 1), "category": aqi_category(pred),
            "confidence_low": round(pred * 0.9, 1),
            "confidence_high": round(pred * 1.1, 1)}


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    args = p.parse_args()
    print(json.dumps(predict_aqi(json.loads(args.input)), indent=2))


if __name__ == "__main__":
    def _cli():
        p = argparse.ArgumentParser()
        p.add_argument("--input", required=True,
                   help="JSON string OR path to a .json file")
        args = p.parse_args()

        raw = args.input
        if raw.endswith(".json") and Path(raw).exists():
            import json as _json
            with open(raw, "r", encoding="utf-8") as f:
                record = _json.load(f)
        else:
            record = json.loads(raw)

        print(json.dumps(predict_aqi(record), indent=2))