from dotenv import load_dotenv
load_dotenv()

import os
import time
import logging
from datetime import datetime, timezone

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

AQICN_BASE = "https://api.waqi.info/feed"

POLLUTANT_MAP = {
    "pm25": "pm25", "pm10": "pm10", "no2": "no2",
    "so2": "so2", "o3": "o3", "co": "co",
}
WEATHER_MAP = {"t": "temperature", "h": "humidity", "w": "wind_speed", "p": "pressure"}
UNIT_MAP = {"pm25": "ug/m3", "pm10": "ug/m3", "no2": "ug/m3",
            "so2": "ug/m3", "o3": "ug/m3", "co": "mg/m3"}
WEATHER_UNIT = {"temperature": "C", "humidity": "%",
                "wind_speed": "m/s", "pressure": "hPa"}


def _get_token() -> str:
    token = os.getenv("AQICN_TOKEN")
    if not token:
        raise RuntimeError("AQICN_TOKEN environment variable is not set.")
    return token


def fetch_city(city: str, token: str | None = None, retries: int = 3) -> pd.DataFrame:
    token = token or _get_token()
    url = f"{AQICN_BASE}/{city}/"
    params = {"token": token}
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=20)
            resp.raise_for_status()
            payload = resp.json()
            if payload.get("status") != "ok":
                raise ValueError(f"AQICN error for '{city}': {payload.get('data')}")
            data = payload["data"]
            iaqi = data.get("iaqi", {})
            ts = data.get("time", {}).get("s") or datetime.now(timezone.utc).isoformat()
            rows = []
            for aqicn_key, canonical in POLLUTANT_MAP.items():
                entry = iaqi.get(aqicn_key)
                if entry is None or entry.get("v") is None:
                    continue
                rows.append({"city": city, "timestamp": ts, "pollutant": canonical,
                             "value": float(entry["v"]), "unit": UNIT_MAP[canonical]})
            for aqicn_key, canonical in WEATHER_MAP.items():
                entry = iaqi.get(aqicn_key)
                if entry is None or entry.get("v") is None:
                    continue
                rows.append({"city": city, "timestamp": ts, "pollutant": canonical,
                             "value": float(entry["v"]), "unit": WEATHER_UNIT[canonical]})
            if not rows:
                raise ValueError(f"No data returned for '{city}'.")
            return pd.DataFrame(rows)
        except Exception as e:
            last_err = e
            log.warning(f"[{city}] attempt {attempt}/{retries} failed: {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to fetch '{city}': {last_err}")


def fetch_cities(cities: list[str], sleep_between: float = 1.0) -> pd.DataFrame:
    token = _get_token()
    frames, failed = [], []
    for city in cities:
        try:
            frames.append(fetch_city(city, token=token))
            log.info(f"[{city}] ok")
        except Exception as e:
            log.error(f"[{city}] skipped: {e}")
            failed.append((city, str(e)))
        time.sleep(sleep_between)
    if not frames:
        raise RuntimeError("All city fetches failed.")
    return pd.concat(frames, ignore_index=True)


def run(cities: list[str] | None = None,
        out_path: str = "data/raw/air_quality.parquet") -> pd.DataFrame:
    if cities is None:
        cities = ["Delhi", "Mumbai", "Bengaluru", "Chennai", "Kolkata", "Shanghai"]
    df = fetch_cities(cities)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        old = pd.read_parquet(out_path)
        df = pd.concat([old, df], ignore_index=True)
        df = df.drop_duplicates(subset=["city", "timestamp", "pollutant"])
        log.info(f"Appended: {len(old)} -> {len(df)} rows")
    df.to_parquet(out_path, index=False)
    log.info(f"Saved {len(df)} rows to {out_path}")
    return df


if __name__ == "__main__":
    run()