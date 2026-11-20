"""
Bootstrap 1-year historical air-quality data from Open-Meteo archive API.
Outputs same long format as ingest.py.

Usage:
    python -m src.ingest_openmeteo
    python -m src.ingest_openmeteo --days 365
"""

import os
import time
import logging
from datetime import datetime, timedelta, timezone
from argparse import ArgumentParser

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

CITIES = {
    "Delhi":     (28.6139, 77.2090),
    "Mumbai":    (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Chennai":   (13.0827, 80.2707),
    "Kolkata":   (22.5726, 88.3639),
    "Shanghai":  (31.2304, 121.4737),
}

VARS = {
    "pm2_5":               ("pm25",        "ug/m3"),
    "pm10":                ("pm10",        "ug/m3"),
    "nitrogen_dioxide":    ("no2",         "ug/m3"),
    "sulphur_dioxide":     ("so2",         "ug/m3"),
    "ozone":               ("o3",          "ug/m3"),
    "carbon_monoxide":     ("co",          "ug/m3"),
    "temperature_2m":      ("temperature", "C"),
    "relative_humidity_2m":("humidity",    "%"),
    "wind_speed_10m":      ("wind_speed",  "m/s"),
}


def fetch_city(city: str, lat: float, lon: float, days_back: int = 365) -> pd.DataFrame:
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days_back)
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start.isoformat(),
        "end_date":   end.isoformat(),
        "hourly":     ",".join(VARS.keys()),
        "timezone":   "UTC",
    }
    r = requests.get(ARCHIVE_URL, params=params, timeout=120)
    r.raise_for_status()
    payload = r.json()

    if payload.get("error"):
        raise ValueError(f"Open-Meteo error for {city}: {payload.get('reason')}")

    hourly = payload.get("hourly", {})
    if "time" not in hourly:
        raise ValueError(f"No hourly time axis for {city}: keys={list(hourly.keys())}")

    times = hourly["time"]
    rows = []
    for om_var, (canonical, unit) in VARS.items():
        values = hourly.get(om_var)
        if not values:
            log.warning(f"[{city}] missing variable: {om_var}")
            continue
        for ts, val in zip(times, values):
            if val is None:
                continue
            rows.append({
                "city": city,
                "timestamp": ts,
                "pollutant": canonical,
                "value": float(val),
                "unit": unit,
            })

    if not rows:
        raise ValueError(
            f"[{city}] no values returned. hourly keys were {list(hourly.keys())}"
        )

    log.info(f"[{city}] {len(rows)} rows, vars={list(hourly.keys())[:5]}...")
    return pd.DataFrame(rows)


def run(days_back: int = 365,
        out_path: str = "data/raw/air_quality.parquet") -> pd.DataFrame:
    frames = []
    for city, (lat, lon) in CITIES.items():
        try:
            frames.append(fetch_city(city, lat, lon, days_back=days_back))
        except Exception as e:
            log.error(f"[{city}] failed: {e}")
        time.sleep(1)

    if not frames:
        raise RuntimeError("No cities succeeded.")

    df = pd.concat(frames, ignore_index=True)

    if os.path.exists(out_path):
        log.info(f"Overwriting existing {out_path}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    log.info(f"Saved {len(df)} rows to {out_path}")
    return df


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--days", type=int, default=365)
    args = p.parse_args()
    run(days_back=args.days)