import pandas as pd
import requests
from pathlib import Path

CITIES = {
    "Delhi":     (28.6139, 77.2090),
    "Mumbai":    (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Chennai":   (13.0827, 80.2707),
    "Kolkata":   (22.5726, 88.3639),
    "Shanghai":  (31.2304, 121.4737),
}

VARS = {
    "temperature_2m":       "temperature",
    "relative_humidity_2m": "humidity",
    "wind_speed_10m":       "wind_speed",
    "surface_pressure":     "pressure",
}

OUT = Path("data/raw/weather.parquet")


def fetch_city(city, lat, lon):
    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude":      lat,
            "longitude":     lon,
            "hourly":        ",".join(VARS.keys()),
            "past_days":     92,
            "forecast_days": 0,
            "timezone":      "UTC",
        },
        timeout=120,
    )
    r.raise_for_status()
    hourly = r.json()["hourly"]
    times = hourly["time"]
    rows = []
    for om_name, canon in VARS.items():
        values = hourly.get(om_name)
        if not values:
            print(f"  [{city}] missing variable: {om_name}")
            continue
        for ts, v in zip(times, values):
            if v is None:
                continue
            rows.append({"city": city, "timestamp": ts,
                         "pollutant": canon, "value": float(v)})
    return pd.DataFrame(rows)


def main():
    frames = []
    for city, (lat, lon) in CITIES.items():
        df_city = fetch_city(city, lat, lon)
        print(f"{city} -> {len(df_city)} rows")
        frames.append(df_city)

    df = pd.concat(frames, ignore_index=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    print("\nTOTAL weather rows:", len(df))
    print(df["pollutant"].value_counts())
    print("\nWrote", OUT)


if __name__ == "__main__":
    main()