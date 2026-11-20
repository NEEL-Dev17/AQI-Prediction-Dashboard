import pandas as pd
import requests
from pathlib import Path

CITIES = {
    # Current Indian Cities
    "Delhi":        (28.6139, 77.2090),
    "Mumbai":       (19.0760, 72.8777),
    "Bengaluru":    (12.9716, 77.5946),
    "Chennai":      (13.0827, 80.2707),
    "Kolkata":      (22.5726, 88.3639),
    # New State Capitals
    "Amaravati":    (16.5149, 80.5163),
    "Itanagar":     (27.0844, 93.6053),
    "Dispur":       (26.1408, 91.7898),
    "Patna":        (25.5941, 85.1376),
    "Raipur":       (21.2514, 81.6296),
    "Panaji":       (15.4909, 73.8278),
    "Gandhinagar":  (23.2156, 72.6369),
    "Chandigarh":   (30.7333, 76.7794),
    "Shimla":       (31.1048, 77.1734),
    "Ranchi":       (23.3441, 85.3096),
    "Thiruvananthapuram": (8.5241, 76.9366),
    "Bhopal":       (23.2599, 77.4126),
    "Imphal":       (24.8170, 93.9368),
    "Shillong":     (25.5788, 91.8933),
    "Aizawl":       (23.7271, 92.7176),
    "Kohima":       (25.6751, 94.1086),
    "Bhubaneswar":  (20.2961, 85.8245),
    "Jaipur":       (26.9124, 75.7873),
    "Gangtok":      (27.3314, 88.6138),
    "Hyderabad":    (17.3850, 78.4867),
    "Agartala":     (23.8315, 91.2868),
    "Lucknow":      (26.8467, 80.9462),
    "Dehradun":     (30.3165, 78.0322),
    # Union Territories
    "Port Blair":   (11.6234, 92.7265),
    "Daman":        (20.3974, 72.8328),
    "New Delhi":    (28.6139, 77.2090),
    "Srinagar":     (34.0837, 74.7973),
    "Jammu":        (32.7266, 74.8570),
    "Leh":          (34.1526, 77.5770),
    "Kavaratti":    (10.5593, 72.6358),
    "Puducherry":   (11.9416, 79.8083),
}

VARS = {
    "pm2_5":                "pm25",
    "pm10":                 "pm10",
    "nitrogen_dioxide":     "no2",
    "sulphur_dioxide":      "so2",
    "ozone":                "o3",
    "carbon_monoxide":      "co",
    "temperature_2m":       "temperature",
    "relative_humidity_2m": "humidity",
    "wind_speed_10m":       "wind_speed",
}

OUT = Path("data/raw/air_quality.parquet")


def fetch_city(city, lat, lon):
    r = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            "latitude":   lat,
            "longitude":  lon,
            "hourly":     ",".join(VARS.keys()),
            "past_days":  92,
            "forecast_days": 0,
            "timezone":   "UTC",
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

    print("\nTOTAL rows:", len(df))
    print("\nCounts per pollutant:")
    print(df["pollutant"].value_counts())
    print("\nWrote", OUT)


if __name__ == "__main__":
    main()