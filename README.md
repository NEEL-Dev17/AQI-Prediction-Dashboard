# 🌍 Air Quality Prediction & Environmental Analytics

An end-to-end data science project that analyzes air pollution patterns across six major cities and predicts the Air Quality Index (AQI) using machine learning. 

🔗 **Live Dashboard:** [https://aqi-prediction-dashboard.streamlit.app](https://aqi-prediction-dashboard.streamlit.app) *(Replace with your actual URL)*

## 📖 Overview

Air pollution is one of the most pressing environmental health challenges worldwide. This project provides a complete pipeline to ingest real air quality and weather data, analyze seasonal and diurnal patterns, study pollutant-weather correlations, and predict AQI levels using XGBoost. Everything is visualized in an interactive Streamlit dashboard.

## ✨ Features

- **Multi-city Analysis** — Compare pollution across 6 major cities (Delhi, Mumbai, Bengaluru, Chennai, Kolkata, Shanghai).
- **Multi-pollutant Support** — PM2.5, PM10, NO₂, SO₂, O₃, CO.
- **Weather Integration** — Temperature, humidity, wind speed, pressure.
- **Seasonal Pattern Detection** — Automatically filters available seasons (Summer, Monsoon).
- **Correlation Analysis** — Pollutant ↔ weather interactions via interactive heatmaps.
- **Machine Learning Prediction** — XGBoost regressor (MAE ~3.0, R² ~0.99).
- **Live Prediction Form** — Interactive sliders to predict AQI in real-time.

## 📊 Model Performance

The model is trained on **13,248 hourly samples** spanning 92 days across 6 cities, using 12 features (6 pollutants + 3 weather + 3 time features).

| Metric | Value |
|--------|-------|
| **MAE** | 3.05 AQI points |
| **R² Score** | 0.9898 |
| **Cross-validation** | 5-fold TimeSeriesSplit |
| **Algorithm** | XGBoost (600 trees, depth 6) |

## 📁 Project Structure

```text
aqi-prediction-dashboard/
│
├── data/
│   ├── raw/                  # Fetched pollutant and weather parquet files
│   └── processed/            # Wide-format feature table with calculated AQI
│
├── src/
│   ├── ingest_clean.py       # Fetches pollutant data from Open-Meteo Air Quality API
│   ├── ingest_weather.py     # Fetches weather data from Open-Meteo Forecast API
│   ├── preprocess.py         # Cleans, pivots, computes AQI, adds time features
│   ├── train.py              # Trains XGBoost model with time-series CV
│   └── predict.py            # CLI + batch prediction utilities
│
├── dashboard/
│   └── app.py                # Interactive Streamlit dashboard
│
├── models/
│   └── xgb_aqi.pkl           # Trained XGBoost model
│
├── requirements.txt
└── README.md
```

## 📷 Screenshots

<img width="1535" height="682" alt="image" src="https://github.com/user-attachments/assets/8f62e818-9af3-4465-91c9-b6ec729a9972" />
<img width="1093" height="582" alt="image" src="https://github.com/user-attachments/assets/472bda4c-c65b-4f2d-baf0-4d376be2908d" />
<img width="1143" height="500" alt="image" src="https://github.com/user-attachments/assets/5683342c-d612-4062-9476-470d2a6835a4" />
<img width="1113" height="530" alt="image" src="https://github.com/user-attachments/assets/53a5af6d-a80e-4aef-b639-d053f76979d7" />
<img width="1102" height="513" alt="image" src="https://github.com/user-attachments/assets/12aade2b-f60e-4936-8192-8063bf835498" />
<img width="1150" height="596" alt="image" src="https://github.com/user-attachments/assets/3c201b00-f243-41dd-8d8f-60ddb9f3a93f" />
<img width="1152" height="327" alt="image" src="https://github.com/user-attachments/assets/dcb77198-9ac9-45b3-b536-2f0605d3355b" />
<img width="1167" height="630" alt="image" src="https://github.com/user-attachments/assets/4517185a-35ec-4290-8064-fee59f6c26f6" />

## 📊 Data Sources

- **[Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api)** — Hourly PM2.5, PM10, NO₂, SO₂, O₃, CO from CAMS.
- **[Open-Meteo Weather Forecast API](https://open-meteo.com/en/docs)** — Temperature, humidity, wind speed, pressure.
- **[World Air Quality Index (AQICN)](https://aqicn.org/)** — Optional live station data (requires free API token).

> ⚠️ **Disclaimer:** Satellite/model data from CAMS may differ from ground-truth station measurements. This project is for educational and research purposes.

---

## 🧰 Technologies Used

| Layer | Technologies |
| :--- | :--- |
| **Data Ingestion** | `requests`, `pandas` |
| **Machine Learning** | `scikit-learn`, `xgboost` |
| **Visualization** | `plotly`, `statsmodels` (for trendlines) |
| **Dashboard** | `streamlit` |
| **Deployment** | Streamlit Community Cloud |

---

## 🤝 Contributing

Contributions are welcome! If you'd like to add more cities, pollutants, or improve the model, feel free to open a Pull Request.

---

## 📄 License

This project is licensed under the **MIT License**.

---

<p align="center">
  Built with ❤️ by <a href="https://github.com/NEEL-Dev17">Neelabja Nag</a>
</p>
