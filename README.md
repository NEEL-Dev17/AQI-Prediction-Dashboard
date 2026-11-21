# 🌍 Air Quality Prediction & Environmental Analytics

An end-to-end, self-updating machine learning platform that analyzes air pollution patterns across **30+ Indian State and Union Territory capitals**. It predicts the Air Quality Index (AQI), explains the predictions using SHAP, forecasts future AQI with Prophet, and visualizes everything in a live Streamlit dashboard.

🔗 **Live Dashboard:** [https://aqi-prediction-dashboard.streamlit.app](https://aqi-prediction-dashboard-eepgmrmapnuwwofrw4xtic.streamlit.app/)
📦 **Repository:** [https://github.com/NEEL-Dev17/AQI-Prediction-Dashboard](https://github.com/NEEL-Dev17/AQI-Prediction-Dashboard)

## 📖 Overview

Air pollution is one of the most pressing environmental health challenges. This project provides a complete, production-grade pipeline to ingest real air quality and weather data, engineer features, train an XGBoost model, explain its predictions, forecast future trends, and serve it all through an interactive web application.

The entire pipeline is **automated via GitHub Actions**, which fetches fresh data, retrains the model, and updates the live dashboard every hour.

## ✨ Features

- **Nationwide Coverage** — Analyzes 30+ Indian State and UT capitals (Delhi, Mumbai, Bengaluru, Chennai, Kolkata, Hyderabad, Jaipur, Lucknow, and more).
- **Multi-Pollutant Analysis** — PM2.5, PM10, NO₂, SO₂, O₃, CO.
- **Weather Integration** — Temperature, humidity, wind speed, pressure.
- **Interactive Map** — Plotly MapLibre map with AQI-coded bubbles.
- **Explainable AI (SHAP)** — Waterfall charts showing exactly *why* the model predicted a specific AQI.
- **7-Day Forecasting (Prophet)** — Predicts future AQI with confidence intervals.
- **Automated CI/CD** — GitHub Actions fetches fresh data and retrains the model every hour.
- **Model Versioning** — Timestamped model files and a performance log (`model_metrics.csv`) to track MAE/R² over time.
- **Live Data Freshness** — Timestamp shown on the dashboard so users know when data was last updated.

## 📊 Model Performance

Trained on **13,248+ hourly samples** across 30+ cities, using 12 features (6 pollutants + 3 weather + 3 time features).

| Metric | Value |
|--------|-------|
| **MAE** | 1.6 AQI points |
| **R² Score** | 0.9915 |
| **Cross-validation** | 5-fold TimeSeriesSplit |
| **Algorithm** | XGBoost (600 trees, depth 6) |

## 📁 Project Structure

```text
aqi-prediction-dashboard/
│
├── .github/
│   └── workflows/
│       └── update_data.yml       # Hourly CI/CD automation
│
├── data/
│   ├── raw/                      # Fetched pollutant & weather parquet files
│   └── processed/                # Wide-format feature table with AQI
│
├── src/
│   ├── ingest_clean.py           # Fetches pollutant data (Open-Meteo Air Quality API)
│   ├── ingest_weather.py         # Fetches weather data (Open-Meteo Forecast API)
│   ├── preprocess.py             # Cleans, pivots, computes AQI, adds time features
│   ├── train.py                  # Trains XGBoost with versioning & metric logging
│   └── predict.py                # CLI + batch prediction utilities
│
├── dashboard/
│   └── app.py                    # Streamlit dashboard (map, SHAP, Prophet)
│
├── models/
│   ├── xgb_aqi.pkl               # Live production model
│   ├── xgb_aqi_YYYY_MM_DD.pkl    # Timestamped backups (gitignored)
│   └── model_metrics.csv         # Historical MAE / R² log
│
├── requirements.txt
└── README.md
```

## 📷 Screenshots

<img width="1535" height="682" alt="image" src="https://github.com/user-attachments/assets/8f62e818-9af3-4465-91c9-b6ec729a9972" />
<img width="1103" height="585" alt="image" src="https://github.com/user-attachments/assets/89cf64e5-8f64-4c93-9937-e4678a94a54d" />
<img width="1143" height="500" alt="image" src="https://github.com/user-attachments/assets/5683342c-d612-4062-9476-470d2a6835a4" />
<img width="1113" height="530" alt="image" src="https://github.com/user-attachments/assets/53a5af6d-a80e-4aef-b639-d053f76979d7" />
<img width="1102" height="513" alt="image" src="https://github.com/user-attachments/assets/12aade2b-f60e-4936-8192-8063bf835498" />
<img width="1150" height="596" alt="image" src="https://github.com/user-attachments/assets/3c201b00-f243-41dd-8d8f-60ddb9f3a93f" />
<img width="1152" height="327" alt="image" src="https://github.com/user-attachments/assets/dcb77198-9ac9-45b3-b536-2f0605d3355b" />
<img width="1167" height="630" alt="image" src="https://github.com/user-attachments/assets/4517185a-35ec-4290-8064-fee59f6c26f6" />
<img width="1108" height="670" alt="image" src="https://github.com/user-attachments/assets/2d54dacc-5882-47a3-8d2c-34015c914b98" />
<img width="1142" height="671" alt="image" src="https://github.com/user-attachments/assets/520b5429-a356-478e-b21a-016b7c7d99ec" />


## 🤖 Automation (GitHub Actions)

The workflow at `.github/workflows/update_data.yml` runs **every hour** to:

1. Fetch the latest 92 days of data for all 30+ cities.
2. Merge, preprocess, and retrain the XGBoost model.
3. Append a new row to `models/model_metrics.csv`.
4. Commit the updated `.parquet` and `.pkl` files back to GitHub.
5. Trigger an automatic rebuild of the Streamlit Cloud dashboard.

> **Required setup:** Go to **Settings → Actions → General → Workflow permissions** and select **"Read and write permissions"**.

---

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
| **Explainability** | `shap`, `matplotlib` |
| **Forecasting** | `prophet` |
| **Visualization** | `plotly` |
| **Dashboard** | `streamlit` |
| **Automation** | GitHub Actions (CI/CD) |
| **Deployment** | Streamlit Community Cloud |

---

## 📈 Tracking Model Performance

Every retraining run appends to `models/model_metrics.csv`:

```csv
timestamp,model_version,mae,r2,rows_trained,cities
2026-11-20 23:38:12,xgb_aqi_2026_11_20_2338.pkl,1.6,0.9915,13248,37
---
```
<p align="center">
  Built with ❤️ by <a href="https://github.com/NEEL-Dev17">Neelabja Nag</a>
</p>
