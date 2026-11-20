# dashboard/app.py
import os
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import joblib

st.set_page_config(page_title="Air Quality Analytics", layout="wide")

@st.cache_data
def load_data():
    return pd.read_parquet("data/processed/features.parquet")

@st.cache_resource
def load_model():
    return joblib.load("models/xgb_aqi.pkl")

df = load_data()
model = load_model()

# ---------- SIDEBAR ----------
st.sidebar.title("🌍 Controls")
cities = st.sidebar.multiselect(
    "Cities", sorted(df.city.unique()),
    default=sorted(df.city.unique())[:3],
)
pollutants = st.sidebar.multiselect(
    "Pollutants", ["pm25", "pm10", "no2", "so2", "o3"],
    default=["pm25", "pm10"],
)
date_range = st.sidebar.date_input(
    "Date range",
    [df.timestamp.min().date(), df.timestamp.max().date()],
)

mask = (
    df.city.isin(cities)
    & (df.timestamp.dt.date >= date_range[0])
    & (df.timestamp.dt.date <= date_range[1])
)
dff = df[mask]

# ---------- KPIs ----------
st.title("Air Quality Prediction & Environmental Analytics")
if dff.empty:
    st.warning("No data in selected range. Adjust filters.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Cities", len(cities))
c2.metric("Avg AQI", f"{dff.AQI.mean():.1f}")
c3.metric("Max PM2.5", f"{dff.pm25.max():.1f} µg/m³")
c4.metric("Worst City", dff.groupby("city").AQI.mean().idxmax())

# ---------- TIME SERIES ----------
st.subheader("📈 AQI Time Series")
fig = px.line(dff, x="timestamp", y="AQI", color="city", title="AQI over time")
st.plotly_chart(fig, use_container_width=True)

# ---------- CITY COMPARISON ----------
st.subheader("🏙️ City-wise Pollution Comparison")
col1, col2 = st.columns(2)
with col1:
    avg = dff.groupby("city")[pollutants].mean().reset_index()
    fig = px.bar(
        avg.melt(id_vars="city"), x="city", y="value",
        color="variable", barmode="group",
        title="Average Pollutant Levels",
    )
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = px.box(dff, x="city", y="AQI", color="city",
                 title="AQI Distribution by City")
    st.plotly_chart(fig, use_container_width=True)

# ---------- SEASONAL ----------
st.subheader("🌦️ Seasonal Pattern Heatmap")

# 1. Create the heatmap data
heat = (dff.groupby(["city", "season"])["AQI"]
           .mean()
           .unstack())

# 2. FORCEFULLY delete Winter and Spring from the columns
cols_to_keep = [c for c in heat.columns if c not in ["Winter", "Spring"]]
heat = heat[cols_to_keep]

# 3. Plot ONLY if there are columns left
if not heat.empty and len(heat.columns) > 0:
    fig = px.imshow(
        heat, 
        text_auto=".1f", 
        color_continuous_scale="YlOrRd",
        title="Mean AQI by City × Season"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No seasonal data available for the selected range.")

# ---------- CORRELATION ----------
st.subheader("🔗 Pollutant–Weather Correlations")
num_cols = ["pm25", "pm10", "no2", "so2", "o3", "AQI",
            "temperature", "humidity", "wind_speed"]
num_cols = [c for c in num_cols if c in dff.columns]
corr = dff[num_cols].corr()
fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r")
st.plotly_chart(fig, use_container_width=True)

# ---------- WEATHER vs AQI ----------
st.subheader("🌤️ Weather vs AQI")
weather_feature = st.selectbox(
    "Weather variable",
    [c for c in ["temperature", "humidity", "wind_speed", "pressure"] if c in dff.columns],
)
if weather_feature:
    fig = px.scatter(
        dff, x=weather_feature, y="AQI", color="city",
        trendline="ols", opacity=0.5,
        title=f"AQI vs {weather_feature}",
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------- CURRENT WEATHER ----------
st.subheader("🌡️ Latest Weather by City")
latest = (
    dff.sort_values("timestamp")
       .groupby("city")
       .tail(1)[["city", "temperature", "humidity",
                 "wind_speed", "pressure", "AQI"]]
)
st.dataframe(latest.set_index("city"), use_container_width=True)

# ---------- PREDICTION ----------
st.subheader("🔮 Predict AQI")
with st.form("pred_form"):
    a, b, c = st.columns(3)
    pm25 = a.slider("PM2.5", 0.0, 500.0, 60.0)
    pm10 = a.slider("PM10", 0.0, 600.0, 100.0)
    no2  = b.slider("NO2", 0.0, 200.0, 30.0)
    so2  = b.slider("SO2", 0.0, 200.0, 10.0)
    o3   = c.slider("O3", 0.0, 200.0, 40.0)
    hour = c.slider("Hour", 0, 23, 8)
    month = c.slider("Month", 1, 12, 1)
    temp = a.slider("Temp °C", -5.0, 45.0, 25.0)
    hum  = b.slider("Humidity %", 0.0, 100.0, 60.0)
    wind = c.slider("Wind m/s", 0.0, 20.0, 3.0)
    submitted = st.form_submit_button("Predict AQI")

if submitted:
    X = pd.DataFrame(
        [[pm25, pm10, no2, so2, o3, hour, 1, month, 0, temp, hum, wind]],
        columns=["pm25", "pm10", "no2", "so2", "o3",
                 "hour", "day", "month", "dow",
                 "temperature", "humidity", "wind_speed"],
    )
    pred = model.predict(X)[0]
    color = ("🟢" if pred <= 50 else
             "🟡" if pred <= 100 else
             "🟠" if pred <= 200 else
             "🔴" if pred <= 300 else "🟣")
    st.success(f"{color} **Predicted AQI: {pred:.0f}**")
    st.caption("0-50 Good · 51-100 Moderate · 101-200 Poor · "
               "201-300 Very Poor · 301+ Severe")

# ---------- ATTRIBUTION ----------
st.caption(
    "Data source: World Air Quality Index Project (aqicn.org) — "
    "© EPA & contributors."
)