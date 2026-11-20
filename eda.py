# src/eda.py
import pandas as pd, plotly.express as px

df = pd.read_parquet("data/processed/features.parquet")

# ---- 1. Seasonal pollution pattern ----
seasonal = (df.groupby(["city", "season"])["AQI"]
              .mean().reset_index())
fig1 = px.line(seasonal, x="season", y="AQI", color="city",
               markers=True,
               title="Seasonal AQI Trends Across Cities",
               category_orders={"season": ["Winter","Spring","Summer","Monsoon"]})
fig1.show()

# ---- 2. Correlation matrix (pollutants + weather) ----
pollutants = ["pm25","pm10","no2","so2","o3","AQI",
              "temperature","humidity","wind_speed"]
corr = df[pollutants].corr()
fig2 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                 title="Pollutant–Weather Correlation Matrix")
fig2.show()

# ---- 3. Diurnal pattern ----
diurnal = df.groupby(["hour"])[["pm25","no2","o3"]].mean().reset_index()
fig3 = px.line(diurnal, x="hour", y=["pm25","no2","o3"],
               title="Average Diurnal Pollution Pattern")
fig3.show()
