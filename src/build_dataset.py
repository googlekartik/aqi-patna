"""Build the Patna AQI + dengue dataset from real public data.

Sources (both free, no API key):
- Weather (daily): Open-Meteo ERA5 archive for Patna (25.5941 N, 85.1376 E),
  2022-01-01 to 2024-12-31. Variables: temperature max/min/mean,
  relative humidity mean, precipitation sum, wind speed max.
- Air quality (hourly): Open-Meteo CAMS air-quality API for the same
  coordinates and range. Variables: PM2.5, PM10, NO2, O3.
  CAMS data for this location starts 2022-08-04; earlier hours are missing,
  so the dataset starts there.

Derived columns:
- AQI (today): Indian CPCB sub-index formula applied to daily mean PM2.5
  and PM10; AQI is the maximum of the two sub-indices.
- aqi_next: tomorrow's AQI (the regression target — next-day forecasting).
- dengue_risk: HIGH / LOW outbreak-risk label. Daily confirmed dengue case
  counts for Patna are not published publicly, so this is a literature-based
  risk indicator, NOT observed cases: Aedes aegypti transmission is favoured
  at 24-33 C with relative humidity >= 60% and recent rainfall that creates
  breeding sites. HIGH when temp_mean is 24-33 C AND humidity_mean >= 60%
  AND 7-day rainfall >= 10 mm. The classifier learns the weather -> risk
  mapping; the label rule and its source are documented in README.md.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = str(Path(__file__).resolve().parent.parent)

# --- CPCB (India) breakpoint tables: (conc_low, conc_high, aqi_low, aqi_high)
PM25_BP = [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200),
           (91, 120, 201, 300), (121, 250, 301, 400), (251, 1000, 401, 500)]
PM10_BP = [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200),
           (251, 350, 201, 300), (351, 430, 301, 400), (431, 2000, 401, 500)]


def sub_index(conc, breakpoints):
    # Compare only the upper bound of each band, so continuous concentrations
    # that fall between the integer band edges (e.g. 30.5) are handled too.
    for c_low, c_high, a_low, a_high in breakpoints:
        if conc <= c_high:
            return round(((a_high - a_low) / (c_high - c_low)) * (conc - c_low) + a_low)
    return 500


def main():
    with open(f"{BASE}/data/raw/patna_weather.json") as f:
        w = json.load(f)["daily"]
    weather = pd.DataFrame({
        "date": pd.to_datetime(w["time"]),
        "temp_max": w["temperature_2m_max"],
        "temp_min": w["temperature_2m_min"],
        "temp_mean": w["temperature_2m_mean"],
        "humidity_mean": w["relative_humidity_2m_mean"],
        "precipitation_sum": w["precipitation_sum"],
        "wind_speed_max": w["wind_speed_10m_max"],
    })

    with open(f"{BASE}/data/raw/patna_airq.json") as f:
        a = json.load(f)["hourly"]
    airq = pd.DataFrame({
        "datetime": pd.to_datetime(a["time"]),
        "pm2_5": a["pm2_5"],
        "pm10": a["pm10"],
        "no2": a["nitrogen_dioxide"],
        "o3": a["ozone"],
    })
    airq["date"] = airq["datetime"].dt.date
    daily_air = airq.groupby("date").agg(
        pm2_5_mean=("pm2_5", "mean"), pm2_5_max=("pm2_5", "max"),
        pm10_mean=("pm10", "mean"), pm10_max=("pm10", "max"),
        no2_mean=("no2", "mean"), o3_mean=("o3", "mean"),
    ).reset_index()
    daily_air["date"] = pd.to_datetime(daily_air["date"])

    df = weather.merge(daily_air, on="date", how="left")
    print("rows before air-quality join filter:", len(df))
    print("missing values per column:\n", df.isna().sum().to_string())
    # CAMS air quality starts 2022-08-04; drop days with no pollution data,
    # then linearly interpolate any small remaining gaps.
    df = df.dropna(subset=["pm2_5_mean", "pm10_mean"])
    df = df.interpolate(limit_direction="both")

    df["aqi"] = [max(sub_index(p25, PM25_BP), sub_index(p10, PM10_BP))
                 for p25, p10 in zip(df["pm2_5_mean"], df["pm10_mean"])]
    df["rain_7d"] = df["precipitation_sum"].rolling(7, min_periods=1).sum()
    df["aqi_next"] = df["aqi"].shift(-1)
    df["dengue_risk"] = np.where(
        (df["temp_mean"].between(24, 33))
        & (df["humidity_mean"] >= 60)
        & (df["rain_7d"] >= 10),
        "HIGH", "LOW")
    df = df.dropna(subset=["aqi_next"]).reset_index(drop=True)

    df.to_csv(f"{BASE}/data/patna_aqi_dengue.csv", index=False)
    print("\nFINAL DATASET")
    print("rows:", len(df), "columns:", len(df.columns))
    print("date range:", df["date"].min().date(), "to", df["date"].max().date())
    print(df.dtypes.to_string())
    print("\ndengue_risk balance:\n", df["dengue_risk"].value_counts().to_string())
    print("\nAQI describe:\n", df["aqi"].describe().to_string())
    print("AQI range:", int(df["aqi"].min()), "-", int(df["aqi"].max()))


if __name__ == "__main__":
    main()
