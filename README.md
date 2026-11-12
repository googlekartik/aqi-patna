# AQI Prediction & Dengue-Prone Areas in Patna

One machine-learning project, two predictions, one dataset — built from **real
public data for Patna, Bihar**.

- **Task A — Regression:** forecast tomorrow's Air Quality Index (AQI) from
  today's weather and pollution measurements.
- **Task B — Classification:** predict dengue outbreak risk (HIGH / LOW) from
  the same day's weather.

## Dataset

**Source (free, no API key):** Open-Meteo — ERA5 weather archive and CAMS
air-quality reanalysis for Patna (25.5941° N, 85.1376° E).

| | |
|---|---|
| Rows | **880 days** |
| Columns | **17** (14 features + `aqi`, `aqi_next`, `dengue_risk`) |
| Period | 4 Aug 2022 – 30 Dec 2024 |
| Features | temp_max, temp_min, temp_mean, humidity_mean, precipitation_sum, wind_speed_max, pm2_5_mean, pm2_5_max, pm10_mean, pm10_max, no2_mean, o3_mean, aqi (today), rain_7d |
| Targets | `aqi_next` (tomorrow's AQI), `dengue_risk` (HIGH / LOW) |

CAMS air-quality data for this location starts on 4 Aug 2022, so the
dataset starts there. AQI is computed from daily mean PM2.5 and PM10 using
the **Indian CPCB sub-index formula**. Dengue-risk balance: LOW 584 days,
HIGH 296 days (33.6% HIGH).

> **Honest note on the dengue label.** Daily confirmed dengue case counts for
> Patna are not published publicly, so `dengue_risk` is a **literature-based
> risk indicator, not observed cases**: Aedes aegypti transmission is
> favoured at 24–33 °C with relative humidity ≥ 60% and recent rainfall that
> creates breeding sites. HIGH when temp_mean is 24–33 °C **and**
> humidity_mean ≥ 60% **and** 7-day rainfall ≥ 10 mm. The classifier learns
> the weather → risk mapping from this definition.

## Dengue-prone areas (15 Patna localities)

The city-wide dengue model says *when* risk is high; this table says
*where*. Each locality gets a base score (0–100) from real, verifiable
factors, and today's weather moves it up or down:

```
elev_score  = clip((56 - elevation_m) / 8 * 100, 0, 100)   # lower ground holds stagnant water
dist_score  = clip((6 - dist_ganga_km) / 6 * 100, 0, 100)  # closer to the river floodplain
water_score = 100 if waterlogging-prone else 30
base_score  = 0.35*elev_score + 0.30*dist_score + 0.35*water_score
final_score = 0.6*base_score + 0.4*(today's HIGH-risk probability * 100)
level: >= 60 High, 35–60 Moderate, < 35 Low
```

Top 5 by base score: Kankarbagh 73.0 (High), Ashok Rajpath 65.2 (High),
Patna City 61.6 (High), Sampatchak 61.2 (High), Patrakar Nagar 61.1 (High).
Overall: 5 High, 8 Moderate, 2 Low (Boring Road 29.0, Khagaul 10.5).

Sources: coordinates from Nominatim (OpenStreetMap), except Patna City from
the Open-Meteo geocoding API; elevation from the Open-Meteo elevation API;
distance to the Ganga = minimum distance to a 5-point polyline along the
river's south bank (Digha Ghat, Kurji, Gandhi Ghat, NIT Ghat, Gaighat);
waterlogging flag = widely reported monsoon waterlogging (notably the 2019
Patna floods). This is a risk index from documented factors — not observed
case counts per locality, which are not published.

## How to run

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/build_dataset.py   # fetch + build data/patna_aqi_dengue.csv
.venv/bin/python src/build_areas.py     # build data/patna_areas.csv
.venv/bin/python src/eda.py            # stats -> outputs/eda_summary.csv, plots -> outputs/figures/
.venv/bin/python src/train.py          # train + evaluate, models -> models/
.venv/bin/streamlit run app.py         # UI at http://localhost:8501
```

On Render the build command is
`pip install -r requirements.txt && python src/train.py && python src/eda.py`,
so the models and figures are generated at deploy time (they are build
artifacts, not stored in the repo); the start command is
`streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.

## Pre-processing

- Missing values: the 215 days before CAMS data starts were dropped; any
  small remaining gaps were linearly interpolated. Final dataset: 0 missing.
- Outliers: IQR rule (values outside Q1 − 1.5·IQR / Q3 + 1.5·IQR) — counted
  in EDA, clipped as a data step using bounds fit on train only (saved in
  `outputs/clip_bounds.json`, reused by the app at prediction time). The
  saved models contain only plain sklearn objects, so they load in any
  process.
- Scaling: `StandardScaler`, fit on train only, applied to test.
- Split: **by time** — first 80% train (4 Aug 2022 – 7 Jul 2024, 704 rows),
  last 20% test (8 Jul – 30 Dec 2024, 176 rows). A random split would leak
  the future into training.

## EDA

Descriptive statistics (mean, variance, std, min, percentiles, skewness,
kurtosis) are in `outputs/eda_summary.csv`. Five plots in
`outputs/figures/`: AQI time series, AQI histogram, PM2.5/PM10 boxplot,
PM2.5-vs-AQI scatter coloured by dengue risk, correlation heatmap.

Key observations: mean AQI 122 (median 86, max 385 — winter spikes);
rainfall is strongly right-skewed (skewness 3.41); PM2.5 correlates 0.98
with same-day AQI (AQI is computed from PM2.5/PM10); tomorrow's AQI
correlates 0.91 with today's, so persistence is a strong baseline the model
must beat.

## Models & results (test set, 176 days)

| Task | Model | Metric | Value |
|---|---|---|---|
| AQI (regression) | Linear Regression | MSE | 1,329.69 |
| | | **RMSE** | **36.46** |
| | | **R²** | **0.878** |
| Dengue (classification) | Logistic Regression | Accuracy | 0.989 |
| | | Precision | 1.000 |
| | | Recall | 0.977 |
| | | F1-score | 0.988 |
| | | Confusion matrix [[89, 0], [2, 85]] | |

**Reading the AQI result honestly:** R² = 0.878 is strong — the model explains
most of next-day AQI variation. But persistence is a tough baseline here
(tomorrow's AQI correlates 0.91 with today's): predicting "tomorrow = today"
gives RMSE 38.43, so the model is only **~2 AQI points better** than that
naive guess. Polynomial Regression (degree 2) was tried and was far worse
(R² = −0.45, overfitting). Adding lag features (AQI from 2/3/7 days ago,
7-day rolling mean) gave a small gain to R² = 0.896; it is noted as future
work because it complicates the UI inputs. Linear Regression is the best of
the listed models for this task.

The dengue classifier scores ~0.99 because its label is defined by a simple
weather rule — the model recovers that boundary almost exactly (2 missed
HIGH days out of 87). That validates the pipeline; it is not a claim about
predicting real case counts.

## UI — theme modelled on aqi.in

`app.py` (Streamlit) follows the aqi.in look: white background, dark text
#31343D, primary blue #4BA9FF, pale lavender panels #F7F7FE, soft rounded
cards (16–20px radius), Poppins-style sans-serif. The Predict tab is a
dashboard like aqi.in's homepage: a hero with a huge category-coloured AQI
number fading into a pink–purple gradient, a weather card, a 6-segment AQI
scale bar with a marker dot, "Major Air Pollutants" cards with a coloured
left border, a dengue-risk card, and an AQI Scale section with health advice
per band. The AQI bands stay the Indian CPCB bands (the data's own
standard); only their colours follow aqi.in's six-colour scale in order
(green #59B61F, yellow #EEC732, orange #EA8C34, pink-red #E95478, purple
#B33FBA, red #C92033). Today's measurements live in the sidebar and both
predictions update live. Four tabs: **Predict tomorrow**, **Dengue-prone
areas** (interactive map + ranked table of 15 localities, final scores
driven by today's weather), **Dataset & EDA** (summary, statistics, all
plots), **Model results** (metrics + actual-vs-predicted chart). The app
forces Streamlit's light theme (`.streamlit/config.toml`) so it stays
readable even when the PC is in dark mode, and all secondary text uses dark
slate #5A6B7B instead of pale grey for contrast on white.

## Live data in the app

The training CSV ends 30 Dec 2024, so the app does not use it as "today".
On load it fetches the latest complete day from Open-Meteo (ERA5 weather +
CAMS air quality, free, no key; cached for 1 hour) and uses those values as
the sidebar defaults, aggregating hourly air quality to daily mean/max and
computing AQI with the same CPCB formula as `src/build_dataset.py`. The page
shows the input date and the forecast date next to the numbers (e.g.
measurements of 11 Nov 2026 → forecast for 12 Nov 2026). When the fetch
fails it falls back to the last training row (30 Dec 2024) and says so on
the page instead of pretending the data is live.

Live validation (12 Nov 2026): feeding the real 10 Nov 2026 measurements
into the trained model predicted 190 for 11 Nov; the actual 11 Nov AQI from
the same source was 208 (error −18, inside the test RMSE of 36). Note these
are CAMS values — CPCB ground stations in Patna read higher (~350–400 on
11 Nov), so the model is accurate against its own data source, not against
ground stations.

## Files

```
patna-aqi-dengue-ml/
  app.py                    Streamlit UI
  requirements.txt
  README.md
  data/raw/                 raw Open-Meteo JSON (weather + air quality)
  data/patna_aqi_dengue.csv built dataset (880 x 17)
  src/build_dataset.py      fetch, merge, AQI formula, dengue label
  src/build_areas.py        15 Patna localities, Ganga distance, risk index
  data/patna_areas.csv      locality table (15 rows)
  src/eda.py                descriptive stats + 5 plots
  src/train.py              both models, metrics, saved models
  models/                   linear_aqi.joblib, logistic_dengue.joblib
  outputs/eda_summary.csv   outputs/metrics.json
  outputs/test_predictions.csv
  outputs/figures/          5 PNG plots
```
