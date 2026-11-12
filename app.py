"""Streamlit UI — AQI Prediction & Dengue-Prone Areas in Patna.

Theme modelled on https://www.aqi.in (light dashboard style):
white background, dark text #31343D, primary blue #4BA9FF, pale lavender
panels #F7F7FE, soft rounded cards, a big category-coloured AQI number,
a 6-segment AQI scale bar, and pollutant cards with a coloured left border.
The AQI bands stay the Indian CPCB bands (the data's own standard); only
their colours follow aqi.in's six-colour scale in order.

Live data: on load the app fetches the latest complete day from Open-Meteo
(ERA5 weather + CAMS air quality, free, no key) and uses it as the sidebar
defaults, so the forecast is always for tomorrow from real latest data.
When the fetch fails it falls back to the last training row (30 Dec 2024)
and says so on the page.

Run:  streamlit run app.py
Then open the local URL it prints (default http://localhost:8501).
"""
import json
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import joblib
import pandas as pd
import pydeck as pdk
import streamlit as st

BASE = Path(__file__).resolve().parent
BLUE = "#4BA9FF"
DARK = "#31343D"
GREY = "#5A6B7B"
PANEL = "#F7F7FE"
BORDER = "#E8EBF3"
PURPLE_MID = "#D799DA"
PURPLE_DEEP = "#B754BD"

FEATURES = ["temp_max", "temp_min", "temp_mean", "humidity_mean",
            "precipitation_sum", "wind_speed_max", "pm2_5_mean", "pm2_5_max",
            "pm10_mean", "pm10_max", "no2_mean", "o3_mean", "aqi", "rain_7d"]
LABELS = {
    "temp_max": "Temperature max (°C)", "temp_min": "Temperature min (°C)",
    "temp_mean": "Temperature mean (°C)", "humidity_mean": "Humidity mean (%)",
    "precipitation_sum": "Rainfall today (mm)", "wind_speed_max": "Wind speed max (km/h)",
    "pm2_5_mean": "PM2.5 mean (µg/m³)", "pm2_5_max": "PM2.5 max (µg/m³)",
    "pm10_mean": "PM10 mean (µg/m³)", "pm10_max": "PM10 max (µg/m³)",
    "no2_mean": "NO₂ mean (µg/m³)", "o3_mean": "O₃ mean (µg/m³)",
    "aqi": "AQI today", "rain_7d": "Rainfall, last 7 days (mm)",
}
# CPCB bands; colours follow aqi.in's six-colour scale in order.
AQI_BANDS = [
    ("Good", 0, 50, "#59B61F",
     "Air is fresh and clean. Enjoy outdoor activities."),
    ("Satisfactory", 51, 100, "#EEC732",
     "Acceptable for most people. Sensitive individuals may feel mild discomfort."),
    ("Moderate", 101, 200, "#EA8C34",
     "Breathing may be slightly uncomfortable, especially for people with respiratory issues."),
    ("Poor", 201, 300, "#E95478",
     "Everyone may feel the effects. Sensitive groups should limit outdoor activity."),
    ("Very Poor", 301, 400, "#B33FBA",
     "Health alert — serious effects possible for everyone. Avoid outdoor activity."),
    ("Severe", 401, 500, "#C92033",
     "Emergency conditions. The entire population is likely to be affected. Stay indoors."),
]
LEVEL_COLOR = {"High": "#C92033", "Moderate": "#EEC732", "Low": "#59B61F"}
LEVEL_RGB = {"High": [201, 32, 51], "Moderate": [238, 199, 50], "Low": [89, 182, 31]}
LEVEL_TEXT = {"High": "white", "Moderate": DARK, "Low": "white"}

PM25_BP = [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200),
           (91, 120, 201, 300), (121, 250, 301, 400), (251, 1000, 401, 500)]
PM10_BP = [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200),
           (251, 350, 201, 300), (351, 430, 301, 400), (431, 2000, 401, 500)]
NO2_BP = [(0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200),
          (181, 280, 201, 300), (281, 400, 301, 400), (401, 1000, 401, 500)]
O3_BP = [(0, 50, 0, 50), (51, 100, 51, 100), (101, 168, 101, 200),
         (169, 208, 201, 300), (209, 748, 301, 400), (749, 2000, 401, 500)]


def sub_index(conc, breakpoints):
    for _lo, c_high, a_low, a_high in breakpoints:
        if conc <= c_high:
            c_low = _lo
            return round(((a_high - a_low) / (c_high - c_low)) * (conc - c_low) + a_low)
    return 500


def aqi_category(aqi):
    for name, lo, hi, color, _desc in AQI_BANDS:
        if lo <= aqi <= hi:
            return name, color
    return "Severe", "#C92033"


# ---- Live data: latest complete day from Open-Meteo (free, no key) ----
# The training CSV ends 30 Dec 2024, so without this the sidebar would
# always show that old day as "today". The fetch runs at most once an hour
# (st.cache_data) and falls back to the last training row when it fails.
WEATHER_URL = (
    "https://api.open-meteo.com/v1/forecast?latitude=25.5941&longitude=85.1376"
    "&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
    "relative_humidity_2m_mean,precipitation_sum,wind_speed_10m_max"
    "&timezone=Asia%2FKolkata&past_days=7&forecast_days=0")
AIRQ_URL = (
    "https://air-quality-api.open-meteo.com/v1/air-quality?latitude=25.5941&longitude=85.1376"
    "&hourly=pm2_5,pm10,nitrogen_dioxide,ozone"
    "&timezone=Asia%2FKolkata&past_days=7&forecast_days=0")


def _get_json(url):
    with urllib.request.urlopen(url, timeout=15) as r:
        return json.load(r)


@st.cache_data(ttl=3600)
def fetch_latest_day():
    """Latest complete day of Patna measurements from Open-Meteo.

    Returns (values dict with the 14 FEATURES, input date) or None when the
    fetch fails for any reason. Aggregation and the AQI formula match
    src/build_dataset.py exactly, so a fetched row is directly comparable
    with the training data.
    """
    try:
        w = _get_json(WEATHER_URL)["daily"]
        a = _get_json(AIRQ_URL)["hourly"]
    except Exception:
        return None
    try:
        weather = pd.DataFrame({
            "date": pd.to_datetime(w["time"]),
            "temp_max": w["temperature_2m_max"],
            "temp_min": w["temperature_2m_min"],
            "temp_mean": w["temperature_2m_mean"],
            "humidity_mean": w["relative_humidity_2m_mean"],
            "precipitation_sum": w["precipitation_sum"],
            "wind_speed_max": w["wind_speed_10m_max"],
        })
        airq = pd.DataFrame({
            "datetime": pd.to_datetime(a["time"]),
            "pm2_5": a["pm2_5"], "pm10": a["pm10"],
            "no2": a["nitrogen_dioxide"], "o3": a["ozone"],
        })
        airq["date"] = airq["datetime"].dt.date
        daily_air = airq.groupby("date").agg(
            pm2_5_mean=("pm2_5", "mean"), pm2_5_max=("pm2_5", "max"),
            pm10_mean=("pm10", "mean"), pm10_max=("pm10", "max"),
            no2_mean=("no2", "mean"), o3_mean=("o3", "mean")).reset_index()
        daily_air["date"] = pd.to_datetime(daily_air["date"])
        merged = weather.merge(daily_air, on="date", how="inner")
        merged = merged.dropna(subset=["pm2_5_mean", "pm10_mean"])
        if merged.empty:
            return None
        row = merged.iloc[-1]
        values = {
            "temp_max": float(row["temp_max"]), "temp_min": float(row["temp_min"]),
            "temp_mean": float(row["temp_mean"]),
            "humidity_mean": float(row["humidity_mean"]),
            "precipitation_sum": float(row["precipitation_sum"]),
            "wind_speed_max": float(row["wind_speed_max"]),
            "pm2_5_mean": float(row["pm2_5_mean"]),
            "pm2_5_max": float(row["pm2_5_max"]),
            "pm10_mean": float(row["pm10_mean"]),
            "pm10_max": float(row["pm10_max"]),
            "no2_mean": float(row["no2_mean"]), "o3_mean": float(row["o3_mean"]),
            "aqi": max(sub_index(row["pm2_5_mean"], PM25_BP),
                       sub_index(row["pm10_mean"], PM10_BP)),
            "rain_7d": float(merged["precipitation_sum"].sum()),
        }
        return values, row["date"].date()
    except Exception:
        return None


def risk_level(score):
    return "High" if score >= 60 else ("Moderate" if score >= 35 else "Low")


def pill(text, color, text_color="white"):
    return (f'<span style="background:{color};color:{text_color};border-radius:20px;'
            f'padding:3px 12px;font-weight:600;font-size:13px;">{text}</span>')


@st.cache_resource
def load_models():
    reg = joblib.load(BASE / "models" / "linear_aqi.joblib")
    clf = joblib.load(BASE / "models" / "logistic_dengue.joblib")
    return reg, clf


@st.cache_data
def load_data():
    df = pd.read_csv(BASE / "data" / "patna_aqi_dengue.csv", parse_dates=["date"])
    metrics = json.loads((BASE / "outputs" / "metrics.json").read_text())
    eda = pd.read_csv(BASE / "outputs" / "eda_summary.csv", index_col=0)
    areas = pd.read_csv(BASE / "data" / "patna_areas.csv")
    bounds = json.loads((BASE / "outputs" / "clip_bounds.json").read_text())
    return df, metrics, eda, areas, bounds


st.set_page_config(page_title="AQI", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');
    .stApp { background: #FFFFFF; font-family: 'Poppins', 'Segoe UI', system-ui, sans-serif; }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: #31343D !important; font-family: 'Poppins', 'Segoe UI', system-ui, sans-serif; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; color: #31343D !important; }
    .stTabs [aria-selected="true"] { color: #4BA9FF !important; }
    [data-testid="stWidgetLabel"] { color: #31343D !important; }
    .stNumberInput input { color: #31343D !important; }
    [data-testid="stSidebar"] p { color: #31343D !important; }
    .stButton>button { background: #4BA9FF; color: white; border-radius: 12px;
        border: none; font-weight: 600; }
    .stButton>button:hover { background: #3D8FE0; color: white; }
    section[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E8EBF3; }
    a { color: #4BA9FF; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    [data-testid="stAppDeployButton"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True)

# ---- Top header: the project is called AQI ----
st.markdown(
    f"""<div style="display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;">
    <span style="color:{BLUE};font-size:46px;font-weight:800;letter-spacing:1px;line-height:1;">AQI</span>
    <span style="color:{DARK};font-size:17px;font-weight:600;">AQI Prediction &amp; Dengue-Prone Areas in Patna</span>
    </div>""",
    unsafe_allow_html=True)

df, metrics, eda, areas, bounds = load_data()
reg, clf = load_models()

# ---- Sidebar: measurements for the latest day (shared by both predictions) ----
live = fetch_latest_day()
if live is not None:
    defaults, input_date = live[0], live[1]
    live_note = f"Live data from Open-Meteo for {input_date:%d %b %Y}."
    is_live = True
else:
    defaults = df.iloc[-1]
    input_date = df["date"].max().date()
    live_note = (f"Live data unavailable — showing {input_date:%d %b %Y} "
                 "(last training day).")
    is_live = False
forecast_date = input_date + timedelta(days=1)

st.sidebar.markdown("### Measurements")
st.sidebar.write(live_note + " Both predictions update live — edit any value below.")
medians = df[FEATURES].median()
values = {}
for feat in FEATURES:
    values[feat] = st.sidebar.number_input(
        LABELS[feat], value=float(defaults[feat]),
        help=f"Typical value: {medians[feat]:.1f}")

row = pd.DataFrame([values])
for feat in FEATURES:
    lo, hi = bounds[feat]
    row[feat] = row[feat].clip(lower=lo, upper=hi)
aqi_next = float(reg.predict(row[FEATURES])[0])
prob = float(clf.predict_proba(row[FEATURES])[0][1])
risk = "HIGH" if prob >= 0.5 else "LOW"
cat, cat_color = aqi_category(aqi_next)
cat_desc = next(d for n, _lo, _hi, _c, d in AQI_BANDS if n == cat)
last_data_date = df["date"].max().date()

tab_predict, tab_areas, tab_data, tab_results = st.tabs(
    ["Predict tomorrow", "Dengue-prone areas", "Dataset & EDA", "Model results"])

with tab_predict:
    # ---- Hero: big AQI number + weather card (aqi.in style) ----
    hero_left = (
        f'<div style="color:{GREY};font-size:13px;font-weight:700;'
        f'letter-spacing:1.5px;">TOMORROW&#8217;S AQI FORECAST</div>'
        f'<div style="color:{cat_color};font-size:96px;font-weight:800;line-height:1;">'
        f'{aqi_next:.0f}</div>'
        f'<div style="color:{DARK};font-size:15px;font-weight:600;">AQI &middot; Forecast for '
        f'{forecast_date:%d %b %Y} &middot; Patna, Bihar, India</div>'
        f'<div style="margin-top:10px;color:{DARK};font-size:16px;">Air Quality is '
        f'<span style="background:{cat_color}22;color:{cat_color};border-radius:20px;'
        f'padding:3px 14px;font-weight:700;">{cat}</span></div>'
        f'<div style="color:{GREY};font-size:14px;margin-top:10px;">{cat_desc}</div>'
        f'<div style="color:{DARK};font-size:14px;margin-top:12px;">'
        f'PM2.5: <b>{values["pm2_5_mean"]:.0f} µg/m³</b> &nbsp;&middot;&nbsp; '
        f'PM10: <b>{values["pm10_mean"]:.0f} µg/m³</b></div>'
        f'<div style="color:{GREY};font-size:12px;margin-top:4px;">'
        f'Forecast from measurements of {input_date:%d %b %Y} &middot; '
        f'{"live data: Open-Meteo (CAMS)" if is_live else f"training data through {last_data_date} (live data unavailable)"}</div>')
    hero_right = (
        f'<div style="background:white;border:1px solid {BORDER};border-radius:20px;'
        f'padding:20px 22px;box-shadow:0 4px 18px rgba(49,52,61,0.06);">'
        f'<div style="color:{DARK};font-size:15px;font-weight:700;">Weather in Patna on {input_date:%d %b}</div>'
        f'<div style="color:{DARK};font-size:40px;font-weight:800;">{values["temp_mean"]:.0f}°C</div>'
        f'<div style="color:{GREY};font-size:14px;">Humidity {values["humidity_mean"]:.0f}% &middot; '
        f'Wind {values["wind_speed_max"]:.0f} km/h<br>'
        f'Rainfall today {values["precipitation_sum"]:.1f} mm &middot; '
        f'last 7 days {values["rain_7d"]:.0f} mm</div></div>')
    st.markdown(
        f"""<div style="background:linear-gradient(180deg,#FFFFFF 0%,#F6E3F7 55%,{PURPLE_MID} 100%);
        border-radius:24px;padding:28px 30px;">
        <div style="display:flex;gap:28px;flex-wrap:wrap;">
        <div style="flex:2;min-width:280px;">{hero_left}</div>
        <div style="flex:1;min-width:260px;">{hero_right}</div>
        </div></div>""",
        unsafe_allow_html=True)
    st.write("")

    # ---- AQI scale bar with marker ----
    pct = max(0.0, min(100.0, aqi_next / 500 * 100))
    names = "".join(
        f'<div style="flex:{hi - lo + 1};text-align:center;font-size:12px;font-weight:700;'
        f'color:{DARK};">{n}</div>' for n, lo, hi, _c, _d in AQI_BANDS)
    bar = "".join(
        f'<div style="flex:{hi - lo + 1};background:{c};height:14px;"></div>'
        for _n, lo, hi, c, _d in AQI_BANDS)
    ranges = "".join(
        f'<div style="flex:{hi - lo + 1};text-align:center;font-size:11px;color:{GREY};">{lo}–{hi}</div>'
        for _n, lo, hi, _c, _d in AQI_BANDS)
    st.markdown(
        f"""<div style="margin-top:4px;">{names}
        <div style="position:relative;margin-top:6px;">
        <div style="margin-left:calc({pct:.1f}% - 7px);width:14px;height:14px;border-radius:50%;
        background:{DARK};border:2px solid white;box-shadow:0 0 0 2px {DARK};"></div></div>
        <div style="display:flex;border-radius:8px;overflow:hidden;">{bar}</div>
        <div style="display:flex;margin-top:4px;">{ranges}</div></div>""",
        unsafe_allow_html=True)
    st.write("")

    # ---- Major air pollutants (aqi.in style cards) ----
    st.markdown("## Major Air Pollutants")
    pollutants = [
        ("PM2.5", "Fine Particulate Matter", values["pm2_5_mean"], "µg/m³",
         sub_index(values["pm2_5_mean"], PM25_BP)),
        ("PM10", "Coarse Particulate Matter", values["pm10_mean"], "µg/m³",
         sub_index(values["pm10_mean"], PM10_BP)),
        ("NO₂", "Nitrogen Dioxide", values["no2_mean"], "µg/m³",
         sub_index(values["no2_mean"], NO2_BP)),
        ("O₃", "Ozone", values["o3_mean"], "µg/m³",
         sub_index(values["o3_mean"], O3_BP)),
    ]
    cols = st.columns(2)
    for i, (code, name, val, unit, sub) in enumerate(pollutants):
        sub_cat, sub_color = aqi_category(sub)
        with cols[i % 2]:
            st.markdown(
                f"""<div style="background:white;border:1px solid {BORDER};border-left:5px solid {sub_color};
                border-radius:16px;padding:16px 20px;margin-bottom:12px;
                box-shadow:0 4px 18px rgba(49,52,61,0.05);">
                <div style="display:flex;justify-content:space-between;align-items:baseline;">
                <div><span style="color:{DARK};font-size:18px;font-weight:700;">{code}</span>
                <span style="color:{GREY};font-size:13px;"> &nbsp;{name}</span></div>
                <div style="color:{sub_color};font-size:13px;font-weight:700;">{sub_cat}</div></div>
                <div style="color:{DARK};font-size:30px;font-weight:800;">{val:.1f}
                <span style="font-size:14px;font-weight:600;color:{GREY};"> {unit}</span></div>
                <div style="color:{GREY};font-size:12px;">Sub-index: {sub}</div></div>""",
                unsafe_allow_html=True)
    st.write("")

    # ---- Dengue risk card ----
    risk_color = "#C92033" if risk == "HIGH" else "#59B61F"
    st.markdown(
        f"""<div style="background:white;border:1px solid {BORDER};border-left:5px solid {risk_color};
        border-radius:16px;padding:18px 22px;box-shadow:0 4px 18px rgba(49,52,61,0.05);">
        <div style="color:{GREY};font-size:12px;font-weight:700;letter-spacing:1px;">
        DENGUE OUTBREAK RISK &middot; {forecast_date:%d %b %Y}</div>
        <div style="color:{DARK};font-size:28px;font-weight:800;">{risk}
        <span style="font-size:15px;font-weight:600;color:{GREY};"> &nbsp;({prob:.0%} probability of HIGH)</span></div>
        <div style="color:{GREY};font-size:14px;">Warm, humid days after rain favour Aedes mosquitoes.
        See the <b>Dengue-prone areas</b> tab for where in Patna the risk is highest.</div></div>""",
        unsafe_allow_html=True)
    st.write("")

    # ---- AQI scale section ----
    st.markdown("## Air Quality Index (AQI) Scale")
    st.write("What each band means for your health:")
    scale_rows = "".join(
        f"""<div style="background:{PANEL};border-radius:14px;padding:14px 18px;
        margin-bottom:10px;display:flex;gap:14px;align-items:center;">
        <div style="width:18px;height:18px;border-radius:5px;background:{c};flex-shrink:0;"></div>
        <div style="min-width:170px;"><b style="color:{DARK};">{n}</b>
        <span style="color:{GREY};"> ({lo}–{hi})</span></div>
        <div style="color:{GREY};font-size:14px;">{d}</div></div>"""
        for n, lo, hi, c, d in AQI_BANDS)
    st.markdown(scale_rows, unsafe_allow_html=True)
    st.caption("Dengue risk here is a weather-based risk indicator, not a count of "
               "confirmed cases — daily Patna case counts are not published.")

with tab_areas:
    st.markdown("## Dengue-prone areas in Patna")
    st.write("Each locality gets a **base score (0–100)** from real, verifiable "
             "factors — elevation, distance to the Ganga, and whether the area "
             "is known for monsoon waterlogging. The latest day's weather then moves the "
             "score up or down:")
    st.code("final score = 0.6 × base score + 0.4 × (today's HIGH-risk probability × 100)",
            language="text")
    st.write(f"The HIGH-risk probability from the {input_date:%d %b %Y} measurements is **{prob:.0%}**, "
             f"so areas are ranked by their final score below.")
    areas = areas.copy()
    areas["final_score"] = (0.6 * areas["base_score"] + 0.4 * prob * 100).round(1)
    areas["final_level"] = areas["final_score"].apply(risk_level)
    areas["color"] = areas["final_level"].map(LEVEL_RGB)
    areas["radius"] = areas["final_score"] * 10
    areas = areas.sort_values("final_score", ascending=False)

    layer = pdk.Layer(
        "ScatterplotLayer", data=areas,
        get_position=["longitude", "latitude"],
        get_radius="radius", get_fill_color="color",
        pickable=True, radius_min_pixels=6, radius_max_pixels=20,
        stroked=True, get_line_color=[49, 52, 61], line_width_min_pixels=1)
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(latitude=25.595, longitude=85.13, zoom=11),
        tooltip={"text": "{locality}\nFinal score: {final_score}\nLevel: {final_level}"})
    st.pydeck_chart(deck)
    st.caption("Dot size = final risk score. "
               + pill("High", LEVEL_COLOR["High"]) + " "
               + pill("Moderate", LEVEL_COLOR["Moderate"], DARK) + " "
               + pill("Low", LEVEL_COLOR["Low"]))

    rows_html = ""
    for i, r in enumerate(areas.itertuples()):
        bg = "#FFFFFF" if i % 2 == 0 else PANEL
        rows_html += (
            f'<tr style="background:{bg};"><td style="padding:10px;font-weight:600;color:{DARK};">{r.locality}</td>'
            f'<td style="padding:10px;color:{DARK};">{r.elevation_m:.0f} m</td>'
            f'<td style="padding:10px;color:{DARK};">{r.dist_ganga_km:.1f} km</td>'
            f'<td style="padding:10px;color:{DARK};">{r.waterlogging_prone}</td>'
            f'<td style="padding:10px;color:{DARK};">{r.base_score:.0f}</td>'
            f'<td style="padding:10px;font-weight:700;color:{DARK};">{r.final_score:.0f}</td>'
            f'<td style="padding:10px;">{pill(r.final_level, LEVEL_COLOR[r.final_level], LEVEL_TEXT[r.final_level])}</td></tr>')
    st.markdown(
        f"""<table style="width:100%;background:white;border:1px solid {BORDER};
        border-radius:14px;border-collapse:collapse;overflow:hidden;">
        <tr style="background:{DARK};color:white;">
        <th style="padding:10px;text-align:left;">Locality</th>
        <th style="padding:10px;text-align:left;">Elevation</th>
        <th style="padding:10px;text-align:left;">Distance to Ganga</th>
        <th style="padding:10px;text-align:left;">Waterlogging-prone</th>
        <th style="padding:10px;text-align:left;">Base score</th>
        <th style="padding:10px;text-align:left;">Final score</th>
        <th style="padding:10px;text-align:left;">Risk level</th></tr>
        {rows_html}</table>""",
        unsafe_allow_html=True)
    st.caption("Base score = 0.35 × elevation score + 0.30 × distance score + "
               "0.35 × waterlogging score. Coordinates: Nominatim / Open-Meteo "
               "geocoding; elevation: Open-Meteo elevation API; waterlogging flag: "
               "widely reported monsoon waterlogging (notably the 2019 Patna floods). "
               "This is a risk index from documented factors — not observed case "
               "counts per locality, which are not published.")

with tab_data:
    st.subheader("Dataset summary")
    st.write(f"**Rows:** {len(df)} &nbsp;&nbsp; **Columns:** {len(df.columns)} &nbsp;&nbsp; "
             f"**Period:** {df['date'].min().date()} to {df['date'].max().date()}")
    st.write("**Columns and types:**")
    st.dataframe(pd.DataFrame({"column": df.columns,
                               "dtype": df.dtypes.astype(str)}),
                 hide_index=True)
    st.subheader("Descriptive statistics")
    st.dataframe(eda)
    st.subheader("Plots")
    for name, caption in [
        ("fig_aqi_timeseries", "Daily AQI in Patna over time"),
        ("fig_hist_aqi", "Distribution of daily AQI"),
        ("fig_box_pollutants", "PM2.5 and PM10 daily means"),
        ("fig_scatter_pm25_aqi", "PM2.5 vs AQI, coloured by dengue risk"),
        ("fig_corr_heatmap", "Correlation heatmap"),
    ]:
        st.image(str(BASE / "outputs" / "figures" / f"{name}.png"), caption=caption)

with tab_results:
    st.subheader("AQI forecasting — Linear Regression (test set)")
    st.json(metrics["aqi_regression"])
    st.subheader("Dengue risk — Logistic Regression (test set)")
    st.json(metrics["dengue_classification"])
    st.write("Confusion matrix rows are the true labels (LOW, HIGH); "
             "columns are the predicted labels.")
    preds = pd.read_csv(BASE / "outputs" / "test_predictions.csv",
                        parse_dates=["date"])
    st.subheader("Test predictions — actual vs predicted AQI")
    st.line_chart(preds.set_index("date")[["aqi_next", "aqi_next_pred"]])

st.markdown(
    f"""<div style="background:{PANEL};border-radius:16px;padding:18px 22px;margin-top:24px;">
    <b style="color:{DARK};">AQI Prediction &amp; Dengue-Prone Areas in Patna</b>
    <span style="color:{GREY};"> &middot; Real weather &amp; air-quality data from Open-Meteo &middot;
    AQI by the Indian CPCB formula &middot; Theme inspired by aqi.in</span></div>""",
    unsafe_allow_html=True)
