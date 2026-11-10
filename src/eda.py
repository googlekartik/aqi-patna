"""Exploratory Data Analysis for the Patna AQI + dengue dataset.

Outputs:
- outputs/eda_summary.csv — descriptive statistics per numeric column
  (count, mean, variance, std, min, percentiles, skewness, kurtosis)
- outputs/figures/*.png — histogram, boxplot, scatter, correlation heatmap,
  plus an AQI time-series line plot.
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASE = str(Path(__file__).resolve().parent.parent)
NAVY = "#0F2A44"
TEAL = "#1B6CA8"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#B9C6D3",
                     "axes.labelcolor": NAVY, "text.color": NAVY,
                     "xtick.color": NAVY, "ytick.color": NAVY})
sns.set_style("whitegrid")


def main():
    df = pd.read_csv(f"{BASE}/data/patna_aqi_dengue.csv", parse_dates=["date"])
    num = df.select_dtypes(include="number")
    os.makedirs(f"{BASE}/outputs/figures", exist_ok=True)

    print("rows:", len(df), "columns:", len(df.columns))
    print("\ndata types:\n", df.dtypes.to_string())
    print("\nmissing values:\n", df.isna().sum().to_string())

    stats = pd.DataFrame({
        "count": num.count(),
        "mean": num.mean(),
        "variance": num.var(),
        "std": num.std(),
        "min": num.min(),
        "p25": num.quantile(0.25),
        "median": num.median(),
        "p75": num.quantile(0.75),
        "max": num.max(),
        "skewness": num.skew(),
        "kurtosis": num.kurtosis(),
    }).round(3)
    stats.to_csv(f"{BASE}/outputs/eda_summary.csv")
    print("\ndescriptive statistics:\n", stats.to_string())

    # Outlier counts (IQR rule) per numeric column
    for col in num.columns:
        q1, q3 = num[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        n_out = ((num[col] < q1 - 1.5 * iqr) | (num[col] > q3 + 1.5 * iqr)).sum()
        print(f"outliers (IQR) {col}: {n_out}")

    # 1. Histogram — AQI distribution
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(df["aqi"], bins=30, color=TEAL, edgecolor="white")
    ax.axvline(df["aqi"].mean(), color=NAVY, linestyle="--",
               label=f"mean = {df['aqi'].mean():.0f}")
    ax.set_xlabel("AQI (today)"); ax.set_ylabel("Days")
    ax.set_title("Distribution of daily AQI, Patna (Aug 2022 – Dec 2024)")
    ax.legend()
    fig.tight_layout(); fig.savefig(f"{BASE}/outputs/figures/fig_hist_aqi.png", dpi=150)
    plt.close(fig)

    # 2. Boxplot — PM2.5 and PM10 (same units, ug/m3)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(data=df[["pm2_5_mean", "pm10_mean"]], ax=ax,
                palette=[TEAL, NAVY])
    ax.set_ylabel("ug/m3 (daily mean)")
    ax.set_title("Boxplot: PM2.5 and PM10 daily means")
    fig.tight_layout(); fig.savefig(f"{BASE}/outputs/figures/fig_box_pollutants.png", dpi=150)
    plt.close(fig)

    # 3. Scatter — PM2.5 vs AQI, coloured by dengue risk
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for risk, color in [("LOW", TEAL), ("HIGH", "#C0392B")]:
        sub = df[df["dengue_risk"] == risk]
        ax.scatter(sub["pm2_5_mean"], sub["aqi"], s=14, alpha=0.6,
                   color=color, label=f"dengue risk {risk}")
    ax.set_xlabel("PM2.5 daily mean (ug/m3)"); ax.set_ylabel("AQI")
    ax.set_title("PM2.5 vs AQI, coloured by dengue-risk label")
    ax.legend()
    fig.tight_layout(); fig.savefig(f"{BASE}/outputs/figures/fig_scatter_pm25_aqi.png", dpi=150)
    plt.close(fig)

    # 4. Correlation heatmap
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="Blues", ax=ax,
                annot_kws={"size": 8})
    ax.set_title("Correlation heatmap — all numeric features")
    fig.tight_layout(); fig.savefig(f"{BASE}/outputs/figures/fig_corr_heatmap.png", dpi=150)
    plt.close(fig)

    # 5. AQI time series
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["date"], df["aqi"], color=TEAL, linewidth=0.8)
    ax.set_xlabel("Date"); ax.set_ylabel("AQI")
    ax.set_title("Daily AQI in Patna over time")
    fig.tight_layout(); fig.savefig(f"{BASE}/outputs/figures/fig_aqi_timeseries.png", dpi=150)
    plt.close(fig)
    print("\nEDA done — figures in outputs/figures/")


if __name__ == "__main__":
    main()
