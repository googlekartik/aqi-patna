"""Train the two models for the Patna AQI + dengue project.

Task A — Regression: predict tomorrow's AQI (aqi_next) with Linear Regression.
Task B — Classification: predict dengue outbreak risk (HIGH / LOW) with
Logistic Regression.

Split is by time (first 80% train, last 20% test) — the correct split for
time-series data; a random split would leak the future into training.
Pre-processing (fit on train only, applied to test):
- IQR outlier clipping per feature, then StandardScaler.
Models are saved as sklearn Pipelines (scaler + model) so the Streamlit app
can reuse the exact same transformation.

Metrics saved to outputs/metrics.json; test predictions to
outputs/test_predictions.csv.
"""
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_squared_error, precision_score, r2_score,
                             recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = str(Path(__file__).resolve().parent.parent)
FEATURES = ["temp_max", "temp_min", "temp_mean", "humidity_mean",
            "precipitation_sum", "wind_speed_max", "pm2_5_mean", "pm2_5_max",
            "pm10_mean", "pm10_max", "no2_mean", "o3_mean", "aqi", "rain_7d"]


def clip_iqr_bounds(X):
    """IQR outlier bounds [Q1 - 1.5*IQR, Q3 + 1.5*IQR] per feature."""
    q1 = X.quantile(0.25)
    q3 = X.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def main():
    # Create the output folders first: on a fresh checkout (e.g. Render)
    # models/ and outputs/ do not exist yet (they are gitignored), and
    # writing clip_bounds.json without them crashes the build.
    os.makedirs(f"{BASE}/outputs", exist_ok=True)
    os.makedirs(f"{BASE}/models", exist_ok=True)
    df = pd.read_csv(f"{BASE}/data/patna_aqi_dengue.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    split = int(len(df) * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]
    print(f"train rows: {len(train)} ({train['date'].min().date()} to "
          f"{train['date'].max().date()})")
    print(f"test rows: {len(test)} ({test['date'].min().date()} to "
          f"{test['date'].max().date()})")

    # ---- Outlier clipping (IQR bounds fit on train only) ----
    # Done as a data step, not a pipeline step, and the bounds are saved to
    # outputs/clip_bounds.json so the Streamlit app can apply the same
    # clipping at prediction time. The saved pipelines then contain only
    # plain sklearn objects (scaler + model), which load in any process.
    lo, hi = clip_iqr_bounds(train[FEATURES])
    bounds = {f: [float(lo[f]), float(hi[f])] for f in FEATURES}
    with open(f"{BASE}/outputs/clip_bounds.json", "w") as f:
        json.dump(bounds, f, indent=2)
    X_train = train[FEATURES].clip(lower=lo, upper=hi, axis=1)
    X_test = test[FEATURES].clip(lower=lo, upper=hi, axis=1)

    # ---- Task A: Linear Regression on next-day AQI ----
    reg = Pipeline([("scale", StandardScaler()), ("model", LinearRegression())])
    reg.fit(X_train, train["aqi_next"])
    pred_aqi = reg.predict(X_test)
    mse = mean_squared_error(test["aqi_next"], pred_aqi)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(test["aqi_next"], pred_aqi)
    print(f"\nAQI Linear Regression — test MSE={mse:.2f} RMSE={rmse:.2f} R2={r2:.3f}")
    joblib.dump(reg, f"{BASE}/models/linear_aqi.joblib")

    # ---- Task B: Logistic Regression on dengue risk ----
    y_train = (train["dengue_risk"] == "HIGH").astype(int)
    y_test = (test["dengue_risk"] == "HIGH").astype(int)
    clf = Pipeline([("scale", StandardScaler()),
                    ("model", LogisticRegression(max_iter=1000))])
    clf.fit(X_train, y_train)
    pred_risk = clf.predict(X_test)
    prob_risk = clf.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, pred_risk).tolist()
    metrics = {
        "aqi_regression": {
            "model": "Linear Regression", "target": "aqi_next (tomorrow's AQI)",
            "test_rows": int(len(test)),
            "MSE": round(float(mse), 2), "RMSE": round(rmse, 2),
            "R2": round(float(r2), 3),
        },
        "dengue_classification": {
            "model": "Logistic Regression", "target": "dengue_risk HIGH/LOW",
            "test_rows": int(len(test)),
            "accuracy": round(float(accuracy_score(y_test, pred_risk)), 3),
            "precision": round(float(precision_score(y_test, pred_risk, zero_division=0)), 3),
            "recall": round(float(recall_score(y_test, pred_risk, zero_division=0)), 3),
            "f1": round(float(f1_score(y_test, pred_risk, zero_division=0)), 3),
            "confusion_matrix": cm,
        },
    }
    print("Dengue Logistic Regression — test",
          {k: v for k, v in metrics["dengue_classification"].items()
           if k not in ("model", "target", "test_rows")})
    joblib.dump(clf, f"{BASE}/models/logistic_dengue.joblib")

    out = test[["date", "aqi_next", "dengue_risk"]].copy()
    out["aqi_next_pred"] = np.round(pred_aqi, 1)
    out["dengue_risk_pred"] = np.where(pred_risk == 1, "HIGH", "LOW")
    out["dengue_high_prob"] = np.round(prob_risk, 3)
    out.to_csv(f"{BASE}/outputs/test_predictions.csv", index=False)
    with open(f"{BASE}/outputs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nTraining done — models in models/, metrics in outputs/metrics.json")


if __name__ == "__main__":
    main()
