"""
Primary chronological soil-moisture forecasting benchmark.

Expected input CSV columns:
sequence_id, timestamp, soil_moisture, air_temperature, air_humidity,
air_pressure, dew_point, precipitation, soil_temperature

The script creates lagged features, constructs exact-horizon targets within
each uninterrupted sequence, performs chronological expanding-window testing,
and compares persistence, Ridge, Random Forest and HistGradientBoosting.

Raw source files are intentionally not bundled with this repository.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HORIZONS = [1, 6, 12, 24]
TRAIN_FRACTION = 0.50
RANDOM_STATE = 42

BASE = [
    "soil_moisture", "air_temperature", "air_humidity", "air_pressure",
    "dew_point", "precipitation", "soil_temperature"
]
LAGS = [1, 3, 6, 12, 24]

def load_data(path):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values(["sequence_id", "timestamp"]).reset_index(drop=True)

def engineer_features(df):
    out = df.copy()
    g = out.groupby("sequence_id", group_keys=False)

    for col in BASE:
        for lag in LAGS:
            out[f"{col}_lag_{lag}h"] = g[col].shift(lag)

    out["soil_change_1h"] = g["soil_moisture"].diff(1)
    out["soil_change_6h"] = g["soil_moisture"].diff(6)
    out["precipitation_sum_6h"] = (
        g["precipitation"].rolling(6, min_periods=6).sum()
        .reset_index(level=0, drop=True)
    )
    out["precipitation_sum_24h"] = (
        g["precipitation"].rolling(24, min_periods=24).sum()
        .reset_index(level=0, drop=True)
    )

    hour = out["timestamp"].dt.hour
    doy = out["timestamp"].dt.dayofyear
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return out

def make_horizon_frame(df, horizon):
    out = df.copy()
    g = out.groupby("sequence_id", group_keys=False)
    out["target"] = g["soil_moisture"].shift(-horizon)
    out["target_time"] = g["timestamp"].shift(-horizon)

    elapsed = (out["target_time"] - out["timestamp"]).dt.total_seconds() / 3600
    out = out[elapsed.eq(horizon)].copy()
    out = out.dropna(subset=FEATURES + ["target"])
    return out

FEATURES = [
    "soil_moisture",
    "soil_moisture_lag_1h", "soil_moisture_lag_3h",
    "soil_moisture_lag_6h", "soil_moisture_lag_12h",
    "soil_moisture_lag_24h", "soil_change_1h", "soil_change_6h",
    "air_temperature", "air_humidity", "air_pressure", "dew_point",
    "precipitation", "soil_temperature",
    "air_temperature_lag_1h", "air_temperature_lag_6h",
    "air_temperature_lag_12h", "air_temperature_lag_24h",
    "air_humidity_lag_1h", "air_humidity_lag_6h",
    "air_humidity_lag_12h", "air_humidity_lag_24h",
    "dew_point_lag_1h", "dew_point_lag_6h",
    "dew_point_lag_12h", "dew_point_lag_24h",
    "precipitation_lag_1h", "precipitation_lag_6h",
    "precipitation_lag_12h", "precipitation_lag_24h",
    "soil_temperature_lag_1h", "soil_temperature_lag_6h",
    "soil_temperature_lag_12h", "soil_temperature_lag_24h",
    "precipitation_sum_6h", "precipitation_sum_24h",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos"
]

def evaluate(y, pred):
    return {
        "MAE": mean_absolute_error(y, pred),
        "RMSE": np.sqrt(mean_squared_error(y, pred)),
        "R2": r2_score(y, pred)
    }

def run(path, output_path):
    df = engineer_features(load_data(path))
    rows = []

    for h in HORIZONS:
        d = make_horizon_frame(df, h)
        cut = int(len(d) * TRAIN_FRACTION)
        train = d.iloc[:cut]
        test = d.iloc[cut:]

        Xtr, Xte = train[FEATURES], test[FEATURES]
        ytr, yte = train["target"], test["target"]

        models = {
            "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=10)),
            "RandomForest": RandomForestRegressor(
                n_estimators=200, max_depth=12, min_samples_leaf=5,
                random_state=RANDOM_STATE, n_jobs=-1
            ),
            "HistGradientBoosting": HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                l2_regularization=1, random_state=RANDOM_STATE
            ),
        }

        persistence = evaluate(yte, test["soil_moisture"].to_numpy())
        rows.append({"horizon_hours": h, "model": "Persistence", **persistence})

        for name, model in models.items():
            model.fit(Xtr, ytr)
            rows.append({
                "horizon_hours": h,
                "model": name,
                **evaluate(yte, model.predict(Xte))
            })

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(pd.DataFrame(rows).to_string(index=False))

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("input_csv")
    p.add_argument("output_csv", nargs="?", default="results/primary_results_generated.csv")
    args = p.parse_args()
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    run(args.input_csv, args.output_csv)
