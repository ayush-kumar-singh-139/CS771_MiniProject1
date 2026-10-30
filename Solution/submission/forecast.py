"""CS771 Mini-Project 1 - Part 5 submission (based on starter/forecast_template.py).

Model: average of a scaled ridge regression and an ExtraTrees forest, both predicting the change in
PM2.5 from the issue time to 24 h later, on 57 features in 7 groups (see features.py).
The model was refitted on all 12:00 targets up to 30 Nov 2024 and saved to model.pkl.

features.py in this folder is a byte-for-byte copy of MP1/src/features.py, the same code that built
the training features (the notebook asserts this). Nothing is refitted inside forecast().
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import HISTORY_HOURS, clean_history, predict_residual_ensemble, rich_features

HERE = Path(__file__).resolve().parent

# ---- load fitted objects ONCE, at import time ----------------------------------------------
BUNDLE = joblib.load(HERE / "model.pkl")
MODELS = BUNDLE["models"]
for _m in MODELS:                 # predicting 468 rows: no need for worker processes
    if hasattr(_m, "n_jobs"):
        _m.n_jobs = 1
CTX = BUNDLE["ctx"]
GROUPS = tuple(BUNDLE["groups"])
LEAD = int(BUNDLE["lead"])
STATION_IDS = [str(s) for s in BUNDLE["station_ids"]]
# a few spare hours beyond the 168 h the features need
WINDOW_HOURS = HISTORY_HOURS + 48


def forecast(history: pd.DataFrame, issue_time: pd.Timestamp) -> pd.Series:
    """history: wide hourly table (DatetimeIndex 'timestamp', one string column per sensor_id),
    every hour from 1 Jan 2024 00:00 up to and including issue_time.
    issue_time: 12:00 on some day (IST).
    Returns predicted PM2.5 for issue_time + 24 h, indexed by sensor_id (all 468 stations)."""
    issue_time = pd.Timestamp(issue_time)
    hist = history.copy()
    hist.columns = hist.columns.astype(str)
    hist = hist.loc[:issue_time, STATION_IDS]                       # information cutoff + column order

    # a regular hourly window ending exactly at the issue time (a missing hour becomes NaN and is
    # then forward-filled by clean_history - on gap-free data nothing changes)
    window = pd.date_range(issue_time - pd.Timedelta(hours=WINDOW_HOURS - 1), issue_time, freq="h")
    hist = hist.reindex(window)
    V = clean_history(hist.to_numpy(dtype=float))
    issue_pos = np.array([len(window) - 1])

    X, _, _ = rich_features(V, window, issue_pos, CTX, lead=LEAD, groups=GROUPS)
    persistence = V[-1]
    prediction = predict_residual_ensemble(MODELS, X, persistence)

    # last-resort safety: never return NaN/inf (falls back to persistence for that station)
    bad = ~np.isfinite(prediction)
    prediction[bad] = np.nan_to_num(persistence[bad], nan=0.0)

    return pd.Series(np.clip(prediction, 0, None), index=pd.Index(STATION_IDS, name="sensor_id"),
                     name="pm25_forecast")
