"""Error metrics, reported overall and per validation month."""
import numpy as np
import pandas as pd


def rmse(pred, true):
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def mae(pred, true):
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    return float(np.mean(np.abs(pred - true)))


def metrics_by_month(pred, true, target_times):
    """RMSE and MAE overall and per target month. target_times has one entry per row."""
    months = pd.DatetimeIndex(target_times).strftime("%b").to_numpy()
    out = {"overall": (rmse(pred, true), mae(pred, true))}
    for m in ["Sep", "Oct", "Nov"]:
        sel = months == m
        if sel.any():
            out[m] = (rmse(pred[sel], true[sel]), mae(pred[sel], true[sel]))
    return out


def results_table(preds, true, target_times):
    """preds: dict name -> prediction vector. Returns a table with RMSE/MAE columns per period."""
    rows = {}
    for name, p in preds.items():
        res = metrics_by_month(np.asarray(p, float), true, target_times)
        row = {}
        for period, (r, a) in res.items():
            row[(period, "RMSE")] = r
            row[(period, "MAE")] = a
        rows[name] = row
    T = pd.DataFrame(rows).T
    T.columns = pd.MultiIndex.from_tuples(T.columns)
    return T.round(2)
