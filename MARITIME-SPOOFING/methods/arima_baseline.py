"""
Statistical-forecasting baseline for vessel trajectory reconstruction -- the
maritime analog of methods/ARIMA.ipynb (read-only reference; fresh
implementation). Independent ARIMA(1,1,1) per coordinate (lat, lon), fit on
the observed half of a phase and forecast forward over the spoofed half --
same univariate short-horizon-trend-extrapolation baseline used in the
aircraft study.
"""

import warnings

import numpy as np
from statsmodels.tsa.arima.model import ARIMA

from kalman_baseline import haversine_km

warnings.filterwarnings("ignore", module="statsmodels")


def fit_arima_and_forecast(series: np.ndarray, steps: int, order=(1, 1, 1)) -> np.ndarray:
    try:
        model = ARIMA(series, order=order)
        fit = model.fit()
        return fit.forecast(steps=steps)
    except Exception:
        # Fall back to last-value hold if ARIMA fails to converge on a short series
        return np.full(steps, series[-1])


def arima_predict_phase(seq: list[dict]):
    lats = np.array([p["latitude"] for p in seq], dtype=float)
    lons = np.array([p["longitude"] for p in seq], dtype=float)

    T = len(seq)
    mid = T // 2
    steps = T - mid

    pred_lats = fit_arima_and_forecast(lats[:mid], steps)
    pred_lons = fit_arima_and_forecast(lons[:mid], steps)

    preds = np.stack([pred_lats, pred_lons], axis=1)
    gt = np.stack([lats[mid:], lons[mid:]], axis=1)
    return preds, gt


def evaluate_phase(sequences: list[list[dict]]) -> list[float]:
    errors = []
    for seq in sequences:
        if len(seq) < 4:
            continue
        try:
            preds, gt = arima_predict_phase(seq)
        except Exception:
            continue
        err = np.mean([haversine_km(gt[i][0], gt[i][1], preds[i][0], preds[i][1]) for i in range(len(preds))])
        errors.append(float(err))
    return errors
