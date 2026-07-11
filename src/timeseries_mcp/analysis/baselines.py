"""Honest baseline forecasts: naive, seasonal-naive, drift, simple exponential smoothing.

These are the reference methods every fancier model must beat (Hyndman, FPP3).
Every forecast ships with a real holdout backtest so the agent can report how
trustworthy the baseline actually is on this series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..models import BacktestMetrics, ForecastPoint, ForecastReport
from ..store import StoreError, clean_values

METHODS = ("naive", "seasonal_naive", "drift", "ses")
MAX_HORIZON = 500
Z95 = 1.96


def forecast(
    s: pd.Series,
    series_id: str,
    horizon: int = 12,
    method: str = "naive",
    period: int | None = None,
) -> ForecastReport:
    if method not in METHODS:
        raise StoreError(f"Unknown method '{method}'. Choose one of {METHODS}.")
    if not 1 <= horizon <= MAX_HORIZON:
        raise StoreError(f"horizon must be between 1 and {MAX_HORIZON}.")
    if method == "seasonal_naive" and (period is None or period < 2):
        raise StoreError("seasonal_naive needs `period` >= 2 (observations per season).")

    clean = clean_values(s, min_points=10, context="Forecasting")
    if method == "seasonal_naive" and len(clean) < 2 * period:
        raise StoreError(f"seasonal_naive needs at least 2*period points ({2 * period}).")

    values = clean.to_numpy()

    # Backtest on the last 20% (bounded to [1, 100] points), then refit on everything.
    n_test = min(100, max(1, len(values) // 5))
    train, test = values[:-n_test], values[-n_test:]
    if len(train) >= 10 and (method != "seasonal_naive" or len(train) >= 2 * period):
        predictions = _predict(train, len(test), method, period)[0]
        backtest = _metrics(test, predictions)
    else:
        backtest = BacktestMetrics(n_test=0, mae=float("nan"), rmse=float("nan"), mape_pct=None)

    points, sigmas = _predict(values, horizon, method, period)
    future_index = _future_timestamps(clean.index, horizon)
    forecasts = [
        ForecastPoint(
            timestamp=future_index[h].isoformat(),
            value=round(float(points[h]), 6),
            lo=round(float(points[h] - Z95 * sigmas[h]), 6),
            hi=round(float(points[h] + Z95 * sigmas[h]), 6),
        )
        for h in range(horizon)
    ]
    return ForecastReport(
        series_id=series_id,
        method=method,
        horizon=horizon,
        forecasts=forecasts,
        backtest=backtest,
        notes=(
            f"Baseline method; backtest = last {backtest.n_test} points held out. "
            "Intervals assume roughly normal one-step errors. If a sophisticated model "
            "cannot beat this backtest, it is not adding value."
        ),
    )


def _predict(
    values: np.ndarray, horizon: int, method: str, period: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """Return (point forecasts, per-step sigma) for the chosen method."""
    h = np.arange(1, horizon + 1, dtype=float)
    one_step = np.diff(values)
    sigma1 = one_step.std(ddof=1) if len(one_step) > 1 else 0.0

    if method == "naive":
        points = np.full(horizon, values[-1])
        sigmas = sigma1 * np.sqrt(h)
    elif method == "drift":
        slope = (values[-1] - values[0]) / (len(values) - 1)
        points = values[-1] + slope * h
        sigmas = sigma1 * np.sqrt(h * (1 + h / (len(values) - 1)))
    elif method == "seasonal_naive":
        assert period is not None
        last_season = values[-period:]
        points = np.array([last_season[int(k) % period] for k in (h - 1)])
        seasonal_diffs = values[period:] - values[:-period]
        sigma_m = seasonal_diffs.std(ddof=1) if len(seasonal_diffs) > 1 else sigma1
        sigmas = sigma_m * np.sqrt(np.floor((h - 1) / period) + 1)
    else:  # ses
        from statsmodels.tsa.holtwinters import SimpleExpSmoothing

        fit = SimpleExpSmoothing(values, initialization_method="estimated").fit(optimized=True)
        points = fit.forecast(horizon)
        alpha = float(fit.params.get("smoothing_level", 0.5))
        resid_sigma = np.std(fit.resid, ddof=1) if len(fit.resid) > 1 else sigma1
        sigmas = resid_sigma * np.sqrt(1 + (h - 1) * alpha**2)

    return np.asarray(points, dtype=float), np.asarray(sigmas, dtype=float)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> BacktestMetrics:
    err = actual - predicted
    mape = float(np.mean(np.abs(err / actual)) * 100.0) if not np.any(actual == 0) else None
    return BacktestMetrics(
        n_test=len(actual),
        mae=round(float(np.mean(np.abs(err))), 6),
        rmse=round(float(np.sqrt(np.mean(err**2))), 6),
        mape_pct=round(mape, 4) if mape is not None else None,
    )


def _future_timestamps(index: pd.DatetimeIndex, horizon: int) -> pd.DatetimeIndex:
    freq = pd.infer_freq(index)
    if freq is not None:
        return pd.date_range(start=index[-1], periods=horizon + 1, freq=freq)[1:]
    # Irregular index: step forward by the median interval. total_seconds() is
    # resolution-safe (pandas indexes may carry s/us/ns units).
    median_seconds = float(np.median((index[1:] - index[:-1]).total_seconds()))
    median_delta = pd.Timedelta(seconds=median_seconds)
    return pd.DatetimeIndex([index[-1] + median_delta * (k + 1) for k in range(horizon)])
