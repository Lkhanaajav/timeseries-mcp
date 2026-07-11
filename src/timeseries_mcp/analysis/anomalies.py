"""Anomaly detection: global z-score, robust MAD, IQR fences, and STL-residual."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..models import Anomaly, AnomalyReport
from ..store import StoreError, clean_values

MAX_REPORTED = 50

METHODS = ("zscore", "mad", "iqr", "stl_residual")


def detect(
    s: pd.Series,
    series_id: str,
    method: str = "zscore",
    threshold: float = 3.0,
    period: int | None = None,
) -> AnomalyReport:
    if method not in METHODS:
        raise StoreError(f"Unknown method '{method}'. Choose one of {METHODS}.")
    clean = clean_values(s, min_points=8, context="Anomaly detection")
    values = clean.to_numpy()

    if method == "zscore":
        scores, note = _zscore_scores(values)
    elif method == "mad":
        scores, note = _mad_scores(values)
    elif method == "iqr":
        scores, note = _iqr_scores(values, k=threshold)
    else:
        scores, note = _stl_residual_scores(clean, period)

    # For IQR, `threshold` is the fence multiplier itself; any point beyond
    # the fence (score > 0) is anomalous.
    mask = scores > 0 if method == "iqr" else scores > threshold

    order = np.argsort(scores[mask])[::-1]
    idx = np.flatnonzero(mask)[order][:MAX_REPORTED]
    anomalies = [
        Anomaly(
            timestamp=clean.index[i].isoformat(),
            value=float(values[i]),
            score=round(float(scores[i]), 4),
        )
        for i in idx
    ]
    return AnomalyReport(
        series_id=series_id,
        method=method,
        threshold=threshold,
        n_anomalies=int(mask.sum()),
        anomalies=anomalies,
        baseline_mean=float(values.mean()),
        baseline_std=float(values.std(ddof=1)),
        notes=note,
    )


def _zscore_scores(values: np.ndarray) -> tuple[np.ndarray, str]:
    std = values.std(ddof=1)
    if std == 0:
        return np.zeros_like(values), "Series is constant; no anomalies possible."
    return np.abs(values - values.mean()) / std, "Score = |x - mean| / std over the full series."


def _mad_scores(values: np.ndarray) -> tuple[np.ndarray, str]:
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0:
        # Degenerate (>=50% identical values): fall back to std so ties don't hide real spikes.
        std = values.std(ddof=1)
        if std == 0:
            return np.zeros_like(values), "Series is constant; no anomalies possible."
        return (
            np.abs(values - median) / std,
            "MAD was 0 (many identical values); fell back to |x - median| / std.",
        )
    return (
        np.abs(values - median) / (1.4826 * mad),
        "Robust score = |x - median| / (1.4826 * MAD); resistant to outlier contamination.",
    )


def _iqr_scores(values: np.ndarray, k: float) -> tuple[np.ndarray, str]:
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        return np.zeros_like(values), "IQR is 0 (middle 50% identical); method not informative here."
    lower, upper = q1 - k * iqr, q3 + k * iqr
    dist = np.maximum(lower - values, values - upper)
    scores = np.maximum(dist / iqr, 0.0)
    return scores, f"Tukey fences at Q1-{k}*IQR / Q3+{k}*IQR; score = distance beyond fence in IQR units."


def _stl_residual_scores(clean: pd.Series, period: int | None) -> tuple[np.ndarray, str]:
    from statsmodels.tsa.seasonal import STL

    if period is None:
        raise StoreError(
            "stl_residual needs `period` (observations per season), e.g. 288 for daily "
            "seasonality at 5-minute sampling. Try `autocorrelation` to find one."
        )
    if period < 2 or len(clean) < 2 * period:
        raise StoreError(
            f"stl_residual needs period >= 2 and at least 2*period points "
            f"(period={period}, points={len(clean)})."
        )
    resid = STL(clean.to_numpy(), period=period, robust=True).fit().resid
    std = resid.std(ddof=1)
    if std == 0:
        return np.zeros_like(resid), "Residuals are constant; no anomalies detected."
    return (
        np.abs(resid - resid.mean()) / std,
        f"Z-score of robust STL residuals (period={period}); catches seasonal-context anomalies "
        "that a global z-score misses.",
    )
