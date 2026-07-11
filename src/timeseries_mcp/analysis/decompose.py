"""Seasonal decomposition (STL or classical) with Hyndman strength diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..models import ComponentSummary, DecompositionReport, Point
from ..store import StoreError, clean_values

PREVIEW_POINTS = 30
METHODS = ("stl", "classical")


def decompose(
    s: pd.Series,
    series_id: str,
    period: int,
    method: str = "stl",
) -> DecompositionReport:
    if method not in METHODS:
        raise StoreError(f"Unknown method '{method}'. Choose one of {METHODS}.")
    clean = clean_values(s, min_points=4, context="Decomposition")
    if period < 2:
        raise StoreError(f"period must be >= 2 (got {period}).")
    if len(clean) < 2 * period:
        raise StoreError(
            f"Decomposition needs at least 2 full periods ({2 * period} points for period={period}); "
            f"series has {len(clean)}."
        )

    if method == "stl":
        from statsmodels.tsa.seasonal import STL

        result = STL(clean.to_numpy(), period=period, robust=True).fit()
        trend, seasonal, resid = result.trend, result.seasonal, result.resid
    else:
        from statsmodels.tsa.seasonal import seasonal_decompose

        result = seasonal_decompose(clean.to_numpy(), model="additive", period=period)
        keep = ~np.isnan(result.trend)
        trend, seasonal, resid = result.trend[keep], result.seasonal[keep], result.resid[keep]
        clean = clean[keep]

    trend_strength = _strength(resid, trend + resid)
    seasonal_strength = _strength(resid, seasonal + resid)

    components = [
        _summarize("trend", trend, clean.index),
        _summarize("seasonal", seasonal, clean.index),
        _summarize("residual", resid, clean.index),
    ]
    interpretation = (
        f"Trend strength {trend_strength:.2f}, seasonal strength {seasonal_strength:.2f} "
        f"(0 = absent, 1 = dominant). "
        + (
            "Strong seasonality — seasonal-aware methods (stl_residual anomalies, "
            "seasonal_naive forecasts) are appropriate."
            if seasonal_strength >= 0.6
            else "Weak-to-moderate seasonality at this period."
        )
    )
    return DecompositionReport(
        series_id=series_id,
        method=method,
        period=period,
        trend_strength=round(trend_strength, 4),
        seasonal_strength=round(seasonal_strength, 4),
        components=components,
        interpretation=interpretation,
    )


def _strength(resid: np.ndarray, combined: np.ndarray) -> float:
    var_combined = combined.var(ddof=1)
    if var_combined == 0:
        return 0.0
    return max(0.0, 1.0 - resid.var(ddof=1) / var_combined)


def _summarize(name: str, values: np.ndarray, index: pd.DatetimeIndex) -> ComponentSummary:
    step = max(1, len(values) // PREVIEW_POINTS)
    preview = [
        Point(timestamp=index[i].isoformat(), value=round(float(values[i]), 6))
        for i in range(0, len(values), step)
    ][:PREVIEW_POINTS]
    return ComponentSummary(
        component=name,
        min=round(float(values.min()), 6),
        max=round(float(values.max()), 6),
        preview=preview,
    )
