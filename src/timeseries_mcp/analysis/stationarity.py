"""Stationarity assessment: ADF and KPSS read together, with a combined verdict."""

from __future__ import annotations

import warnings

import pandas as pd

from ..models import HypothesisTest, StationarityReport
from ..store import clean_values

ALPHA = 0.05


def assess(s: pd.Series, series_id: str) -> StationarityReport:
    from statsmodels.tsa.stattools import adfuller, kpss

    clean = clean_values(s, min_points=12, context="Stationarity testing")
    values = clean.to_numpy()

    adf_stat, adf_p, *_ = adfuller(values, autolag="AIC")
    with warnings.catch_warnings():
        # KPSS p-values are table-interpolated; outside [0.01, 0.1] statsmodels
        # warns and clamps. The clamped value is still the right reading.
        warnings.simplefilter("ignore")
        kpss_stat, kpss_p, *_ = kpss(values, regression="c", nlags="auto")

    adf_stationary = adf_p < ALPHA  # ADF null: unit root (non-stationary)
    kpss_stationary = kpss_p >= ALPHA  # KPSS null: stationary

    adf_test = HypothesisTest(
        test="ADF",
        statistic=round(float(adf_stat), 4),
        p_value=round(float(adf_p), 6),
        conclusion=(
            f"p={adf_p:.4f} < {ALPHA}: reject unit root — evidence of stationarity."
            if adf_stationary
            else f"p={adf_p:.4f} >= {ALPHA}: cannot reject unit root — consistent with non-stationarity."
        ),
    )
    kpss_test = HypothesisTest(
        test="KPSS",
        statistic=round(float(kpss_stat), 4),
        p_value=round(float(kpss_p), 6),
        conclusion=(
            f"p={kpss_p:.4f} >= {ALPHA}: cannot reject stationarity."
            if kpss_stationary
            else f"p={kpss_p:.4f} < {ALPHA}: reject stationarity."
        ),
    )

    if adf_stationary and kpss_stationary:
        verdict = "Both tests agree: the series is stationary."
        hint = "No differencing needed."
    elif not adf_stationary and not kpss_stationary:
        verdict = "Both tests agree: the series is non-stationary."
        hint = "Apply first differencing, then re-test."
    elif adf_stationary:
        verdict = "Conflicting: ADF says stationary, KPSS says not — often difference-stationarity."
        hint = "Try first differencing; a deterministic trend is also plausible."
    else:
        verdict = "Conflicting: KPSS says stationary, ADF says not — often trend-stationarity."
        hint = "Consider detrending (see trend_test) rather than differencing."

    return StationarityReport(
        series_id=series_id, adf=adf_test, kpss=kpss_test, verdict=verdict, differencing_hint=hint
    )
