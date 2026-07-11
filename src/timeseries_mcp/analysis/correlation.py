"""Autocorrelation structure and cross-series comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..models import AutocorrelationReport, CompareReport, HypothesisTest
from ..store import StoreError, clean_values


def autocorrelation(s: pd.Series, series_id: str, nlags: int | None = None) -> AutocorrelationReport:
    from statsmodels.tsa.stattools import acf, pacf

    clean = clean_values(s, min_points=8, context="Autocorrelation")
    n = len(clean)
    max_allowed = n // 2 - 1
    nlags = min(nlags or 40, max_allowed)
    if nlags < 1:
        raise StoreError(f"Series too short for autocorrelation (n={n}).")

    values = clean.to_numpy()
    acf_vals = acf(values, nlags=nlags, fft=True)
    pacf_vals = pacf(values, nlags=nlags, method="ywm")
    bound = 1.96 / np.sqrt(n)
    significant = [int(k) for k in range(1, nlags + 1) if abs(acf_vals[k]) > bound]

    return AutocorrelationReport(
        series_id=series_id,
        nlags=nlags,
        acf=[round(float(v), 4) for v in acf_vals],
        pacf=[round(float(v), 4) for v in pacf_vals],
        confidence_bound=round(float(bound), 4),
        significant_lags=significant,
        suggested_period=_suggest_period(acf_vals, bound),
    )


def _suggest_period(acf_vals: np.ndarray, bound: float) -> int | None:
    """First local ACF maximum at lag >= 2 that is significant and reasonably strong."""
    for k in range(2, len(acf_vals) - 1):
        if (
            acf_vals[k] > max(bound, 0.3)
            and acf_vals[k] >= acf_vals[k - 1]
            and acf_vals[k] >= acf_vals[k + 1]
        ):
            return int(k)
    return None


def compare(sa: pd.Series, sb: pd.Series, id_a: str, id_b: str, max_lag: int = 48) -> CompareReport:
    from scipy import stats

    joined = pd.concat([sa.rename("a"), sb.rename("b")], axis=1, join="inner").dropna()
    n = len(joined)
    if n < 8:
        raise StoreError(
            f"Series share only {n} timestamps; need at least 8 overlapping observations. "
            "Consider `resample` to align them onto a common grid first."
        )
    a, b = joined["a"].to_numpy(), joined["b"].to_numpy()

    pearson = stats.pearsonr(a, b)
    spearman = stats.spearmanr(a, b)
    best_lag, best_ccf = _best_lag(a, b, min(max_lag, n // 3))

    strength = abs(pearson.statistic)
    label = "strong" if strength >= 0.7 else "moderate" if strength >= 0.4 else "weak"
    lag_note = (
        f" Strongest alignment at lag {best_lag} (r={best_ccf:.2f}), i.e. '{id_b}' "
        f"{'lags' if best_lag > 0 else 'leads'} '{id_a}' by {abs(best_lag)} steps."
        if best_lag != 0
        else ""
    )
    return CompareReport(
        series_a=id_a,
        series_b=id_b,
        n_overlap=n,
        pearson=HypothesisTest(
            test="Pearson r",
            statistic=round(float(pearson.statistic), 4),
            p_value=round(float(pearson.pvalue), 6),
            conclusion=f"{label} linear relationship (r={pearson.statistic:.3f}).",
        ),
        spearman=HypothesisTest(
            test="Spearman rho",
            statistic=round(float(spearman.statistic), 4),
            p_value=round(float(spearman.pvalue), 6),
            conclusion=f"rank correlation rho={spearman.statistic:.3f}.",
        ),
        best_lag=best_lag,
        ccf_at_best_lag=round(float(best_ccf), 4),
        interpretation=f"{label.capitalize()} contemporaneous correlation on {n} shared points.{lag_note}",
    )


def _best_lag(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[int, float]:
    best = (0, np.corrcoef(a, b)[0, 1])
    for lag in range(-max_lag, max_lag + 1):
        if lag == 0:
            continue
        if lag > 0:
            xa, xb = a[:-lag], b[lag:]
        else:
            xa, xb = a[-lag:], b[:lag]
        if len(xa) < 8 or xa.std() == 0 or xb.std() == 0:
            continue
        r = np.corrcoef(xa, xb)[0, 1]
        if abs(r) > abs(best[1]):
            best = (lag, r)
    return int(best[0]), float(best[1])
