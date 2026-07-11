"""Trend estimation: OLS, robust Theil-Sen, and the Mann-Kendall test."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..models import HypothesisTest, TrendReport
from ..store import clean_values

ALPHA = 0.05
MK_MAX_N = 2000  # Mann-Kendall is O(n^2); larger series are evenly subsampled.


def trend_test(s: pd.Series, series_id: str) -> TrendReport:
    from scipy import stats

    clean = clean_values(s, min_points=8, context="Trend testing")
    y = clean.to_numpy()
    x = np.arange(len(y), dtype=float)

    ols = stats.linregress(x, y)
    ts_slope, _, ts_lo, ts_hi = stats.theilslopes(y, x, alpha=0.95)
    mk_z, mk_p, subsampled = _mann_kendall(y)

    significant = mk_p < ALPHA
    if significant and ts_slope > 0:
        direction = "increasing"
    elif significant and ts_slope < 0:
        direction = "decreasing"
    else:
        direction = "no significant trend"

    mk_note = " (evenly subsampled to 2000 points)" if subsampled else ""
    return TrendReport(
        series_id=series_id,
        ols_slope_per_step=round(float(ols.slope), 8),
        ols_r_squared=round(float(ols.rvalue**2), 4),
        theil_sen_slope_per_step=round(float(ts_slope), 8),
        theil_sen_ci_low=round(float(ts_lo), 8),
        theil_sen_ci_high=round(float(ts_hi), 8),
        mann_kendall=HypothesisTest(
            test="Mann-Kendall",
            statistic=round(float(mk_z), 4),
            p_value=round(float(mk_p), 6),
            conclusion=(
                f"p={mk_p:.4f} {'<' if significant else '>='} {ALPHA}: "
                f"{'monotonic trend detected' if significant else 'no monotonic trend detected'}{mk_note}."
            ),
        ),
        direction=direction,
        change_over_span=round(float(ts_slope * (len(y) - 1)), 6),
    )


def _mann_kendall(y: np.ndarray) -> tuple[float, float, bool]:
    """Mann-Kendall z and two-sided p with tie correction."""
    from scipy import stats

    subsampled = len(y) > MK_MAX_N
    if subsampled:
        y = y[np.linspace(0, len(y) - 1, MK_MAX_N).astype(int)]
    n = len(y)
    diff_sign = np.sign(np.subtract.outer(y, y))  # [i, j] = sign(y_i - y_j)
    s_stat = int(-diff_sign[np.triu_indices(n, k=1)].sum())  # S = sum_{i<j} sign(y_j - y_i)

    _, counts = np.unique(y, return_counts=True)
    tie_term = (counts * (counts - 1) * (2 * counts + 5)).sum()
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    if var_s == 0:
        return 0.0, 1.0, subsampled
    if s_stat > 0:
        z = (s_stat - 1) / np.sqrt(var_s)
    elif s_stat < 0:
        z = (s_stat + 1) / np.sqrt(var_s)
    else:
        z = 0.0
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return float(z), float(p), subsampled
