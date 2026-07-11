import numpy as np
import pytest

from timeseries_mcp.analysis import correlation, trend
from timeseries_mcp.store import StoreError


def test_known_slope_recovered(minute_series):
    rng = np.random.default_rng(10)
    values = 3.0 + 0.25 * np.arange(400) + rng.normal(0, 1.0, 400)
    report = trend.trend_test(minute_series(values), "ts1")
    assert report.direction == "increasing"
    assert report.theil_sen_slope_per_step == pytest.approx(0.25, abs=0.02)
    assert report.ols_slope_per_step == pytest.approx(0.25, abs=0.02)
    assert report.mann_kendall.p_value < 0.001


def test_no_trend_in_noise(minute_series):
    rng = np.random.default_rng(11)
    report = trend.trend_test(minute_series(rng.normal(5, 1, 300)), "ts1")
    assert report.direction == "no significant trend"


def test_decreasing_trend(minute_series):
    rng = np.random.default_rng(12)
    values = 100 - 0.1 * np.arange(300) + rng.normal(0, 0.5, 300)
    report = trend.trend_test(minute_series(values), "ts1")
    assert report.direction == "decreasing"
    assert report.change_over_span == pytest.approx(-0.1 * 299, rel=0.1)


def test_mann_kendall_subsamples_large_series(minute_series):
    values = np.arange(5000, dtype=float)
    report = trend.trend_test(minute_series(values), "ts1")
    assert report.direction == "increasing"
    assert "subsampled" in report.mann_kendall.conclusion


def test_acf_finds_seasonal_period(minute_series):
    t = np.arange(400)
    rng = np.random.default_rng(13)
    values = np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.2, 400)
    report = correlation.autocorrelation(minute_series(values), "ts1", nlags=60)
    assert report.suggested_period == pytest.approx(24, abs=1)
    assert 24 in report.significant_lags or 23 in report.significant_lags


def test_compare_correlated_series(minute_series):
    rng = np.random.default_rng(14)
    a = rng.normal(0, 1, 300)
    b = 2 * a + rng.normal(0, 0.1, 300)
    report = correlation.compare(minute_series(a), minute_series(b), "ts1", "ts2")
    assert report.pearson.statistic > 0.99
    assert report.best_lag == 0
    assert "strong" in report.interpretation.lower()


def test_compare_finds_lag(minute_series):
    rng = np.random.default_rng(15)
    base = np.sin(2 * np.pi * np.arange(400) / 50) + rng.normal(0, 0.05, 400)
    lagged = np.roll(base, 5)  # b at time t equals a at t-5 -> b lags a by 5
    report = correlation.compare(
        minute_series(base), minute_series(lagged), "ts1", "ts2", max_lag=20
    )
    assert abs(report.best_lag) == 5
    assert abs(report.ccf_at_best_lag) > 0.9


def test_compare_requires_overlap(minute_series):
    a = minute_series([1.0] * 10, start="2026-01-01")
    b = minute_series([2.0] * 10, start="2027-06-01")
    with pytest.raises(StoreError, match="overlap"):
        correlation.compare(a, b, "ts1", "ts2")
