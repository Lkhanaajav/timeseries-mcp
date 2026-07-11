import numpy as np
import pytest

from timeseries_mcp.analysis import decompose, stationarity
from timeseries_mcp.store import StoreError


def test_strong_seasonality_detected(minute_series):
    rng = np.random.default_rng(6)
    t = np.arange(480)
    values = 20 + 8 * np.sin(2 * np.pi * t / 48) + rng.normal(0, 0.5, 480)
    report = decompose.decompose(minute_series(values), "ts1", period=48, method="stl")
    assert report.seasonal_strength > 0.9
    assert report.trend_strength < 0.6
    assert {c.component for c in report.components} == {"trend", "seasonal", "residual"}


def test_strong_trend_detected(minute_series):
    rng = np.random.default_rng(7)
    t = np.arange(300)
    values = 0.5 * t + rng.normal(0, 1.0, 300)
    report = decompose.decompose(minute_series(values), "ts1", period=24, method="stl")
    assert report.trend_strength > 0.9
    assert report.seasonal_strength < 0.5


def test_classical_method_runs(minute_series):
    t = np.arange(200)
    values = 10 + 3 * np.sin(2 * np.pi * t / 20)
    report = decompose.decompose(minute_series(values), "ts1", period=20, method="classical")
    assert report.method == "classical"
    assert report.seasonal_strength > 0.9


def test_too_short_for_period(minute_series):
    with pytest.raises(StoreError, match="2 full periods"):
        decompose.decompose(minute_series(np.arange(50.0)), "ts1", period=48)


def test_white_noise_is_stationary(minute_series):
    rng = np.random.default_rng(8)
    report = stationarity.assess(minute_series(rng.normal(0, 1, 500)), "ts1")
    assert "stationary" in report.verdict
    assert "non-stationary" not in report.verdict


def test_random_walk_is_not_stationary(minute_series):
    rng = np.random.default_rng(9)
    values = np.cumsum(rng.normal(0, 1, 500))
    report = stationarity.assess(minute_series(values), "ts1")
    assert "non-stationary" in report.verdict
    assert "differencing" in report.differencing_hint.lower()
