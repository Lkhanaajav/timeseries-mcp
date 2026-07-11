import numpy as np
import pandas as pd
import pytest

from timeseries_mcp.analysis import baselines, quality
from timeseries_mcp.store import StoreError


def test_clean_series_verdict(minute_series):
    report = quality.audit(minute_series(np.arange(100.0)), "ts1")
    assert report.verdict.startswith("Clean")
    assert report.n_gaps_total == 0
    assert report.sampling_regularity_pct == 100.0


def test_gap_detected(minute_series):
    s = minute_series(np.arange(100.0))
    s = s.drop(s.index[40:50])  # 10-point hole
    report = quality.audit(s, "ts1")
    assert report.n_gaps_total == 1
    assert report.gaps[0].expected_points_missed == 10
    assert "gap" in report.verdict


def test_duplicates_and_missing_flagged(minute_series):
    s = minute_series([1.0, 2.0, np.nan, 4.0, 5.0])
    dup = pd.Series([9.0], index=[s.index[1]])
    s = pd.concat([s, dup]).sort_index()
    report = quality.audit(s, "ts1")
    assert report.duplicate_timestamps == 1
    assert report.missing_values == 1
    assert not report.is_monotonic  # duplicates break strict monotonicity


def test_naive_forecast_flat(minute_series):
    values = np.arange(50.0)
    report = baselines.forecast(minute_series(values), "ts1", horizon=5, method="naive")
    assert all(p.value == pytest.approx(49.0) for p in report.forecasts)
    assert report.backtest.n_test == 10


def test_naive_intervals_widen_with_noise(minute_series):
    rng = np.random.default_rng(18)
    values = 20 + rng.normal(0, 1, 100)
    report = baselines.forecast(minute_series(values), "ts1", horizon=5, method="naive")
    first, last = report.forecasts[0], report.forecasts[-1]
    assert first.lo < first.value < first.hi
    assert (last.hi - last.lo) > (first.hi - first.lo)  # sigma * sqrt(h) growth


def test_drift_extrapolates(minute_series):
    values = np.arange(100.0)  # perfect line, slope 1
    report = baselines.forecast(minute_series(values), "ts1", horizon=3, method="drift")
    assert [p.value for p in report.forecasts] == pytest.approx([100.0, 101.0, 102.0])
    assert report.backtest.mae == pytest.approx(0.0, abs=1e-9)


def test_seasonal_naive_beats_naive_on_seasonal_data(minute_series):
    t = np.arange(480)
    rng = np.random.default_rng(16)
    values = 10 + 5 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.3, 480)
    s = minute_series(values)
    naive = baselines.forecast(s, "ts1", horizon=24, method="naive")
    seasonal = baselines.forecast(s, "ts1", horizon=24, method="seasonal_naive", period=24)
    assert seasonal.backtest.mae < naive.backtest.mae


def test_ses_runs_and_backtests(minute_series):
    rng = np.random.default_rng(17)
    values = 50 + np.cumsum(rng.normal(0, 0.5, 200))
    report = baselines.forecast(minute_series(values), "ts1", horizon=10, method="ses")
    assert len(report.forecasts) == 10
    assert report.backtest.rmse > 0


def test_seasonal_naive_requires_period(minute_series):
    with pytest.raises(StoreError, match="period"):
        baselines.forecast(minute_series(np.arange(100.0)), "ts1", method="seasonal_naive")


def test_future_timestamps_continue_grid(minute_series):
    report = baselines.forecast(minute_series(np.arange(50.0)), "ts1", horizon=2, method="naive")
    assert report.forecasts[0].timestamp == "2026-01-01T00:50:00"
    assert report.forecasts[1].timestamp == "2026-01-01T00:51:00"


def test_future_timestamps_on_gapped_index_use_median_interval(minute_series):
    """Regression: index resolution (s/us/ns) must not corrupt the future step size."""
    s = minute_series(np.arange(60.0))
    s = s.drop(s.index[30:35])  # gap makes infer_freq return None
    report = baselines.forecast(s, "ts1", horizon=2, method="naive")
    assert report.forecasts[0].timestamp == "2026-01-01T01:00:00"
    assert report.forecasts[1].timestamp == "2026-01-01T01:01:00"


def test_median_interval_seconds_is_correct(minute_series):
    """Regression companion: 1-minute sampling must report exactly 60s."""
    report = quality.audit(minute_series(np.arange(20.0)), "ts1")
    assert report.median_interval_seconds == pytest.approx(60.0)
