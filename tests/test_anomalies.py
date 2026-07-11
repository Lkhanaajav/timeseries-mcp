import numpy as np
import pytest

from timeseries_mcp.analysis import anomalies
from timeseries_mcp.store import StoreError


def test_zscore_finds_injected_spike(minute_series):
    rng = np.random.default_rng(0)
    values = rng.normal(10, 1, 500)
    values[250] = 25.0
    report = anomalies.detect(minute_series(values), "ts1", method="zscore", threshold=3.0)
    assert report.n_anomalies >= 1
    assert report.anomalies[0].value == pytest.approx(25.0)
    assert report.anomalies[0].score > 3.0


def test_mad_robust_to_contamination(minute_series):
    """With 10% extreme contamination, MAD still flags the outliers; plain z-score's
    inflated std can miss milder ones."""
    rng = np.random.default_rng(1)
    values = rng.normal(0, 1, 200)
    values[:20] = 60.0  # heavy contamination inflates mean/std
    report = anomalies.detect(minute_series(values), "ts1", method="mad", threshold=3.5)
    assert report.n_anomalies >= 20


def test_iqr_fence_scores(minute_series):
    values = np.concatenate([np.linspace(0, 1, 100), [10.0]])
    report = anomalies.detect(minute_series(values), "ts1", method="iqr", threshold=1.5)
    assert report.n_anomalies == 1
    assert report.anomalies[0].value == 10.0


def test_stl_residual_catches_seasonal_context_anomaly(minute_series):
    """A value normal in absolute terms but wrong for its phase of the cycle:
    only the seasonal-aware method should flag it."""
    t = np.arange(576)
    values = 10 + 5 * np.sin(2 * np.pi * t / 48)
    values[300] = 10.0  # trough phase expects ~5; 10 is globally unremarkable
    series = minute_series(values)

    global_report = anomalies.detect(series, "ts1", method="zscore", threshold=3.0)
    stl_report = anomalies.detect(series, "ts1", method="stl_residual", threshold=3.0, period=48)

    flagged_global = {a.timestamp for a in global_report.anomalies}
    flagged_stl = {a.timestamp for a in stl_report.anomalies}
    target = series.index[300].isoformat()
    assert target not in flagged_global
    assert target in flagged_stl


def test_stl_requires_period(minute_series):
    with pytest.raises(StoreError, match="period"):
        anomalies.detect(minute_series(np.zeros(100) + 1), "ts1", method="stl_residual")


def test_constant_series_no_anomalies(minute_series):
    report = anomalies.detect(minute_series(np.full(50, 7.0)), "ts1", method="zscore")
    assert report.n_anomalies == 0
    assert "constant" in report.notes.lower()


def test_unknown_method_rejected(minute_series):
    with pytest.raises(StoreError, match="Unknown method"):
        anomalies.detect(minute_series(np.arange(20.0)), "ts1", method="isolation_forest")
