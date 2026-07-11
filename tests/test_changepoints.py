import numpy as np

from timeseries_mcp.analysis import changepoints


def test_single_step_detected_near_true_index(minute_series):
    rng = np.random.default_rng(2)
    values = np.concatenate([rng.normal(0, 1, 200), rng.normal(5, 1, 200)])
    report = changepoints.detect(minute_series(values), "ts1")
    assert report.n_changepoints == 1
    cp = report.changepoints[0]
    assert abs(cp.index - 200) <= 3
    assert cp.delta > 4.0


def test_two_steps_detected(minute_series):
    rng = np.random.default_rng(3)
    values = np.concatenate(
        [rng.normal(0, 1, 150), rng.normal(6, 1, 150), rng.normal(-3, 1, 150)]
    )
    report = changepoints.detect(minute_series(values), "ts1")
    assert report.n_changepoints == 2
    indices = sorted(cp.index for cp in report.changepoints)
    assert abs(indices[0] - 150) <= 5
    assert abs(indices[1] - 300) <= 5


def test_no_changepoints_in_stationary_noise(minute_series):
    rng = np.random.default_rng(4)
    report = changepoints.detect(minute_series(rng.normal(0, 1, 400)), "ts1")
    assert report.n_changepoints == 0


def test_constant_series(minute_series):
    report = changepoints.detect(minute_series(np.full(100, 3.0)), "ts1")
    assert report.n_changepoints == 0
    assert "constant" in report.notes.lower()


def test_changepoints_sorted_chronologically(minute_series):
    rng = np.random.default_rng(5)
    values = np.concatenate(
        [rng.normal(0, 1, 100), rng.normal(8, 1, 100), rng.normal(2, 1, 100)]
    )
    report = changepoints.detect(minute_series(values), "ts1")
    indices = [cp.index for cp in report.changepoints]
    assert indices == sorted(indices)
