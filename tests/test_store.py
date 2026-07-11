import numpy as np
import pytest

from timeseries_mcp.store import DATA_ROOT_ENV, SeriesStore, StoreError


@pytest.fixture
def fresh_store():
    return SeriesStore()


def test_load_values_with_synthesized_index(fresh_store):
    sid = fresh_store.load_values([1.0, 2.0, 3.0], start="2026-01-01", freq="1h")
    info = fresh_store.info(sid)
    assert info.n_points == 3
    assert info.start == "2026-01-01T00:00:00"
    assert info.inferred_freq is not None


def test_load_values_mismatched_lengths(fresh_store):
    with pytest.raises(StoreError, match="must match"):
        fresh_store.load_values([1.0, 2.0], timestamps=["2026-01-01"])


def test_unknown_id_lists_known_ids(fresh_store):
    fresh_store.load_values([1.0, 2.0, 3.0])
    with pytest.raises(StoreError, match="ts1"):
        fresh_store.get("ts99")


def test_series_sorted_by_timestamp(fresh_store):
    sid = fresh_store.load_values(
        [3.0, 1.0, 2.0],
        timestamps=["2026-01-03", "2026-01-01", "2026-01-02"],
    )
    values = fresh_store.get(sid)
    assert list(values.to_numpy()) == [1.0, 2.0, 3.0]


def test_load_csv_detects_columns(fresh_store, tmp_path, monkeypatch):
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    csv = tmp_path / "sensor.csv"
    csv.write_text("timestamp,temp_c\n2026-01-01T00:00,20.5\n2026-01-01T00:05,21.0\n")
    sid = fresh_store.load_csv("sensor.csv")
    info = fresh_store.info(sid)
    assert info.n_points == 2
    assert "temp_c" in info.name


def test_load_csv_rejects_path_escape(fresh_store, tmp_path, monkeypatch):
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path / "root"))
    (tmp_path / "root").mkdir()
    (tmp_path / "secret.csv").write_text("t,v\n2026-01-01,1\n")
    with pytest.raises(StoreError, match="outside the allowed data root"):
        fresh_store.load_csv("../secret.csv")


def test_load_csv_rejects_absolute_escape(fresh_store, tmp_path, monkeypatch):
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    with pytest.raises(StoreError, match="outside the allowed data root"):
        fresh_store.load_csv("/etc/passwd")


def test_empty_series_rejected(fresh_store):
    with pytest.raises(StoreError, match="empty"):
        fresh_store.load_values([])


def test_nan_values_preserved_not_dropped(fresh_store):
    sid = fresh_store.load_values([1.0, float("nan"), 3.0])
    info = fresh_store.info(sid)
    assert info.stats.missing == 1
    assert info.n_points == 3
    assert not np.isnan(info.stats.mean)
