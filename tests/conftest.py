import numpy as np
import pandas as pd
import pytest

from timeseries_mcp.server import store


@pytest.fixture(autouse=True)
def clean_store():
    store.clear()
    yield
    store.clear()


@pytest.fixture
def minute_series():
    """Regular 1-minute series factory."""

    def _make(values, start="2026-01-01"):
        index = pd.date_range(start, periods=len(values), freq="1min")
        return pd.Series(np.asarray(values, dtype=float), index=index)

    return _make
