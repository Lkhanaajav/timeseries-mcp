"""In-memory series registry.

Agents work with short series handles (``ts1``, ``ts2``, ...) instead of
re-sending raw arrays on every tool call — the raw data stays server-side,
which keeps token usage flat regardless of series length.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .models import SeriesInfo, ValueStats

MAX_POINTS = 1_000_000
MAX_SERIES = 200

DATA_ROOT_ENV = "TIMESERIES_MCP_DATA_ROOT"


class StoreError(ValueError):
    """Raised for any user-correctable store problem (bad path, bad id, limits)."""


def data_root() -> Path:
    """Directory CSV loading is sandboxed to (env override, default: cwd)."""
    return Path(os.environ.get(DATA_ROOT_ENV, os.getcwd())).resolve()


class SeriesStore:
    """Holds named pandas Series with DatetimeIndex, keyed by short ids."""

    def __init__(self) -> None:
        self._series: dict[str, pd.Series] = {}
        self._meta: dict[str, dict[str, str]] = {}
        self._counter = 0

    # -- registration -------------------------------------------------------

    def add(self, values: pd.Series, name: str, source: str) -> str:
        if len(self._series) >= MAX_SERIES:
            raise StoreError(f"Store is full ({MAX_SERIES} series). Load fewer series per session.")
        if len(values) > MAX_POINTS:
            raise StoreError(f"Series has {len(values)} points; the limit is {MAX_POINTS}.")
        if len(values) == 0:
            raise StoreError("Series is empty.")
        if not isinstance(values.index, pd.DatetimeIndex):
            raise StoreError("Internal error: series index must be a DatetimeIndex.")
        values = values.sort_index()
        self._counter += 1
        series_id = f"ts{self._counter}"
        self._series[series_id] = values.astype(float)
        self._meta[series_id] = {"name": name, "source": source}
        return series_id

    def get(self, series_id: str) -> pd.Series:
        if series_id not in self._series:
            known = ", ".join(self._series) or "none loaded yet"
            raise StoreError(f"Unknown series_id '{series_id}'. Known ids: {known}.")
        return self._series[series_id]

    def clear(self) -> None:
        self._series.clear()
        self._meta.clear()
        self._counter = 0

    # -- loading ------------------------------------------------------------

    def load_csv(
        self,
        path: str,
        timestamp_column: str | None = None,
        value_column: str | None = None,
    ) -> str:
        resolved = self._safe_path(path)
        try:
            df = pd.read_csv(resolved)
        except Exception as exc:  # pandas raises many types; surface one clean message
            raise StoreError(f"Could not parse CSV '{path}': {exc}") from exc
        if len(df) > MAX_POINTS:
            raise StoreError(f"CSV has {len(df)} rows; the limit is {MAX_POINTS}.")
        if df.empty:
            raise StoreError(f"CSV '{path}' has no rows.")

        ts_col = timestamp_column or self._detect_timestamp_column(df)
        if ts_col not in df.columns:
            raise StoreError(f"Timestamp column '{ts_col}' not in CSV columns {list(df.columns)}.")
        val_col = value_column or self._detect_value_column(df, exclude=ts_col)
        if val_col not in df.columns:
            raise StoreError(f"Value column '{val_col}' not in CSV columns {list(df.columns)}.")

        try:
            index = pd.DatetimeIndex(pd.to_datetime(df[ts_col], utc=False, format="mixed"))
        except Exception as exc:
            raise StoreError(f"Column '{ts_col}' could not be parsed as timestamps: {exc}") from exc
        values = pd.to_numeric(df[val_col], errors="coerce")
        series = pd.Series(values.to_numpy(dtype=float), index=index)
        return self.add(series, name=f"{resolved.stem}.{val_col}", source=f"csv:{resolved.name}")

    def load_values(
        self,
        values: list[float],
        timestamps: list[str] | None = None,
        start: str | None = None,
        freq: str | None = None,
        name: str = "inline",
    ) -> str:
        if timestamps is not None:
            if len(timestamps) != len(values):
                raise StoreError(
                    f"Got {len(values)} values but {len(timestamps)} timestamps — they must match."
                )
            try:
                index = pd.DatetimeIndex(pd.to_datetime(timestamps, format="mixed"))
            except Exception as exc:
                raise StoreError(f"Timestamps could not be parsed: {exc}") from exc
        else:
            start_ts = pd.Timestamp(start) if start else pd.Timestamp("2026-01-01")
            index = pd.date_range(start=start_ts, periods=len(values), freq=freq or "1min")
        series = pd.Series(np.asarray(values, dtype=float), index=index)
        return self.add(series, name=name, source="inline")

    # -- descriptions -------------------------------------------------------

    def info(self, series_id: str) -> SeriesInfo:
        s = self.get(series_id)
        meta = self._meta[series_id]
        return SeriesInfo(
            series_id=series_id,
            name=meta["name"],
            n_points=int(len(s)),
            start=s.index[0].isoformat(),
            end=s.index[-1].isoformat(),
            inferred_freq=pd.infer_freq(s.index) if len(s) >= 3 else None,
            source=meta["source"],
            stats=value_stats(s),
        )

    def all_infos(self) -> list[SeriesInfo]:
        return [self.info(sid) for sid in self._series]

    # -- helpers ------------------------------------------------------------

    def _safe_path(self, path: str) -> Path:
        """Resolve *path* and refuse anything outside the data root."""
        root = data_root()
        candidate = (root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
        if not candidate.is_relative_to(root):
            raise StoreError(
                f"Path '{path}' is outside the allowed data root '{root}'. "
                f"Set {DATA_ROOT_ENV} to change the sandbox."
            )
        if not candidate.is_file():
            raise StoreError(f"No file at '{candidate}'.")
        return candidate

    @staticmethod
    def _detect_timestamp_column(df: pd.DataFrame) -> str:
        pattern = re.compile(r"time|date|ts|stamp", re.IGNORECASE)
        for col in df.columns:
            if pattern.search(str(col)):
                return str(col)
        return str(df.columns[0])

    @staticmethod
    def _detect_value_column(df: pd.DataFrame, exclude: str) -> str:
        for col in df.columns:
            if str(col) == exclude:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                return str(col)
        raise StoreError(
            f"No numeric value column found in {list(df.columns)}; pass value_column explicitly."
        )


def value_stats(s: pd.Series) -> ValueStats:
    clean = s.dropna()
    if clean.empty:
        raise StoreError("Series contains only NaN values.")
    q = clean.quantile([0.25, 0.5, 0.75])
    return ValueStats(
        count=int(len(s)),
        mean=float(clean.mean()),
        std=float(clean.std(ddof=1)) if len(clean) > 1 else 0.0,
        min=float(clean.min()),
        p25=float(q.loc[0.25]),
        median=float(q.loc[0.5]),
        p75=float(q.loc[0.75]),
        max=float(clean.max()),
        missing=int(s.isna().sum()),
    )


def clean_values(s: pd.Series, min_points: int, context: str) -> pd.Series:
    """Drop NaNs and enforce a minimum length with a clear error."""
    clean = s.dropna()
    if len(clean) < min_points:
        raise StoreError(
            f"{context} needs at least {min_points} non-missing points; series has {len(clean)}."
        )
    return clean
