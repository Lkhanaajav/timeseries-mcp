"""timeseries-mcp server: deterministic statistical tools over MCP.

Design rules:
- Every tool returns a typed Pydantic model -> the host gets a validated
  ``outputSchema`` and structured content, never free-form prose.
- Raw data stays server-side behind series handles; responses are capped so
  a million-point series never floods the model's context.
- No code execution. Every number is produced by a unit-tested routine.
"""

from __future__ import annotations

import argparse
from typing import Annotated, Literal

import pandas as pd
from fastmcp import FastMCP
from pydantic import Field

from . import datasets
from .analysis import anomalies as anomalies_mod
from .analysis import baselines as baselines_mod
from .analysis import changepoints as changepoints_mod
from .analysis import correlation as correlation_mod
from .analysis import decompose as decompose_mod
from .analysis import quality as quality_mod
from .analysis import stationarity as stationarity_mod
from .analysis import trend as trend_mod
from .models import (
    AnomalyReport,
    AutocorrelationReport,
    CatalogResult,
    ChangepointReport,
    CompareReport,
    DataQualityReport,
    DecompositionReport,
    DescribeResult,
    ForecastReport,
    Point,
    RollingStatPreview,
    RollingStatsResult,
    SeriesInfo,
    StationarityReport,
    TrendReport,
    WindowResult,
)
from .store import SeriesStore, StoreError, value_stats

WINDOW_CAP = 500
ROLLING_PREVIEW = 60

mcp = FastMCP(
    name="timeseries-mcp",
    instructions=(
        "Deterministic time-series statistics. Load data once (load_csv / load_values / "
        "load_sample) to get a series_id, then analyze by id — raw data stays server-side. "
        "Typical flow: load -> data_quality -> describe -> decompose/autocorrelation -> "
        "detect_anomalies/detect_changepoints/trend_test -> forecast_baseline."
    ),
)

store = SeriesStore()


# --------------------------------------------------------------------------
# Loading & catalog
# --------------------------------------------------------------------------


@mcp.tool
def load_csv(
    path: Annotated[str, Field(description="CSV path inside the data root (see TIMESERIES_MCP_DATA_ROOT).")],
    timestamp_column: Annotated[str | None, Field(description="Timestamp column; auto-detected if omitted.")] = None,
    value_column: Annotated[str | None, Field(description="Numeric value column; first numeric column if omitted.")] = None,
) -> SeriesInfo:
    """Load one column of a CSV as a time series and register it under a series_id."""
    series_id = store.load_csv(path, timestamp_column, value_column)
    return store.info(series_id)


@mcp.tool
def load_values(
    values: Annotated[list[float], Field(description="The observations, in time order.", max_length=100_000)],
    timestamps: Annotated[list[str] | None, Field(description="ISO-8601 timestamps matching `values`.")] = None,
    start: Annotated[str | None, Field(description="If no timestamps: start time for a regular grid.")] = None,
    freq: Annotated[str | None, Field(description="If no timestamps: pandas frequency (default '1min').")] = None,
    name: Annotated[str, Field(description="Human-readable label for the series.")] = "inline",
) -> SeriesInfo:
    """Register a series from inline values (small data; prefer load_csv for files)."""
    series_id = store.load_values(values, timestamps, start, freq, name)
    return store.info(series_id)


@mcp.tool
def load_sample(
    name: Annotated[
        Literal["server_room_temp", "cpu_utilization", "methane_ppm"],
        Field(description="Which bundled deterministic sample dataset to load."),
    ],
) -> SeriesInfo:
    """Load a bundled synthetic sample (seeded, reproducible) — useful for demos and evals."""
    series = datasets.make(name)
    series_id = store.add(series, name=name, source=f"sample:{name}")
    return store.info(series_id)


@mcp.tool
def list_series() -> CatalogResult:
    """List every series currently loaded, with basic stats."""
    infos = store.all_infos()
    return CatalogResult(n_series=len(infos), series=infos)


# --------------------------------------------------------------------------
# Inspection
# --------------------------------------------------------------------------


@mcp.tool
def describe(series_id: str) -> DescribeResult:
    """Distributional summary: quartiles, spread, skewness, kurtosis, missing count."""
    s = store.get(series_id)
    clean = s.dropna()
    from scipy import stats as scipy_stats

    info = store.info(series_id)
    return DescribeResult(
        series_id=series_id,
        stats=value_stats(s),
        skewness=round(float(scipy_stats.skew(clean)), 4) if len(clean) > 2 else 0.0,
        kurtosis=round(float(scipy_stats.kurtosis(clean)), 4) if len(clean) > 3 else 0.0,
        first_timestamp=info.start,
        last_timestamp=info.end,
        inferred_freq=info.inferred_freq,
    )


@mcp.tool
def get_window(
    series_id: str,
    start: Annotated[str | None, Field(description="ISO-8601 window start (inclusive).")] = None,
    end: Annotated[str | None, Field(description="ISO-8601 window end (inclusive).")] = None,
    limit: Annotated[int, Field(ge=1, le=WINDOW_CAP, description="Max points to return.")] = 200,
) -> WindowResult:
    """Fetch raw observations in a time window (evenly thinned if over the limit)."""
    s = store.get(series_id)
    window = s.loc[slice(pd.Timestamp(start) if start else None, pd.Timestamp(end) if end else None)]
    n = len(window)
    step = max(1, -(-n // limit))  # ceil division: keep <= limit points
    thinned = window.iloc[::step]
    points = [
        Point(timestamp=ts.isoformat(), value=None if pd.isna(v) else round(float(v), 6))
        for ts, v in thinned.items()
    ]
    return WindowResult(
        series_id=series_id, n_in_window=n, returned=len(points), truncated=step > 1, points=points
    )


# --------------------------------------------------------------------------
# Transformation
# --------------------------------------------------------------------------


@mcp.tool
def resample(
    series_id: str,
    rule: Annotated[str, Field(description="Pandas offset alias, e.g. '5min', '1h', '1D'.")],
    agg: Annotated[Literal["mean", "sum", "min", "max", "median", "count"], Field(description="Aggregation.")] = "mean",
) -> SeriesInfo:
    """Resample onto a regular grid; registers and returns a NEW derived series."""
    s = store.get(series_id)
    try:
        resampled = s.resample(rule).agg(agg).dropna()
    except ValueError as exc:
        raise StoreError(f"Invalid resample rule '{rule}': {exc}") from exc
    if resampled.empty:
        raise StoreError(f"Resampling '{series_id}' with rule '{rule}' produced no points.")
    meta_name = f"{store.info(series_id).name}[{rule},{agg}]"
    new_id = store.add(resampled, name=meta_name, source=f"resample({series_id},{rule},{agg})")
    return store.info(new_id)


@mcp.tool
def rolling_stats(
    series_id: str,
    window: Annotated[int, Field(ge=2, le=100_000, description="Window size in observations.")],
    stats: Annotated[
        list[Literal["mean", "std", "min", "max", "median"]],
        Field(description="Which rolling statistics to compute."),
    ] = ["mean", "std"],  # noqa: B006 — literal default is fine for a tool signature
) -> RollingStatsResult:
    """Rolling-window statistics with an evenly spaced preview per stat."""
    s = store.get(series_id).dropna()
    if window >= len(s):
        raise StoreError(f"window ({window}) must be smaller than the series length ({len(s)}).")
    previews = []
    for stat in dict.fromkeys(stats):
        rolled = getattr(s.rolling(window), stat)().dropna()
        step = max(1, len(rolled) // ROLLING_PREVIEW)
        previews.append(
            RollingStatPreview(
                stat=stat,
                min=round(float(rolled.min()), 6),
                max=round(float(rolled.max()), 6),
                last=round(float(rolled.iloc[-1]), 6),
                preview=[
                    Point(timestamp=ts.isoformat(), value=round(float(v), 6))
                    for ts, v in rolled.iloc[::step].items()
                ][:ROLLING_PREVIEW],
            )
        )
    return RollingStatsResult(series_id=series_id, window=window, stats=previews)


# --------------------------------------------------------------------------
# Diagnostics & detection
# --------------------------------------------------------------------------


@mcp.tool
def data_quality(series_id: str) -> DataQualityReport:
    """Audit sampling gaps, duplicate timestamps, missing values, and regularity."""
    return quality_mod.audit(store.get(series_id), series_id)


@mcp.tool
def detect_anomalies(
    series_id: str,
    method: Annotated[
        Literal["zscore", "mad", "iqr", "stl_residual"],
        Field(description="zscore/mad/iqr are global; stl_residual is seasonal-aware (needs period)."),
    ] = "zscore",
    threshold: Annotated[float, Field(gt=0, description="Score cutoff (zscore/mad/stl) or IQR fence multiplier (iqr).")] = 3.0,
    period: Annotated[int | None, Field(description="Seasonal period, required for stl_residual.")] = None,
) -> AnomalyReport:
    """Flag anomalous observations; returns scored anomalies, strongest first."""
    return anomalies_mod.detect(store.get(series_id), series_id, method, threshold, period)


@mcp.tool
def detect_changepoints(
    series_id: str,
    threshold: Annotated[float, Field(gt=0, description="CUSUM significance bound; 1.36 ~ 95%.")] = 1.36,
    min_segment_length: Annotated[int, Field(ge=2, description="Minimum points between changepoints.")] = 10,
    max_changepoints: Annotated[int, Field(ge=1, le=50)] = 10,
) -> ChangepointReport:
    """Detect level shifts (mean changes) via CUSUM binary segmentation."""
    return changepoints_mod.detect(
        store.get(series_id), series_id, threshold, min_segment_length, max_changepoints
    )


@mcp.tool
def decompose(
    series_id: str,
    period: Annotated[int, Field(ge=2, description="Observations per season, e.g. 288 for daily @ 5min.")],
    method: Annotated[Literal["stl", "classical"], Field(description="STL is robust to outliers.")] = "stl",
) -> DecompositionReport:
    """Split the series into trend/seasonal/residual and quantify each component's strength."""
    return decompose_mod.decompose(store.get(series_id), series_id, period, method)


@mcp.tool
def stationarity(series_id: str) -> StationarityReport:
    """Run ADF and KPSS together and give a combined stationarity verdict."""
    return stationarity_mod.assess(store.get(series_id), series_id)


@mcp.tool
def autocorrelation(
    series_id: str,
    nlags: Annotated[int | None, Field(ge=1, le=500, description="Defaults to min(40, n/2 - 1).")] = None,
) -> AutocorrelationReport:
    """ACF/PACF with significance bounds; suggests a seasonal period when one stands out."""
    return correlation_mod.autocorrelation(store.get(series_id), series_id, nlags)


@mcp.tool
def trend_test(series_id: str) -> TrendReport:
    """Estimate trend three ways: OLS, robust Theil-Sen, and the Mann-Kendall test."""
    return trend_mod.trend_test(store.get(series_id), series_id)


@mcp.tool
def compare_series(
    series_a: str,
    series_b: str,
    max_lag: Annotated[int, Field(ge=0, le=1000, description="Max lead/lag (in steps) to scan.")] = 48,
) -> CompareReport:
    """Correlate two series on shared timestamps and find the lag of strongest coupling."""
    return correlation_mod.compare(
        store.get(series_a), store.get(series_b), series_a, series_b, max_lag
    )


@mcp.tool
def forecast_baseline(
    series_id: str,
    horizon: Annotated[int, Field(ge=1, le=500, description="Steps ahead to forecast.")] = 12,
    method: Annotated[
        Literal["naive", "seasonal_naive", "drift", "ses"],
        Field(description="Reference methods per Hyndman FPP3; every result includes a holdout backtest."),
    ] = "naive",
    period: Annotated[int | None, Field(description="Seasonal period, required for seasonal_naive.")] = None,
) -> ForecastReport:
    """Baseline forecast with 95% intervals and an honest holdout backtest."""
    return baselines_mod.forecast(store.get(series_id), series_id, horizon, method, period)


# --------------------------------------------------------------------------
# Resources & prompts
# --------------------------------------------------------------------------


@mcp.resource("timeseries://catalog")
def catalog_resource() -> str:
    """Markdown table of every loaded series."""
    infos = store.all_infos()
    if not infos:
        return "No series loaded yet. Use load_csv, load_values, or load_sample."
    lines = ["| id | name | points | start | end | freq |", "|---|---|---|---|---|---|"]
    lines += [
        f"| {i.series_id} | {i.name} | {i.n_points} | {i.start} | {i.end} | {i.inferred_freq or '-'} |"
        for i in infos
    ]
    return "\n".join(lines)


@mcp.resource("timeseries://{series_id}/summary")
def summary_resource(series_id: str) -> str:
    """One-paragraph summary of a stored series."""
    i = store.info(series_id)
    return (
        f"Series {i.series_id} ('{i.name}', {i.source}): {i.n_points} points from {i.start} "
        f"to {i.end} (freq: {i.inferred_freq or 'irregular'}). Mean {i.stats.mean:.4g}, "
        f"std {i.stats.std:.4g}, range [{i.stats.min:.4g}, {i.stats.max:.4g}], "
        f"{i.stats.missing} missing."
    )


@mcp.prompt
def analyze_series(goal: str = "general health check") -> str:
    """Guided end-to-end analysis workflow for a loaded series."""
    return (
        f"Analyze the loaded time series with this goal: {goal}.\n\n"
        "Work through these steps, citing tool outputs for every claim:\n"
        "1. list_series, then data_quality — report gaps/duplicates before trusting anything.\n"
        "2. describe + autocorrelation — note distribution shape and any seasonal period.\n"
        "3. If a period exists: decompose (report trend/seasonal strength).\n"
        "4. detect_anomalies (pick a method suited to the seasonality) and detect_changepoints.\n"
        "5. trend_test + stationarity for the long-run behavior.\n"
        "6. forecast_baseline and report the backtest error alongside the forecast.\n"
        "Finish with a short summary: only numbers that appeared in tool outputs."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="timeseries-mcp server")
    parser.add_argument(
        "--transport", choices=["stdio", "http"], default="stdio", help="stdio (default) or Streamable HTTP"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.transport == "http":
        mcp.run(transport="http", host=args.host, port=args.port)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
