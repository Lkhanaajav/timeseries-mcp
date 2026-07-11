"""Typed output models for every tool.

Each model becomes the tool's MCP ``outputSchema``, so hosts and agents get
schema-validated structured results instead of free-form text.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ValueStats(BaseModel):
    """Five-number-style summary of the values in a series."""

    count: int
    mean: float
    std: float
    min: float
    p25: float
    median: float
    p75: float
    max: float
    missing: int = Field(description="Number of NaN values.")


class SeriesInfo(BaseModel):
    """Registry entry describing one stored series."""

    series_id: str
    name: str
    n_points: int
    start: str = Field(description="ISO-8601 timestamp of the first observation.")
    end: str = Field(description="ISO-8601 timestamp of the last observation.")
    inferred_freq: str | None = Field(
        description="Pandas frequency string inferred from the index, e.g. '5min'; null if irregular."
    )
    source: str = Field(description="Where the series came from: csv path, inline, sample, or a derivation.")
    stats: ValueStats


class CatalogResult(BaseModel):
    """All series currently loaded in the store."""

    n_series: int
    series: list[SeriesInfo]


class Point(BaseModel):
    """One observation."""

    timestamp: str
    value: float | None


class WindowResult(BaseModel):
    """Raw observations inside a requested time window (capped)."""

    series_id: str
    n_in_window: int
    returned: int
    truncated: bool
    points: list[Point]


class DescribeResult(BaseModel):
    """Distributional summary of a series."""

    series_id: str
    stats: ValueStats
    skewness: float
    kurtosis: float = Field(description="Excess kurtosis (normal distribution = 0).")
    first_timestamp: str
    last_timestamp: str
    inferred_freq: str | None


class RollingStatPreview(BaseModel):
    """Summary of one rolling statistic over the whole series."""

    stat: str
    min: float
    max: float
    last: float
    preview: list[Point] = Field(description="Evenly spaced sample of the rolling stat (capped).")


class RollingStatsResult(BaseModel):
    """Rolling-window statistics."""

    series_id: str
    window: int
    stats: list[RollingStatPreview]


class Gap(BaseModel):
    """A run of missing observations relative to the expected sampling interval."""

    start: str
    end: str
    expected_points_missed: int


class DataQualityReport(BaseModel):
    """Sensor-style data-quality audit of a series."""

    series_id: str
    n_points: int
    missing_values: int
    duplicate_timestamps: int
    is_monotonic: bool = Field(description="Whether timestamps are strictly increasing.")
    median_interval_seconds: float | None
    sampling_regularity_pct: float | None = Field(
        description="Percent of intervals within 10% of the median interval. Null for n < 3."
    )
    gaps: list[Gap] = Field(description="Largest gaps first, capped at 20.")
    n_gaps_total: int
    verdict: str = Field(description="One-line plain-language assessment.")


class Anomaly(BaseModel):
    """One anomalous observation."""

    timestamp: str
    value: float
    score: float = Field(description="Method-specific severity; higher = more anomalous.")


class AnomalyReport(BaseModel):
    """Anomaly-detection result."""

    series_id: str
    method: str
    threshold: float
    n_anomalies: int
    anomalies: list[Anomaly] = Field(description="Highest scores first, capped at 50.")
    baseline_mean: float
    baseline_std: float
    notes: str


class Changepoint(BaseModel):
    """A detected shift in the level of the series."""

    timestamp: str
    index: int
    mean_before: float
    mean_after: float
    delta: float
    score: float = Field(description="Normalized CUSUM statistic at the split.")


class ChangepointReport(BaseModel):
    """Level-shift detection result (binary segmentation over a CUSUM statistic)."""

    series_id: str
    method: str
    n_changepoints: int
    changepoints: list[Changepoint]
    min_segment_length: int
    threshold: float
    notes: str


class ComponentSummary(BaseModel):
    """Range summary of one decomposition component."""

    component: str
    min: float
    max: float
    preview: list[Point] = Field(description="Evenly spaced sample of the component (capped).")


class DecompositionReport(BaseModel):
    """Seasonal decomposition result with strength diagnostics."""

    series_id: str
    method: str
    period: int
    trend_strength: float = Field(description="0-1; Hyndman F_T = max(0, 1 - Var(resid)/Var(trend+resid)).")
    seasonal_strength: float = Field(description="0-1; Hyndman F_S = max(0, 1 - Var(resid)/Var(seasonal+resid)).")
    components: list[ComponentSummary]
    interpretation: str


class HypothesisTest(BaseModel):
    """A single statistical test outcome."""

    test: str
    statistic: float
    p_value: float
    conclusion: str


class StationarityReport(BaseModel):
    """ADF + KPSS stationarity assessment."""

    series_id: str
    adf: HypothesisTest
    kpss: HypothesisTest
    verdict: str = Field(description="Combined reading of both tests.")
    differencing_hint: str


class AutocorrelationReport(BaseModel):
    """ACF/PACF structure of a series."""

    series_id: str
    nlags: int
    acf: list[float]
    pacf: list[float]
    confidence_bound: float = Field(description="±1.96/sqrt(n) significance band for the ACF.")
    significant_lags: list[int] = Field(description="Lags (>=1) where |ACF| exceeds the band.")
    suggested_period: int | None = Field(description="First strong non-trivial ACF peak, if any.")


class TrendReport(BaseModel):
    """Trend estimate via OLS, Theil-Sen, and the Mann-Kendall test."""

    series_id: str
    ols_slope_per_step: float
    ols_r_squared: float
    theil_sen_slope_per_step: float
    theil_sen_ci_low: float
    theil_sen_ci_high: float
    mann_kendall: HypothesisTest
    direction: str = Field(description="'increasing', 'decreasing', or 'no significant trend'.")
    change_over_span: float = Field(description="Theil-Sen slope × (n-1): total modeled change.")


class CompareReport(BaseModel):
    """Relationship between two series on their overlapping timestamps."""

    series_a: str
    series_b: str
    n_overlap: int
    pearson: HypothesisTest
    spearman: HypothesisTest
    best_lag: int = Field(description="Lag (in steps, b relative to a) maximizing |cross-correlation|.")
    ccf_at_best_lag: float
    interpretation: str


class ForecastPoint(BaseModel):
    """One forecast step with a 95% interval."""

    timestamp: str
    value: float
    lo: float
    hi: float


class BacktestMetrics(BaseModel):
    """Holdout accuracy of the chosen baseline method."""

    n_test: int
    mae: float
    rmse: float
    mape_pct: float | None = Field(description="Null when actuals contain zeros.")


class ForecastReport(BaseModel):
    """Baseline forecast (naive / seasonal-naive / drift / exponential smoothing)."""

    series_id: str
    method: str
    horizon: int
    forecasts: list[ForecastPoint]
    backtest: BacktestMetrics
    notes: str
