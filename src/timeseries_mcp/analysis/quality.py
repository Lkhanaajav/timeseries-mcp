"""Sensor-style data-quality audit: gaps, duplicates, sampling regularity."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..models import DataQualityReport, Gap

MAX_GAPS_REPORTED = 20
GAP_FACTOR = 1.5  # an interval > 1.5x the median counts as a gap
REGULAR_TOLERANCE = 0.10


def audit(s: pd.Series, series_id: str) -> DataQualityReport:
    n = len(s)
    missing = int(s.isna().sum())
    duplicates = int(s.index.duplicated().sum())
    strictly_increasing = bool(s.index.is_monotonic_increasing and s.index.is_unique)

    median_interval: float | None = None
    regularity: float | None = None
    gaps: list[Gap] = []
    n_gaps = 0

    if n >= 3:
        # total_seconds() is resolution-safe: pandas indexes may carry s/us/ns units.
        deltas = (s.index[1:] - s.index[:-1]).total_seconds().to_numpy()
        median_interval = float(np.median(deltas))
        if median_interval > 0:
            within = np.abs(deltas - median_interval) <= REGULAR_TOLERANCE * median_interval
            regularity = float(100.0 * within.mean())
            gap_idx = np.flatnonzero(deltas > GAP_FACTOR * median_interval)
            n_gaps = int(len(gap_idx))
            largest = gap_idx[np.argsort(deltas[gap_idx])[::-1][:MAX_GAPS_REPORTED]]
            gaps = [
                Gap(
                    start=s.index[i].isoformat(),
                    end=s.index[i + 1].isoformat(),
                    expected_points_missed=int(round(deltas[i] / median_interval)) - 1,
                )
                for i in sorted(largest)
            ]

    issues = []
    if missing:
        issues.append(f"{missing} missing values")
    if duplicates:
        issues.append(f"{duplicates} duplicate timestamps")
    if n_gaps:
        issues.append(f"{n_gaps} sampling gaps")
    if not strictly_increasing:
        issues.append("timestamps not strictly increasing")
    if regularity is not None and regularity < 90 and not n_gaps:
        issues.append(f"irregular sampling ({regularity:.0f}% of intervals near the median)")

    verdict = "Clean: regular sampling, no gaps, no duplicates, no missing values." if not issues else (
        "Issues found: " + "; ".join(issues) + "."
    )
    return DataQualityReport(
        series_id=series_id,
        n_points=n,
        missing_values=missing,
        duplicate_timestamps=duplicates,
        is_monotonic=strictly_increasing,
        median_interval_seconds=median_interval,
        sampling_regularity_pct=round(regularity, 2) if regularity is not None else None,
        gaps=gaps,
        n_gaps_total=n_gaps,
        verdict=verdict,
    )
