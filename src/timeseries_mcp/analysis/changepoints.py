"""Level-shift detection: binary segmentation over a standardized CUSUM statistic.

For a segment x[a:b) the statistic is max_k |S_k| / (sigma * sqrt(n)) where
S_k is the centered cumulative sum. Under the no-change hypothesis this behaves
like the sup of a Brownian bridge, so 1.36 approximates a 95% significance
bound (Kolmogorov-Smirnov). Deterministic, O(n log n)-ish, no dependencies.

Sigma is estimated robustly from first differences (MAD-based, Donoho-style):
a global standard deviation would be inflated by the very level shifts we are
trying to detect, costing power on short shifted segments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..models import Changepoint, ChangepointReport
from ..store import clean_values


def detect(
    s: pd.Series,
    series_id: str,
    threshold: float = 1.36,
    min_segment_length: int = 10,
    max_changepoints: int = 10,
) -> ChangepointReport:
    clean = clean_values(s, min_points=2 * min_segment_length, context="Changepoint detection")
    values = clean.to_numpy()
    sigma = _noise_sigma(values)

    splits: list[tuple[int, float]] = []
    if sigma > 0:
        _segment(values, 0, len(values), threshold, min_segment_length, sigma, splits)
    splits.sort(key=lambda t: -t[1])
    splits = sorted(splits[:max_changepoints])

    boundaries = [0] + [i for i, _ in splits] + [len(values)]
    changepoints = []
    for rank, (idx, score) in enumerate(splits):
        seg_start = boundaries[rank]
        seg_end = boundaries[rank + 2]
        before = values[seg_start:idx]
        after = values[idx:seg_end]
        changepoints.append(
            Changepoint(
                timestamp=clean.index[idx].isoformat(),
                index=int(idx),
                mean_before=round(float(before.mean()), 6),
                mean_after=round(float(after.mean()), 6),
                delta=round(float(after.mean() - before.mean()), 6),
                score=round(float(score), 4),
            )
        )

    notes = (
        "Binary segmentation over standardized CUSUM (noise sigma from MAD of first "
        "differences); threshold 1.36 ~ 95% significance. Detects mean shifts only — "
        "on strongly seasonal or trending data, decompose or resample first."
    )
    if sigma == 0:
        notes = "Series is constant; no changepoints possible."
    return ChangepointReport(
        series_id=series_id,
        method="cusum_binseg",
        n_changepoints=len(changepoints),
        changepoints=changepoints,
        min_segment_length=min_segment_length,
        threshold=threshold,
        notes=notes,
    )


def _noise_sigma(values: np.ndarray) -> float:
    """Robust noise scale: 1.4826 * MAD(diff) / sqrt(2), immune to level shifts.

    Falls back to the global std when differences are degenerate (e.g. a
    perfect staircase where most diffs are identical).
    """
    d = np.diff(values)
    if len(d) == 0:
        return 0.0
    mad = np.median(np.abs(d - np.median(d)))
    sigma = 1.4826 * mad / np.sqrt(2.0)
    if sigma == 0:
        sigma = float(values.std(ddof=1))
    return float(sigma)


def _segment(
    values: np.ndarray,
    start: int,
    end: int,
    threshold: float,
    min_len: int,
    sigma: float,
    out: list[tuple[int, float]],
) -> None:
    n = end - start
    if n < 2 * min_len:
        return
    x = values[start:end]
    centered = np.cumsum(x - x.mean())
    # Candidate splits keep both children >= min_len.
    lo, hi = min_len - 1, n - min_len
    if hi <= lo:
        return
    window = np.abs(centered[lo:hi])
    k = int(np.argmax(window)) + lo
    stat = float(window[k - lo] / (sigma * np.sqrt(n)))
    if stat <= threshold:
        return
    split = start + k + 1  # first index of the "after" segment
    out.append((split, stat))
    _segment(values, start, split, threshold, min_len, sigma, out)
    _segment(values, split, end, threshold, min_len, sigma, out)
