"""Deterministic synthetic sample datasets (seeded), so demos and evals reproduce exactly.

Shapes are modeled on real IoT/ops telemetry: daily cycles, drift, spikes,
level shifts from equipment changes, and transmission gaps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SAMPLES = ("server_room_temp", "cpu_utilization", "methane_ppm")


def make(name: str) -> pd.Series:
    if name == "server_room_temp":
        return _server_room_temp()
    if name == "cpu_utilization":
        return _cpu_utilization()
    if name == "methane_ppm":
        return _methane_ppm()
    raise ValueError(f"Unknown sample '{name}'. Available: {SAMPLES}.")


def _server_room_temp() -> pd.Series:
    """7 days at 5-min sampling: daily cycle + mild trend + 3 spikes + HVAC-failure shift + a gap."""
    rng = np.random.default_rng(42)
    n = 7 * 288  # 288 five-minute samples/day
    t = np.arange(n)
    daily = 2.5 * np.sin(2 * np.pi * t / 288 - np.pi / 2)
    trend = 0.0008 * t
    noise = rng.normal(0, 0.3, n)
    values = 21.0 + daily + trend + noise
    values[400] += 6.0  # spot-cooler trip
    values[900] -= 5.0  # door left open
    values[1500] += 7.5  # sensor glitch
    values[1728:] += 2.0  # HVAC degradation on day 7 (level shift)
    index = pd.date_range("2026-06-01", periods=n, freq="5min")
    series = pd.Series(values, index=index)
    return series.drop(series.index[1000:1024])  # 2-hour transmission gap


def _cpu_utilization() -> pd.Series:
    """30 days hourly: mild weekday/weekend cycle + bad-deploy plateau + 2 saturation spikes."""
    rng = np.random.default_rng(7)
    n = 30 * 24
    t = np.arange(n)
    hour = t % 24
    weekday = (t // 24) % 7 < 5
    base = np.where(weekday, 45.0, 38.0) + 10.0 * np.exp(-((hour - 14) ** 2) / 18.0)
    noise = rng.normal(0, 3.0, n)
    values = np.clip(base + noise, 0, 100)
    values[336:456] += 22.0  # bad deploy on day 14, rolled back 5 days later
    values[550] = 99.5
    values[551] = 98.0
    index = pd.date_range("2026-05-01", periods=n, freq="1h")
    return pd.Series(values, index=index)


def _methane_ppm() -> pd.Series:
    """14 days at 15-min sampling: sensor drift + temperature cross-sensitivity + calibration reset."""
    rng = np.random.default_rng(13)
    n = 14 * 96
    t = np.arange(n)
    drift = 0.004 * t  # uncalibrated NDIR drift
    daily = 0.4 * np.sin(2 * np.pi * t / 96)
    noise = rng.normal(0, 0.15, n)
    values = 2.0 + drift + daily + noise
    values[672:] -= drift[672]  # field recalibration at day 7
    values[1100] += 3.2  # leak event
    index = pd.date_range("2026-04-01", periods=n, freq="15min")
    return pd.Series(values, index=index)
