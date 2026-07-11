"""Generate the README charts (light + dark) from the library's real detections.

Nothing here is drawn by hand: the anomaly markers, gap band, and changepoint
segments are the actual outputs of detect_anomalies, data_quality, and
detect_changepoints on the bundled sample datasets.

Run from the repo root (matplotlib is not a package dependency):

    uv run --with matplotlib python examples/make_readme_charts.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from timeseries_mcp.analysis import anomalies, changepoints, quality
from timeseries_mcp.datasets import make

OUT = Path(__file__).parent.parent / "docs" / "charts"

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "baseline": "#c3c2b7",
        "series": "#2a78d6",
        "critical": "#d03b3b",
        "band": "#e1e0d9",
        "segment": "#52514e",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "baseline": "#383835",
        "series": "#3987e5",
        "critical": "#d03b3b",
        "band": "#2c2c2a",
        "segment": "#c3c2b7",
    },
}


def style_axis(ax, t):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["baseline"])
    ax.tick_params(colors=t["muted"], labelsize=9)
    ax.grid(True, axis="y", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))


def render(mode: str) -> None:
    t = THEMES[mode]
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9.6, 6.4), dpi=160, facecolor=t["surface"],
        gridspec_kw={"hspace": 0.52},
    )

    # -- Panel A: seasonal-context anomalies + telemetry gap ----------------
    temp = make("server_room_temp")
    report = anomalies.detect(temp, "ts", method="stl_residual", threshold=4.0, period=288)
    gap = quality.audit(temp, "ts").gaps[0]

    ax1.plot(temp.index, temp.to_numpy(), color=t["series"], linewidth=1.4, zorder=2)
    ax1.axvspan(
        pd.Timestamp(gap.start), pd.Timestamp(gap.end),
        color=t["band"], alpha=0.9 if mode == "light" else 0.7, zorder=1,
    )
    xs = [pd.Timestamp(a.timestamp) for a in report.anomalies]
    ys = [a.value for a in report.anomalies]
    ax1.scatter(xs, ys, s=46, color=t["critical"], zorder=3,
                edgecolors=t["surface"], linewidths=1.4)
    top = report.anomalies[0]
    ax1.annotate(
        f"anomaly  score {top.score:.0f}",
        xy=(pd.Timestamp(top.timestamp), top.value),
        xytext=(14, 6), textcoords="offset points",
        fontsize=9, color=t["ink"],
        arrowprops={"arrowstyle": "-", "color": t["muted"], "linewidth": 0.8},
    )
    ax1.annotate(
        "2h telemetry gap",
        xy=(pd.Timestamp(gap.start), float(temp.max()) - 0.4),
        xytext=(-6, 0), textcoords="offset points",
        fontsize=9, color=t["muted"], ha="right", va="top",
    )
    ax1.set_title(
        "detect_anomalies(method=stl_residual, period=288) — server_room_temp sample",
        loc="left", fontsize=10.5, color=t["ink"], pad=10,
    )
    ax1.set_ylabel("°C", color=t["muted"], fontsize=9)
    style_axis(ax1, t)

    # -- Panel B: level shifts on daily CPU means ---------------------------
    daily = make("cpu_utilization").resample("1D").mean().dropna()
    cps = changepoints.detect(daily, "d", min_segment_length=4)

    ax2.plot(daily.index, daily.to_numpy(), color=t["series"], linewidth=1.6, zorder=2)
    boundaries = (
        [daily.index[0]]
        + [pd.Timestamp(c.timestamp) for c in cps.changepoints]
        + [daily.index[-1]]
    )
    values = daily.to_numpy()
    idx = [0] + [c.index for c in cps.changepoints] + [len(values)]
    for k in range(len(idx) - 1):
        seg_mean = values[idx[k]:idx[k + 1]].mean()
        ax2.hlines(seg_mean, boundaries[k], boundaries[k + 1],
                   color=t["segment"], linewidth=1.2, linestyle=(0, (4, 3)), zorder=3)
    deploy = max(cps.changepoints, key=lambda c: c.delta)
    ax2.annotate(
        f"level shift  +{deploy.delta:.0f} (bad deploy)",
        xy=(pd.Timestamp(deploy.timestamp), deploy.mean_after),
        xytext=(10, 10), textcoords="offset points",
        fontsize=9, color=t["ink"],
        arrowprops={"arrowstyle": "-", "color": t["muted"], "linewidth": 0.8},
    )
    ax2.set_title(
        "detect_changepoints(min_segment_length=4) — cpu_utilization, daily means",
        loc="left", fontsize=10.5, color=t["ink"], pad=10,
    )
    ax2.set_ylabel("% CPU", color=t["muted"], fontsize=9)
    style_axis(ax2, t)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"detections_{mode}.png"
    fig.savefig(path, facecolor=t["surface"], bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    for mode in THEMES:
        render(mode)
