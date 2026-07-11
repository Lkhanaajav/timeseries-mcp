# timeseries-mcp

**Deterministic time-series statistics for AI agents.** An MCP server that gives any LLM agent unit-tested statistical tools — anomaly detection, changepoint detection, seasonal decomposition, stationarity/trend tests, data-quality audits, baseline forecasts — with schema-validated structured output and **no arbitrary code execution**.

[![CI](https://github.com/Lkhanaajav/timeseries-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/Lkhanaajav/timeseries-mcp/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)

```
Agent: "Is anything wrong with the server room this week?"

  load_csv(server_room_temp.csv)          → ts1: 1992 points @ 5min
  data_quality(ts1)                       → 1 sampling gap (2h, ~24 points missed)
  detect_anomalies(ts1, stl_residual,     → 3 spikes flagged, seasonal-context aware
                   period=288)
  detect_changepoints(ts1_daily)          → level shift on day 7: 21.4°C → 23.4°C
  forecast_baseline(ts1, seasonal_naive)  → next hour ± honest backtest error

Agent: "There's a 2-hour telemetry gap on June 4, three temperature spikes,
        and a sustained +2°C shift starting June 7 — likely HVAC degradation.
        Baseline forecast error is MAE 2.2°C, so alert thresholds under 3°C
        will false-positive."
```

## Why this exists

LLMs are unreliable at arithmetic over long arrays, and the common workaround — handing the model a Python sandbox — is a non-starter in locked-down environments and unauditable everywhere else. The existing "data analysis" MCP servers are mostly `run_script` shims: the model writes pandas code, executes it server-side, and hopes.

This server takes the opposite position:

- **Deterministic** — same input, same output, every time. Every number comes from a unit-tested routine (57 tests), not model-generated code.
- **No code execution** — the tool surface is 17 typed functions. There is nothing to inject into. Safe for enterprise hosts that cannot allow `exec()`.
- **Schema-validated** — every tool returns a Pydantic model published as an MCP `outputSchema`, so hosts get structured content they can verify, log, and post-process.
- **Token-frugal by design** — data loads once into a server-side registry and gets a handle (`ts1`). A million-point series never enters the model's context; every response is capped and previews are evenly thinned.

## Tools

| Tool | What it does |
|---|---|
| `load_csv` / `load_values` / `load_sample` | Register a series, get a handle + summary stats back |
| `list_series` / `describe` / `get_window` | Catalog, distribution summary, capped raw windows |
| `resample` / `rolling_stats` | Regularize onto a grid; rolling mean/std/min/max/median |
| `data_quality` | Gaps, duplicate timestamps, missing values, sampling regularity |
| `detect_anomalies` | `zscore`, `mad` (robust), `iqr`, `stl_residual` (seasonal-context) |
| `detect_changepoints` | Level shifts via CUSUM binary segmentation, MAD-robust noise scale |
| `decompose` | STL / classical split + Hyndman trend/seasonal strength (0–1) |
| `stationarity` | ADF + KPSS read together, combined verdict + differencing hint |
| `autocorrelation` | ACF/PACF, significance bounds, seasonal-period suggestion |
| `trend_test` | OLS + robust Theil-Sen + Mann-Kendall (tie-corrected) |
| `compare_series` | Pearson/Spearman on shared timestamps + best lead/lag scan |
| `forecast_baseline` | naive / seasonal-naive / drift / SES, 95% intervals, **holdout backtest included** |

Plus MCP resources (`timeseries://catalog`, `timeseries://{id}/summary`) and a guided `analyze_series` prompt.

Statistical choices worth noting: anomaly scores are method-honest (MAD falls back with an explanation when 50%+ of values tie); changepoint noise is estimated from first differences so the shifts being hunted don't inflate their own denominator; every forecast ships with a real holdout backtest because a baseline you can't beat is information.

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

**Claude Code**

```bash
claude mcp add timeseries -- uvx --from git+https://github.com/Lkhanaajav/timeseries-mcp timeseries-mcp
```

**Claude Desktop / Cursor** (`claude_desktop_config.json` / `mcp.json`)

```json
{
  "mcpServers": {
    "timeseries": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/Lkhanaajav/timeseries-mcp", "timeseries-mcp"],
      "env": { "TIMESERIES_MCP_DATA_ROOT": "/path/to/your/csv/files" }
    }
  }
}
```

**Streamable HTTP** (remote / multi-client)

```bash
uvx --from git+https://github.com/Lkhanaajav/timeseries-mcp timeseries-mcp --transport http --port 8000
```

**Try it without an MCP host** — the example walkthrough runs the full agent workflow over the in-memory transport, no API key needed:

```bash
git clone https://github.com/Lkhanaajav/timeseries-mcp && cd timeseries-mcp
uv sync && uv run python examples/demo.py
```

## Architecture

```
MCP host (Claude Code / Desktop / Cursor / any client)
    │  stdio or Streamable HTTP
    ▼
FastMCP server — 17 typed tools, 2 resources, 1 prompt
    │  series handles (ts1, ts2, ...) — raw data never re-enters context
    ▼
SeriesStore ── path-sandboxed CSV loader (TIMESERIES_MCP_DATA_ROOT)
    │
    ▼
analysis/ — pure, deterministic, unit-tested routines
    anomalies · changepoints · decompose · stationarity
    correlation · trend · quality · baselines
    (numpy / scipy / statsmodels underneath)
```

Tool logic is transport-agnostic and per-session state is a single registry object — aligned with where the MCP spec is heading (stateless Streamable HTTP core in the 2026-07-28 revision).

## Security posture

- **No code execution.** No `eval`, no `exec`, no model-written scripts.
- **Filesystem sandbox.** `load_csv` resolves paths against `TIMESERIES_MCP_DATA_ROOT` (default: the server's working directory) and refuses traversal outside it — tested, including absolute-path escapes.
- **No network access.** The server reads local CSVs and inline arrays only; no URL fetching, no SSRF surface.
- **Bounded everything.** Row caps on ingestion, point caps on every response, series-count caps on the registry.
- **Self-correcting errors.** Invalid inputs return actionable tool errors (`Unknown series_id 'ts9'. Known ids: ts1, ts2.`) so agents recover instead of hallucinating.

## Testing

```bash
uv run pytest        # 57 tests, ~2s
```

- **Golden statistical tests** — injected spikes are found, known slopes are recovered within tolerance, random walks fail stationarity, seasonal-naive beats naive on seasonal data.
- **Behavioral contrasts** — a value that is globally unremarkable but wrong for its phase of the daily cycle is caught by `stl_residual` and correctly *not* caught by global z-score.
- **Protocol tests** — the full workflow runs over the real MCP transport in memory; every tool is asserted to publish an `outputSchema`; error paths surface as MCP tool errors, not crashes.

## Honest limitations

- Changepoint detection assumes shifts-plus-noise; on strongly seasonal or trending series, `decompose` or `resample` first (the sample demo shows this workflow).
- Forecasts are reference baselines, deliberately. If your ARIMA can't beat `seasonal_naive`'s backtest here, it's not adding value.
- The series registry is in-process memory: restart = clean slate, and horizontal HTTP scaling would need a shared store (roadmap).
- No multivariate methods yet beyond pairwise comparison.

## Related work

[mcp-server-data-exploration](https://github.com/reading-plus-ai/mcp-server-data-exploration) and [pandas-mcp-server](https://github.com/marlonluo2018/pandas-mcp-server) take the code-execution route — maximum flexibility, minimum auditability. Vendor servers like [InfluxDB MCP](https://www.influxdata.com/blog/influxdb-mcp-server/) front their own databases. This server is the deterministic, self-contained middle: bring a CSV, get defensible statistics.

An agent-facing evaluation suite for this server — scoring whether agents pick the right tools with the right arguments — lives at [mcp-trajectory-evals](https://github.com/Lkhanaajav/mcp-trajectory-evals).

## Development notes

Built with AI assistance (Claude Code) for scaffolding and test generation; statistical method selection, API design, parameter defaults, and final review are mine. Notable choices I'd defend in review: MAD-of-differences noise estimation for CUSUM (a global σ is inflated by the shifts being detected), reading ADF and KPSS jointly rather than either alone, and refusing to ship forecasts without a holdout backtest.

MIT © Lkhanaajav Mijiddorj
