"""End-to-end walkthrough over the in-memory MCP transport.

Runs the exact tool sequence an AI agent would: load -> audit -> detect ->
forecast, and prints the structured results. No API key needed.

    uv run python examples/demo.py
"""

import asyncio
import json

from fastmcp import Client

from timeseries_mcp.server import mcp


async def main() -> None:
    async with Client(mcp) as client:
        print("=== load_sample('server_room_temp') ===")
        loaded = await client.call_tool("load_sample", {"name": "server_room_temp"})
        sid = loaded.data.series_id
        print(f"  {sid}: {loaded.data.n_points} points, {loaded.data.start} -> {loaded.data.end}")

        print("\n=== data_quality ===")
        dq = await client.call_tool("data_quality", {"series_id": sid})
        print(f"  {dq.data.verdict}")
        for gap in dq.data.gaps:
            print(f"  gap: {gap.start} -> {gap.end} (~{gap.expected_points_missed} points missed)")

        print("\n=== detect_anomalies (seasonal-aware, period=288) ===")
        an = await client.call_tool(
            "detect_anomalies",
            {"series_id": sid, "method": "stl_residual", "period": 288, "threshold": 4.0},
        )
        print(f"  {an.data.n_anomalies} anomalies; top 3:")
        for a in an.data.anomalies[:3]:
            print(f"    {a.timestamp}  value={a.value:.2f}  score={a.score:.1f}")

        # Changepoint detection assumes no strong seasonality — the daily cycle
        # would read as endless "shifts". Correct workflow: resample first.
        print("\n=== detect_changepoints (cpu_utilization, resampled to daily) ===")
        cpu = await client.call_tool("load_sample", {"name": "cpu_utilization"})
        daily = await client.call_tool(
            "resample", {"series_id": cpu.data.series_id, "rule": "1D", "agg": "mean"}
        )
        cp = await client.call_tool(
            "detect_changepoints",
            {"series_id": daily.data.series_id, "min_segment_length": 4},
        )
        for c in cp.data.changepoints:
            print(f"  level shift at {c.timestamp}: {c.mean_before:.2f} -> {c.mean_after:.2f}")

        print("\n=== forecast_baseline (seasonal_naive, 1 hour ahead) ===")
        fc = await client.call_tool(
            "forecast_baseline",
            {"series_id": sid, "method": "seasonal_naive", "period": 288, "horizon": 12},
        )
        bt = fc.data.backtest
        print(f"  backtest on last {bt.n_test} points: MAE={bt.mae:.3f}, RMSE={bt.rmse:.3f}")
        first = fc.data.forecasts[0]
        print(f"  next point {first.timestamp}: {first.value:.2f} [{first.lo:.2f}, {first.hi:.2f}]")

        print("\n=== structured output (raw JSON the host receives) ===")
        print(json.dumps(fc.structured_content["forecasts"][0], indent=2))


if __name__ == "__main__":
    asyncio.run(main())
