"""End-to-end tests over the in-memory MCP transport — the real protocol path."""

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from timeseries_mcp.server import mcp


async def test_full_agent_workflow():
    """The workflow an agent would actually run: load -> audit -> analyze -> forecast."""
    async with Client(mcp) as client:
        loaded = await client.call_tool("load_sample", {"name": "server_room_temp"})
        sid = loaded.data.series_id
        assert loaded.data.n_points > 1900

        dq = await client.call_tool("data_quality", {"series_id": sid})
        assert dq.data.n_gaps_total >= 1  # the sample has an injected 2h gap

        anomalies = await client.call_tool(
            "detect_anomalies",
            {"series_id": sid, "method": "stl_residual", "period": 288, "threshold": 4.0},
        )
        assert anomalies.data.n_anomalies >= 3  # three injected spikes

        cps = await client.call_tool("detect_changepoints", {"series_id": sid})
        assert cps.data.n_changepoints >= 1  # injected HVAC level shift

        forecast = await client.call_tool(
            "forecast_baseline",
            {"series_id": sid, "method": "seasonal_naive", "period": 288, "horizon": 12},
        )
        assert len(forecast.data.forecasts) == 12
        assert forecast.data.backtest.mae > 0


async def test_all_seventeen_tools_registered():
    async with Client(mcp) as client:
        tools = await client.list_tools()
        names = {t.name for t in tools}
        expected = {
            "load_csv", "load_values", "load_sample", "list_series", "describe",
            "get_window", "resample", "rolling_stats", "data_quality",
            "detect_anomalies", "detect_changepoints", "decompose", "stationarity",
            "autocorrelation", "trend_test", "compare_series", "forecast_baseline",
        }
        assert expected <= names


async def test_every_tool_declares_output_schema():
    """The structured-output contract: every tool must publish an outputSchema."""
    async with Client(mcp) as client:
        for tool in await client.list_tools():
            assert tool.outputSchema is not None, f"{tool.name} lacks outputSchema"


async def test_unknown_series_id_is_actionable_tool_error():
    async with Client(mcp) as client:
        with pytest.raises(ToolError, match="Unknown series_id"):
            await client.call_tool("describe", {"series_id": "ts404"})


async def test_bad_method_is_tool_error_not_crash():
    async with Client(mcp) as client:
        await client.call_tool("load_values", {"values": [1.0, 2.0, 3.0] * 10})
        with pytest.raises(ToolError):
            await client.call_tool(
                "detect_anomalies", {"series_id": "ts1", "method": "quantum"}
            )


async def test_resample_creates_derived_series():
    async with Client(mcp) as client:
        await client.call_tool("load_sample", {"name": "cpu_utilization"})
        derived = await client.call_tool("resample", {"series_id": "ts1", "rule": "1D", "agg": "max"})
        assert derived.data.series_id == "ts2"
        assert derived.data.n_points == 30
        catalog = await client.call_tool("list_series", {})
        assert catalog.data.n_series == 2


async def test_get_window_thins_to_limit():
    async with Client(mcp) as client:
        await client.call_tool("load_sample", {"name": "methane_ppm"})
        window = await client.call_tool("get_window", {"series_id": "ts1", "limit": 50})
        assert window.data.returned <= 50
        assert window.data.truncated is True
        assert window.data.n_in_window == 14 * 96


async def test_catalog_resource_lists_series():
    async with Client(mcp) as client:
        await client.call_tool("load_sample", {"name": "server_room_temp"})
        content = await client.read_resource("timeseries://catalog")
        assert "ts1" in content[0].text
        assert "server_room_temp" in content[0].text


async def test_summary_resource():
    async with Client(mcp) as client:
        await client.call_tool("load_sample", {"name": "cpu_utilization"})
        content = await client.read_resource("timeseries://ts1/summary")
        assert "720 points" in content[0].text


async def test_analyze_prompt_available():
    async with Client(mcp) as client:
        prompts = await client.list_prompts()
        assert "analyze_series" in {p.name for p in prompts}
