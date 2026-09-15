from __future__ import annotations

import asyncio

from power_agent.mcp_server import mcp


def test_mcp_v2_server_registers_expected_tools() -> None:
    tools = asyncio.run(mcp.list_tools())
    names = {tool.name for tool in tools}
    assert {
        "toolkit_capabilities",
        "llc_defaults",
        "llc_analyze",
        "validate_engineering_data",
    } <= names
