"""MCP v2 server for Power Design Toolkit.

The MCP layer is deliberately thin: it exposes deterministic functions from
``power_agent.tools`` instead of embedding engineering equations in protocol
handlers.
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server import MCPServer

from .tools import (
    llc_analyze as _llc_analyze,
    llc_defaults as _llc_defaults,
    toolkit_capabilities as _toolkit_capabilities,
    validate_engineering_data as _validate_engineering_data,
)

mcp = MCPServer("Power Design Toolkit")


@mcp.tool()
def toolkit_capabilities() -> dict[str, Any]:
    """Describe available engineering tools and their evidence boundaries."""

    return _toolkit_capabilities()


@mcp.tool()
def llc_defaults() -> dict[str, Any]:
    """Return bounded LLC defaults and supported public design parameters."""

    return _llc_defaults()


@mcp.tool()
def llc_analyze(spec: dict[str, Any]) -> dict[str, Any]:
    """Analyze one LLC design with the shared deterministic engineering kernel."""

    return _llc_analyze(spec)


@mcp.tool()
def validate_engineering_data() -> dict[str, Any]:
    """Check bundled engineering-data provenance and hardware-release gates."""

    return _validate_engineering_data()


@mcp.resource("power-design://capabilities")
def capabilities_resource() -> str:
    """Machine-readable Power Design Toolkit capability manifest."""

    return json.dumps(_toolkit_capabilities(), ensure_ascii=False, indent=2)


def main() -> None:
    """Run the MCP server over stdio, the safest default for local Agent use."""

    mcp.run()


if __name__ == "__main__":
    main()
