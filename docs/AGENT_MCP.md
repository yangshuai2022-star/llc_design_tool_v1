# Agent / MCP interface

Power Design Toolkit exposes a thin MCP interface so an LLM can **select and explain deterministic engineering tools** without re-implementing power-electronics equations inside the model.

## Architecture

```text
ChatGPT / Claude / Codex / other MCP host
                 |
                 v
        Power Design MCP server
                 |
                 v
          power_agent.tools
                 |
                 v
      shared deterministic kernels
       (web/CLI/GUI reuse the same code)
```

The LLM is an orchestration and explanation layer.  Numerical authority remains the toolkit kernel and its traceable evidence.

## Install

```bash
python -m pip install -e ".[agent]"
```

The optional `agent` dependency intentionally targets the official MCP Python SDK v2 (`mcp>=2,<3`).

## Run locally over stdio

```bash
power-design-mcp
```

Stdio is the default because it gives local MCP hosts a small attack surface and does not expose an unauthenticated network service.

## Initial tools

### `toolkit_capabilities`
Returns the current workspace/tool manifest and the evidence contract.

### `llc_defaults`
Returns the same bounded LLC public-input defaults used by the Web/API service.

### `llc_analyze`
Runs the shared deterministic LLC analysis kernel.  Results include an `agent_evidence` block that states the exact software scope and keeps `verified_for_hardware=false`.

### `validate_engineering_data`
Runs the bundled LLC device/material provenance gate and reports whether the current reference datasets are structurally valid and whether they are hardware-release-ready.

## Resource

`power-design://capabilities` exposes the capability manifest as JSON.

## Design rules

1. **No duplicate equations in MCP handlers.** Add capabilities to shared kernels first, then expose them through `power_agent.tools`.
2. **No AI-created verification claims.** The model may summarize evidence but may not change `UNKNOWN` to `VERIFIED`.
3. **No hidden unit conversion.** Tool contracts should retain engineering units in field names or schema documentation.
4. **Bound expensive solvers.** Public/agent input surfaces must preserve the same range/complexity guards used by the Web API.
5. **Return machine-readable evidence.** Every future high-value calculation tool should identify model scope and hardware-verification status.

## Next MCP tools

The current server intentionally starts narrow.  Natural next additions are PFC/Vienna operating-point analysis, Control Tools controller synthesis, FRA import/redesign, and C99 generation.  Each should call its existing shared service/kernel rather than creating an Agent-only implementation.
