"""Deterministic engineering tools exposed to Agent/MCP front ends.

LLMs may choose and explain these tools; they do not replace the underlying
numerical kernels or change the evidence status of their results.
"""

from __future__ import annotations

from typing import Any

from llc_design.validation.provenance import validate_bundled_data
from webapp.service import analyze_llc, default_payload


def toolkit_capabilities() -> dict[str, Any]:
    """Return machine-readable capability and evidence-boundary metadata."""

    return {
        "product": "Power Design Toolkit",
        "workspaces": [
            "LLC Design",
            "PFC Design",
            "Control Tools",
            "FRA Loop Designer",
        ],
        "agent_tools": [
            "toolkit_capabilities",
            "llc_defaults",
            "llc_analyze",
            "validate_engineering_data",
        ],
        "engineering_contract": {
            "llm_role": "interpret requirements, select tools, explain results",
            "kernel_role": "deterministic engineering computation",
            "evidence_statuses": ["VERIFIED", "UNKNOWN"],
            "hardware_rule": (
                "software/model evidence never implies hardware verification; "
                "verified_for_hardware is a separate gate"
            ),
        },
    }


def llc_defaults() -> dict[str, Any]:
    """Return the bounded LLC input schema/defaults used by the shared service."""

    return default_payload()


def llc_analyze(spec: dict[str, Any]) -> dict[str, Any]:
    """Run the shared LLC deterministic kernel and attach evidence semantics."""

    result = analyze_llc(spec)
    result["agent_evidence"] = {
        "status": "VERIFIED",
        "scope": "execution of the repository software model for the supplied input",
        "verified_for_hardware": False,
        "boundary": (
            "The returned result is a software-model calculation. Device/material "
            "data and bench correlation must be verified separately for hardware release."
        ),
    }
    return result


def validate_engineering_data() -> dict[str, Any]:
    """Return the current bundled LLC data-provenance report."""

    return validate_bundled_data().to_dict()
