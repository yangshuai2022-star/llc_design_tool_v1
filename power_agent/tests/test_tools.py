from __future__ import annotations

from power_agent.tools import (
    llc_defaults,
    toolkit_capabilities,
    validate_engineering_data,
)


def test_capabilities_keep_llm_and_kernel_roles_separate() -> None:
    result = toolkit_capabilities()
    contract = result["engineering_contract"]
    assert "deterministic" in contract["kernel_role"]
    assert "hardware verification" in contract["hardware_rule"]


def test_llc_defaults_are_exposed_from_shared_service() -> None:
    result = llc_defaults()
    assert "spec" in result
    assert result["spec"]["vbus_nom_v"] > 0.0
    assert "FULL_BRIDGE" in result["topologies"]


def test_engineering_data_report_never_promotes_unknown_to_hardware_ready() -> None:
    result = validate_engineering_data()
    assert result["valid"]
    assert not result["hardware_release_ready"]
