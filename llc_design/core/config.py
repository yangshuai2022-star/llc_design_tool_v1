"""JSON configuration import/export helpers."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from llc_design import __version__

from .spec import LLCDesignSpec, PrimaryTopology, SecondaryTopology, TankParameterMode

PROJECT_SCHEMA_VERSION = 1


def _normalize(data: dict) -> dict:
    normalized = dict(data)
    if "primary_topology" in normalized:
        normalized["primary_topology"] = PrimaryTopology(normalized["primary_topology"])
    if "secondary_topology" in normalized:
        normalized["secondary_topology"] = SecondaryTopology(normalized["secondary_topology"])
    if "parameter_mode" in normalized:
        normalized["parameter_mode"] = TankParameterMode(normalized["parameter_mode"])
    return normalized


def load_spec(path: str | Path) -> LLCDesignSpec:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "spec" in data and isinstance(data["spec"], dict):
        data = data["spec"]
    return LLCDesignSpec(**_normalize(data))


def save_spec(spec: LLCDesignSpec, path: str | Path) -> Path:
    data = _spec_payload(spec)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def _spec_payload(spec: LLCDesignSpec) -> dict[str, Any]:
    data = asdict(spec)
    for key, value in list(data.items()):
        if isinstance(value, Enum):
            data[key] = value.value
    return data


def create_project_document(spec: LLCDesignSpec, analysis: Any | None = None) -> dict[str, Any]:
    from ..validation.provenance import validate_bundled_data

    analysis_state: dict[str, Any] = {"status": "not_run"}
    if analysis is not None:
        nominal = analysis.nominal
        analysis_state = {
            "status": "complete",
            "feasible": bool(analysis.feasible),
            "feasibility_reasons": list(analysis.feasibility_reasons),
            "warnings": list(analysis.warnings),
            "nominal": {
                "switching_frequency_hz": float(
                    nominal.operating_point.switching_frequency_hz
                ),
                "total_loss_w": float(nominal.total_loss_w),
                "efficiency": float(nominal.efficiency),
            },
        }

    return {
        "schema": "power-design-toolkit/project",
        "schema_version": PROJECT_SCHEMA_VERSION,
        "toolkit_version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "spec": _spec_payload(spec),
        "analysis": analysis_state,
        "data_provenance": validate_bundled_data().to_dict(),
    }


def save_project(spec: LLCDesignSpec, path: str | Path, analysis: Any | None = None) -> Path:
    data = create_project_document(spec, analysis)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return output
