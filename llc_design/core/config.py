"""JSON configuration import/export helpers."""

from __future__ import annotations

from dataclasses import asdict
from enum import Enum
import json
from pathlib import Path

from .spec import LLCDesignSpec, PrimaryTopology, SecondaryTopology


def _normalize(data: dict) -> dict:
    normalized = dict(data)
    if "primary_topology" in normalized:
        normalized["primary_topology"] = PrimaryTopology(normalized["primary_topology"])
    if "secondary_topology" in normalized:
        normalized["secondary_topology"] = SecondaryTopology(normalized["secondary_topology"])
    return normalized


def load_spec(path: str | Path) -> LLCDesignSpec:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "spec" in data and isinstance(data["spec"], dict):
        data = data["spec"]
    return LLCDesignSpec(**_normalize(data))


def save_spec(spec: LLCDesignSpec, path: str | Path) -> Path:
    data = asdict(spec)
    for key, value in list(data.items()):
        if isinstance(value, Enum):
            data[key] = value.value
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return output
