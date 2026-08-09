"""Reference MOSFET database loader."""

from __future__ import annotations

import json
from pathlib import Path

from ..core.spec import MosfetSpec


class DeviceDatabase:
    def __init__(self, path: str | None = None):
        source = Path(path) if path else Path(__file__).parent.parent / "data" / "devices.json"
        with source.open(encoding="utf-8") as f:
            data = json.load(f)
        self.metadata = data.get("metadata", {})
        self.primary = [MosfetSpec(**entry) for entry in data["primary_mosfets"]]
        self.sr = [MosfetSpec(**entry) for entry in data["sr_mosfets"]]

    def get_primary(self, part_number: str) -> MosfetSpec:
        return self._get(self.primary, part_number)

    def get_sr(self, part_number: str) -> MosfetSpec:
        return self._get(self.sr, part_number)

    @staticmethod
    def _get(records: list[MosfetSpec], part_number: str) -> MosfetSpec:
        for record in records:
            if record.part_number.casefold() == part_number.casefold():
                return record
        choices = ", ".join(r.part_number for r in records)
        raise KeyError(f"device '{part_number}' not found; choices: {choices}")
