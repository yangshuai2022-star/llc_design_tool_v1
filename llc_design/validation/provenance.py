from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class EvidenceStatus(str, Enum):
    VERIFIED = "VERIFIED"
    UNKNOWN = "UNKNOWN"


class ReferenceDataNotReleaseReady(RuntimeError):
    pass


@dataclass(frozen=True)
class ProvenanceIssue:
    dataset: str
    record: str | None
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetProvenance:
    dataset: str
    source: str
    provenance: str
    status: EvidenceStatus
    verified_for_hardware: bool
    record_count: int

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["status"] = self.status.value
        return result


@dataclass(frozen=True)
class ProvenanceReport:
    datasets: tuple[DatasetProvenance, ...]
    issues: tuple[ProvenanceIssue, ...]

    @property
    def valid(self) -> bool:
        return not self.issues

    @property
    def hardware_release_ready(self) -> bool:
        return self.valid and bool(self.datasets) and all(
            item.verified_for_hardware and item.status is EvidenceStatus.VERIFIED
            for item in self.datasets
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "hardware_release_ready": self.hardware_release_ready,
            "datasets": [item.to_dict() for item in self.datasets],
            "issues": [item.to_dict() for item in self.issues],
        }


_DATASETS = {
    "devices.json": ("primary_mosfets", "sr_mosfets"),
    "cores.json": ("cores",),
    "materials.json": ("materials",),
    "transformer_core_presets.json": ("presets",),
}
_REQUIRED_METADATA = ("source", "provenance", "status", "verified_for_hardware")


def _record_name(record: dict[str, Any]) -> str:
    for key in ("part_number", "key", "preset_key"):
        if key in record:
            return str(record[key])
    return "<unnamed>"


def _iter_records(data: dict[str, Any], collections: Iterable[str]) -> Iterable[dict[str, Any]]:
    for collection in collections:
        records = data[collection]
        yield from records


def validate_bundled_data(data_directory: str | Path | None = None) -> ProvenanceReport:
    base = Path(data_directory) if data_directory else Path(__file__).parent.parent / "data"
    datasets: list[DatasetProvenance] = []
    issues: list[ProvenanceIssue] = []

    for filename, collections in _DATASETS.items():
        path = base / filename
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(ProvenanceIssue(filename, None, f"cannot load dataset: {exc}"))
            continue

        metadata = data.get("metadata")
        if not isinstance(metadata, dict):
            issues.append(ProvenanceIssue(filename, None, "metadata must be an object"))
            continue
        for field in _REQUIRED_METADATA:
            if field not in metadata:
                issues.append(ProvenanceIssue(filename, None, f"missing metadata.{field}"))

        source = metadata.get("source")
        provenance = metadata.get("provenance")
        status_value = metadata.get("status")
        verified = metadata.get("verified_for_hardware")
        if not isinstance(source, str) or not source.strip():
            issues.append(ProvenanceIssue(filename, None, "metadata.source must be non-empty"))
        if not isinstance(provenance, str) or not provenance.strip():
            issues.append(ProvenanceIssue(filename, None, "metadata.provenance must be non-empty"))
        try:
            status = EvidenceStatus(status_value)
        except (TypeError, ValueError):
            status = EvidenceStatus.UNKNOWN
            issues.append(ProvenanceIssue(filename, None, "metadata.status must be VERIFIED or UNKNOWN"))
        if not isinstance(verified, bool):
            issues.append(ProvenanceIssue(filename, None, "metadata.verified_for_hardware must be boolean"))
            verified = False
        if verified and status is not EvidenceStatus.VERIFIED:
            issues.append(ProvenanceIssue(
                filename, None, "hardware verification requires status VERIFIED"
            ))

        valid_collections = True
        for collection in collections:
            records_value = data.get(collection)
            if not isinstance(records_value, list):
                issues.append(ProvenanceIssue(
                    filename, None, f"{collection} must be a list"
                ))
                valid_collections = False
            elif any(not isinstance(record, dict) for record in records_value):
                issues.append(ProvenanceIssue(
                    filename, None, f"{collection} entries must be objects"
                ))
                valid_collections = False
        records = tuple(_iter_records(data, collections)) if valid_collections else ()
        for record in records:
            record_status = record.get("status", status.value)
            record_verified = record.get("verified_for_hardware", verified)
            if not isinstance(record_status, str) or record_status not in {
                item.value for item in EvidenceStatus
            }:
                issues.append(ProvenanceIssue(
                    filename, _record_name(record), "status override must be VERIFIED or UNKNOWN"
                ))
            if not isinstance(record_verified, bool):
                issues.append(ProvenanceIssue(
                    filename, _record_name(record), "verified_for_hardware override must be boolean"
                ))
            elif record_verified and record_status != EvidenceStatus.VERIFIED.value:
                issues.append(ProvenanceIssue(
                    filename, _record_name(record),
                    "hardware verification requires status VERIFIED",
                ))
        datasets.append(DatasetProvenance(
            dataset=filename,
            source=source if isinstance(source, str) else "",
            provenance=provenance if isinstance(provenance, str) else "",
            status=status,
            verified_for_hardware=bool(verified),
            record_count=len(records),
        ))

    return ProvenanceReport(tuple(datasets), tuple(issues))


def assert_hardware_release_ready(
    report: ProvenanceReport | None = None,
) -> ProvenanceReport:
    checked = report or validate_bundled_data()
    if not checked.hardware_release_ready:
        blocked = ", ".join(
            item.dataset for item in checked.datasets
            if not item.verified_for_hardware or item.status is not EvidenceStatus.VERIFIED
        )
        detail = blocked or "provenance validation errors"
        raise ReferenceDataNotReleaseReady(
            f"hardware release blocked by non-verified reference data: {detail}"
        )
    return checked
