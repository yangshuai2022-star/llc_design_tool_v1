from __future__ import annotations

from typing import Any

from .baseline import ValidationCheck, ValidationMatrix, run_builtin_baseline_validation
from .provenance import (
    DatasetProvenance,
    EvidenceStatus,
    ProvenanceIssue,
    ProvenanceReport,
    ReferenceDataNotReleaseReady,
    assert_hardware_release_ready,
    validate_bundled_data,
)


def validation_summary() -> dict[str, Any]:
    return {
        "data_provenance": validate_bundled_data().to_dict(),
        "baseline_matrix": run_builtin_baseline_validation().to_dict(),
    }


__all__ = [
    "DatasetProvenance",
    "EvidenceStatus",
    "ProvenanceIssue",
    "ProvenanceReport",
    "ReferenceDataNotReleaseReady",
    "ValidationCheck",
    "ValidationMatrix",
    "assert_hardware_release_ready",
    "run_builtin_baseline_validation",
    "validate_bundled_data",
    "validation_summary",
]
