"""Validation helpers for versioned engineering-project provenance envelopes.

This module validates traceability metadata only.  It deliberately does not
interpret topology-specific numerical results or promote software evidence to
hardware verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .provenance import EvidenceStatus


_REQUIRED_TOP_LEVEL = (
    "schema_version",
    "toolkit_version",
    "workspace",
    "input",
    "results",
    "evidence",
    "model_boundaries",
)


@dataclass(frozen=True)
class ProjectProvenanceIssue:
    field: str
    message: str


@dataclass(frozen=True)
class ProjectProvenanceReport:
    issues: tuple[ProjectProvenanceIssue, ...]

    @property
    def valid(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "issues": [
                {"field": issue.field, "message": issue.message}
                for issue in self.issues
            ],
        }


def validate_project_document(document: Any) -> ProjectProvenanceReport:
    """Validate the common provenance envelope used by exported projects."""

    issues: list[ProjectProvenanceIssue] = []
    if not isinstance(document, dict):
        return ProjectProvenanceReport((
            ProjectProvenanceIssue("$", "project document must be a JSON object"),
        ))

    for field in _REQUIRED_TOP_LEVEL:
        if field not in document:
            issues.append(ProjectProvenanceIssue(field, "required field is missing"))

    for field in ("schema_version", "toolkit_version", "workspace"):
        value = document.get(field)
        if not isinstance(value, str) or not value.strip():
            issues.append(ProjectProvenanceIssue(field, "must be a non-empty string"))

    for field in ("input", "results"):
        value = document.get(field)
        if not isinstance(value, dict):
            issues.append(ProjectProvenanceIssue(field, "must be an object"))

    boundaries = document.get("model_boundaries")
    if not isinstance(boundaries, list) or any(
        not isinstance(item, str) or not item.strip() for item in boundaries
    ):
        issues.append(ProjectProvenanceIssue(
            "model_boundaries", "must be a list of non-empty strings"
        ))

    evidence = document.get("evidence")
    if not isinstance(evidence, list):
        issues.append(ProjectProvenanceIssue("evidence", "must be a list"))
    else:
        for index, item in enumerate(evidence):
            path = f"evidence[{index}]"
            if not isinstance(item, dict):
                issues.append(ProjectProvenanceIssue(path, "must be an object"))
                continue
            status = item.get("status")
            try:
                EvidenceStatus(status)
            except (TypeError, ValueError):
                issues.append(ProjectProvenanceIssue(
                    f"{path}.status", "must be VERIFIED or UNKNOWN"
                ))
            scope = item.get("scope")
            if not isinstance(scope, str) or not scope.strip():
                issues.append(ProjectProvenanceIssue(
                    f"{path}.scope", "must be a non-empty string"
                ))
            verified = item.get("verified_for_hardware")
            if not isinstance(verified, bool):
                issues.append(ProjectProvenanceIssue(
                    f"{path}.verified_for_hardware", "must be boolean"
                ))
            elif verified and status != EvidenceStatus.VERIFIED.value:
                issues.append(ProjectProvenanceIssue(
                    f"{path}.verified_for_hardware",
                    "hardware verification requires status VERIFIED",
                ))

    sources = document.get("sources", [])
    if not isinstance(sources, list):
        issues.append(ProjectProvenanceIssue("sources", "must be a list when present"))
    else:
        for index, source in enumerate(sources):
            if not isinstance(source, dict):
                issues.append(ProjectProvenanceIssue(
                    f"sources[{index}]", "must be an object"
                ))
                continue
            for field in ("id", "kind", "locator"):
                value = source.get(field)
                if not isinstance(value, str) or not value.strip():
                    issues.append(ProjectProvenanceIssue(
                        f"sources[{index}].{field}", "must be a non-empty string"
                    ))

    return ProjectProvenanceReport(tuple(issues))
