from __future__ import annotations

from llc_design.validation.project import validate_project_document


def _valid_project() -> dict:
    return {
        "schema_version": "1.0",
        "toolkit_version": "9.2.2",
        "workspace": "llc",
        "input": {"vbus_nom_v": 400.0},
        "results": {"lr_h": 10e-6},
        "sources": [
            {"id": "input", "kind": "repository", "locator": "example.json"}
        ],
        "evidence": [
            {
                "status": "VERIFIED",
                "scope": "software regression",
                "verified_for_hardware": False,
            },
            {
                "status": "UNKNOWN",
                "scope": "bench measurement",
                "verified_for_hardware": False,
            },
        ],
        "model_boundaries": ["software regression is not hardware validation"],
    }


def test_valid_project_provenance_envelope() -> None:
    report = validate_project_document(_valid_project())
    assert report.valid
    assert report.to_dict()["issues"] == []


def test_unknown_evidence_cannot_be_hardware_verified() -> None:
    project = _valid_project()
    project["evidence"][1]["verified_for_hardware"] = True
    report = validate_project_document(project)
    assert not report.valid
    assert any("hardware verification" in issue.message for issue in report.issues)


def test_missing_model_boundaries_is_rejected() -> None:
    project = _valid_project()
    del project["model_boundaries"]
    report = validate_project_document(project)
    assert not report.valid
    assert any(issue.field == "model_boundaries" for issue in report.issues)
