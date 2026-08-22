from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from web_api.contracts import (
    ArtifactRef,
    ErrorCode,
    ErrorDetail,
    JobRequest,
    JobResult,
    JobResultPayload,
    JobStatus,
    JobView,
    Workspace,
)


def test_job_request_accepts_only_public_workspaces():
    request = JobRequest(
        workspace="llc",
        operation="system",
        config={"vin_nom_v": 400.0},
    )

    assert request.workspace is Workspace.LLC
    assert request.export_options == {}

    with pytest.raises(ValidationError):
        JobRequest(workspace="python", operation="eval", config={})


def test_job_contracts_reject_unknown_fields_and_invalid_progress():
    now = datetime.now(UTC)
    artifact = ArtifactRef(
        id="report",
        name="design-report.pdf",
        media_type="application/pdf",
        size_bytes=128,
        sha256="a" * 64,
    )
    view = JobView(
        id="job-1",
        workspace=Workspace.LLC,
        operation="system",
        status=JobStatus.SUCCEEDED,
        stage="complete",
        progress=1.0,
        warnings=[],
        artifacts=[artifact],
        created_at=now,
        updated_at=now,
        expires_at=now,
    )

    assert view.artifacts[0].name == "design-report.pdf"

    invalid_view = view.model_dump()
    invalid_view["progress"] = 1.1
    invalid_view["unexpected"] = "not allowed"
    with pytest.raises(ValidationError):
        JobView(**invalid_view)


def test_error_detail_has_stable_machine_and_user_fields():
    detail = ErrorDetail(
        code="invalid_parameters",
        stage="validation",
        message="参数不符合约束。",
        details={"exception_type": "ValueError"},
        retryable=False,
    )

    assert detail.model_dump()["code"] == "invalid_parameters"
    assert detail.code is ErrorCode.INVALID_PARAMETERS


def test_request_operation_and_artifact_digest_are_strict():
    with pytest.raises(ValidationError):
        JobRequest(workspace="llc", operation="   ")

    with pytest.raises(ValidationError):
        ArtifactRef(
            id="report",
            name="report.pdf",
            media_type="application/pdf",
            size_bytes=0,
            sha256="A" * 64,
        )


def test_job_view_rejects_blank_operation_and_normalizes_valid_operation():
    created = datetime(2026, 1, 1, tzinfo=UTC)
    view = JobView(
        id="job-1",
        workspace="llc",
        operation="  system  ",
        status="queued",
        stage="queued",
        progress=0.0,
        created_at=created,
        updated_at=created,
    )
    assert view.operation == "system"

    with pytest.raises(ValidationError):
        JobView(
            id="job-1",
            workspace="llc",
            operation="   ",
            status="queued",
            stage="queued",
            progress=0.0,
            created_at=created,
            updated_at=created,
        )


def test_job_timestamps_require_timezone_and_monotonic_order():
    created = datetime(2026, 1, 1, tzinfo=UTC)
    with pytest.raises(ValidationError):
        JobView(
            id="job-1",
            workspace="llc",
            operation="system",
            status="queued",
            stage="queued",
            progress=0.0,
            created_at=created.replace(tzinfo=None),
            updated_at=created,
        )

    with pytest.raises(ValidationError):
        JobView(
            id="job-1",
            workspace="llc",
            operation="system",
            status="queued",
            stage="queued",
            progress=0.0,
            created_at=created,
            updated_at=created - timedelta(seconds=1),
        )


def test_succeeded_job_result_requires_complete_typed_payload():
    payload = JobResultPayload(
        toolkit_version="7.5.0",
        algorithm_version="llc-fha-v1",
        workspace="llc",
        operation="system",
        config_snapshot={"vin_nom_v": 400.0},
        parameters={"pout_w": 1000.0},
        stage="analysis",
        elapsed_s=0.25,
        metrics={"efficiency": 0.95},
        tables={"summary": [{"name": "Lr", "value": 1e-6}]},
        series={"frequency_hz": [100000.0, 110000.0]},
        plots={"gain": "gain.png"},
        units={"frequency_hz": "Hz"},
        feasibility=True,
        warnings=[],
        evidence=["validated against FHA"],
    )
    result = JobResult(id="job-1", status="succeeded", result=payload)

    assert result.result.metrics["efficiency"] == 0.95
    assert result.result.workspace is Workspace.LLC

    with pytest.raises(ValidationError):
        JobResult(id="job-1", status="succeeded")

    with pytest.raises(ValidationError):
        JobResult(id="job-1", status="succeeded", result=payload, unexpected=True)
