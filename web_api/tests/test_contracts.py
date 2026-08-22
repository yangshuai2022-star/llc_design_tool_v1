from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from web_api.contracts import (
    ArtifactRef,
    ErrorDetail,
    JobRequest,
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
