"""Public request, job, and artifact contracts for the web API.

The models in this module are intentionally small.  They are the boundary
between the HTTP service and the existing engineering algorithms; algorithm
objects should not be exposed directly through FastAPI responses.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import TypeAliasType


class _ApiModel(BaseModel):
    """Common API configuration shared by every public model."""

    model_config = ConfigDict(extra="forbid")


class Workspace(str, Enum):
    """Supported engineering workspaces."""

    LLC = "llc"
    TTPL = "ttpl"
    VIENNA = "vienna"
    PFC_TOOLBOX = "pfc_toolbox"


class JobStatus(str, Enum):
    """Stable lifecycle states returned for an asynchronous job."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class ErrorCode(str, Enum):
    """Machine-readable failure taxonomy exposed by the API."""

    INVALID_PARAMETERS = "invalid_parameters"
    GAIN_NOT_REACHABLE = "gain_not_reachable"
    CORE_NO_SOLUTION = "core_no_solution"
    DEVICE_NOT_FOUND = "device_not_found"
    RESOURCE_NOT_FOUND = "resource_not_found"
    INDUCTOR_NO_SOLUTION = "inductor_no_solution"
    MAGNETICS_NO_SOLUTION = "magnetics_no_solution"
    SIMULATION_NON_CONVERGENCE = "simulation_non_convergence"
    UNSTABLE_LOOP = "unstable_loop"
    STABILITY_GATE_REJECTED = "stability_gate_rejected"
    TASK_TIMEOUT = "task_timeout"
    TASK_CANCELLED = "task_cancelled"
    TASK_EXPIRED = "task_expired"
    INTERNAL_ERROR = "internal_error"


JSONValue = TypeAliasType(
    "JSONValue",
    dict[str, "JSONValue"] | list["JSONValue"] | str | int | float | bool | None,
)
JSONObject = dict[str, JSONValue]


class JobRequest(_ApiModel):
    """Input accepted when submitting an engineering job."""

    workspace: Workspace
    operation: str
    config: dict[str, Any] = Field(default_factory=dict)
    export_options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("operation")
    @classmethod
    def operation_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("operation must not be blank")
        return value


class ArtifactRef(_ApiModel):
    """Metadata for a generated file without embedding file contents."""

    id: str
    name: str
    media_type: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ErrorDetail(_ApiModel):
    """Stable, user-facing description of a failed job."""

    code: ErrorCode
    stage: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    retryable: bool = False


class JobView(_ApiModel):
    """Status and metadata returned while a job is being processed."""

    id: str
    workspace: Workspace
    operation: str
    status: JobStatus
    stage: str
    progress: float = Field(ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None = None
    error: ErrorDetail | None = None

    @field_validator("created_at", "updated_at", "expires_at")
    @classmethod
    def timestamps_must_be_timezone_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def timestamps_must_be_ordered(self) -> JobView:
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not precede created_at")
        if self.expires_at is not None and self.expires_at < self.updated_at:
            raise ValueError("expires_at must not precede updated_at")
        return self


class JobResultPayload(_ApiModel):
    """Complete, recursively JSON-safe engineering result payload."""

    toolkit_version: str
    algorithm_version: str
    workspace: Workspace
    operation: str
    config_snapshot: JSONObject = Field(default_factory=dict)
    parameters: JSONObject = Field(default_factory=dict)
    stage: str
    elapsed_s: float = Field(ge=0.0)
    metrics: JSONObject = Field(default_factory=dict)
    tables: JSONValue = Field(default_factory=dict)
    series: JSONValue = Field(default_factory=dict)
    plots: JSONValue = Field(default_factory=dict)
    units: dict[str, str] = Field(default_factory=dict)
    feasibility: bool | None = None
    warnings: list[str] = Field(default_factory=list)
    evidence: JSONValue = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)

    @field_validator("operation")
    @classmethod
    def operation_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("operation must not be blank")
        return value


class JobResult(_ApiModel):
    """Completed job payload and any generated artifact references."""

    id: str
    status: JobStatus
    result: JobResultPayload | None = None
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: ErrorDetail | None = None

    @model_validator(mode="after")
    def succeeded_result_must_have_payload(self) -> JobResult:
        if self.status is JobStatus.SUCCEEDED and self.result is None:
            raise ValueError("succeeded jobs must include a result payload")
        return self


__all__ = [
    "ArtifactRef",
    "ErrorCode",
    "ErrorDetail",
    "JSONObject",
    "JSONValue",
    "JobRequest",
    "JobResult",
    "JobResultPayload",
    "JobStatus",
    "JobView",
    "Workspace",
]
