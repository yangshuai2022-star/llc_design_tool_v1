"""Public request, job, and artifact contracts for the web API.

The models in this module are intentionally small.  They are the boundary
between the HTTP service and the existing engineering algorithms; algorithm
objects should not be exposed directly through FastAPI responses.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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


class JobRequest(_ApiModel):
    """Input accepted when submitting an engineering job."""

    workspace: Workspace
    operation: str
    config: dict[str, Any] = Field(default_factory=dict)
    export_options: dict[str, Any] = Field(default_factory=dict)


class ArtifactRef(_ApiModel):
    """Metadata for a generated file without embedding file contents."""

    id: str
    name: str
    media_type: str
    size_bytes: int = Field(ge=0)
    sha256: str


class ErrorDetail(_ApiModel):
    """Stable, user-facing description of a failed job."""

    code: str
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


class JobResult(_ApiModel):
    """Completed job payload and any generated artifact references."""

    id: str
    status: JobStatus
    result: Any = None
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: ErrorDetail | None = None


__all__ = [
    "ArtifactRef",
    "ErrorDetail",
    "JobRequest",
    "JobResult",
    "JobStatus",
    "JobView",
    "Workspace",
]
