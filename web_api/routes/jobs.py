"""Job and artifact endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response, status
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from ..artifacts import ArtifactError, ArtifactNotFoundError
from ..contracts import ArtifactRef, JobRequest, JobResult, JobStatus, JobView
from ..jobs import AdmissionError, JobManager, JobNotFoundError

router = APIRouter(prefix="/api/v1")


class ArtifactRegistrationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    name: str | None = None
    media_type: str | None = None


def _manager(request: Request) -> JobManager:
    return request.app.state.job_manager


def _not_found() -> HTTPException:
    return HTTPException(status_code=404, detail={"code": "resource_not_found", "message": "请求的资源不存在。"})


def _state_response(payload: Any, state: JobStatus) -> JSONResponse:
    code = 409 if state in (JobStatus.QUEUED, JobStatus.RUNNING) else 410
    return JSONResponse(status_code=code, content=payload.model_dump(mode="json"))


def _ensure_job_available(view: JobView) -> None:
    if view.status in (JobStatus.CANCELLED, JobStatus.EXPIRED):
        raise HTTPException(
            status_code=410,
            detail={"code": view.status.value, "message": "任务已取消或过期。"},
        )


@router.post("/jobs", response_model=JobView, status_code=status.HTTP_202_ACCEPTED)
def create_job(request: Request, payload: JobRequest, response: Response) -> JobView:
    try:
        view = _manager(request).submit(payload)
    except AdmissionError as exc:
        raise HTTPException(status_code=429, detail={"code": "admission_limit", "message": str(exc)}) from exc
    response.headers["Location"] = f"/api/v1/jobs/{view.id}"
    return view


@router.get("/jobs/{job_id}", response_model=JobView)
def get_job(job_id: str, request: Request) -> JobView | JSONResponse:
    try:
        view = _manager(request).get(job_id)
    except JobNotFoundError as exc:
        raise _not_found() from exc
    if view.status in (JobStatus.CANCELLED, JobStatus.EXPIRED):
        return _state_response(view, view.status)
    return view


@router.delete("/jobs/{job_id}", response_model=JobView)
def cancel_job(job_id: str, request: Request) -> JobView | JSONResponse:
    try:
        view = _manager(request).cancel(job_id)
    except JobNotFoundError as exc:
        raise _not_found() from exc
    return view


@router.get("/jobs/{job_id}/result", response_model=JobResult)
def get_result(job_id: str, request: Request) -> JobResult | JSONResponse:
    try:
        result = _manager(request).get_result(job_id)
    except JobNotFoundError as exc:
        raise _not_found() from exc
    if result.status in (JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.CANCELLED, JobStatus.EXPIRED):
        return _state_response(result, result.status)
    return result


@router.get("/jobs/{job_id}/artifacts", response_model=list[ArtifactRef])
def list_artifacts(job_id: str, request: Request) -> list[ArtifactRef] | JSONResponse:
    manager = _manager(request)
    try:
        view = manager.get(job_id)
        registry = manager.artifacts(job_id)
    except JobNotFoundError as exc:
        raise _not_found() from exc
    if view.status in (JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.CANCELLED, JobStatus.EXPIRED):
        return _state_response(view, view.status)
    return registry.artifacts


@router.post("/jobs/{job_id}/artifacts", response_model=ArtifactRef, status_code=status.HTTP_201_CREATED)
def register_artifact(job_id: str, payload: ArtifactRegistrationRequest, request: Request) -> ArtifactRef:
    manager = _manager(request)
    try:
        view = manager.get(job_id)
        _ensure_job_available(view)
        if view.status in (JobStatus.QUEUED, JobStatus.RUNNING):
            raise HTTPException(status_code=409, detail={"code": "not_ready", "message": "任务尚未完成。"})
        return manager.register_artifact(
            job_id,
            payload.path,
            original_name=payload.name,
            media_type=payload.media_type,
        )
    except JobNotFoundError as exc:
        raise _not_found() from exc
    except ArtifactError as exc:
        raise HTTPException(status_code=422, detail={"code": "invalid_artifact", "message": str(exc)}) from exc


@router.get("/jobs/{job_id}/artifacts/{artifact_id}")
def download_artifact(job_id: str, artifact_id: str, request: Request) -> FileResponse:
    manager = _manager(request)
    try:
        view = manager.get(job_id)
        _ensure_job_available(view)
        registry = manager.artifacts(job_id)
        item = registry.resolve_for_download(artifact_id)
        disposition = registry.content_disposition(artifact_id)
    except JobNotFoundError as exc:
        raise _not_found() from exc
    except ArtifactNotFoundError as exc:
        raise _not_found() from exc
    except ArtifactError as exc:
        raise HTTPException(status_code=410, detail={"code": "artifact_unavailable", "message": str(exc)}) from exc
    return FileResponse(
        item.path,
        media_type=item.ref.media_type,
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "Content-Disposition": disposition,
        },
    )


@router.get("/jobs/{job_id}/artifacts.zip")
def download_artifacts_zip(job_id: str, request: Request) -> StreamingResponse:
    manager = _manager(request)
    try:
        view = manager.get(job_id)
        _ensure_job_available(view)
        archive = manager.artifacts(job_id).build_zip()
    except JobNotFoundError as exc:
        raise _not_found() from exc
    except ArtifactError as exc:
        raise HTTPException(status_code=410, detail={"code": "artifact_unavailable", "message": str(exc)}) from exc
    return StreamingResponse(
        archive,
        media_type="application/zip",
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "Content-Disposition": 'attachment; filename="artifacts.zip"',
        },
    )


__all__ = ["router"]
