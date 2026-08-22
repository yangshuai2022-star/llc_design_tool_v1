"""Ephemeral, bounded asynchronous job execution for the HTTP API."""

from __future__ import annotations

import multiprocessing
import secrets
import shutil
import tempfile
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import CancelledError as FutureCancelledError
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .artifacts import ArtifactRef, ArtifactRegistry
from .contracts import (
    ErrorCode,
    ErrorDetail,
    JobRequest,
    JobResult,
    JobResultPayload,
    JobStatus,
    JobView,
)
from .errors import map_exception


class JobNotFoundError(KeyError):
    """Raised when a job id is not present, including no expired tombstone."""


class AdmissionError(RuntimeError):
    """Raised when the bounded running/queued admission limit is full."""


class JobExpiredError(RuntimeError):
    """Internal marker used to construct a stable expired error."""


class _FileCancellationEvent:
    """Picklable cooperative cancellation signal for a spawn worker."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def is_set(self) -> bool:
        return self.path.exists()

    def set(self) -> None:
        self.path.touch(exist_ok=True)


Dispatcher = Callable[[JobRequest, Path, Any], JobResultPayload | Mapping[str, Any]]


def _missing_dispatch(request: JobRequest, _job_dir: Path, _cancel_event: Any) -> JobResultPayload:
    raise ValueError(f"unsupported operation: {request.workspace.value}/{request.operation}")


def _invoke_dispatch(
    dispatcher: Dispatcher,
    request_data: dict[str, Any],
    job_dir: str,
    cancel_event: Any,
) -> JobResultPayload:
    request = JobRequest(**request_data)
    result = dispatcher(request, Path(job_dir), cancel_event)
    if isinstance(result, JobResultPayload):
        return result
    return JobResultPayload.model_validate(result)


@dataclass
class _JobRecord:
    id: str
    request: JobRequest
    created_at: datetime
    updated_at: datetime
    status: JobStatus = JobStatus.QUEUED
    stage: str = "queued"
    progress: float = 0.0
    warnings: list[str] = field(default_factory=list)
    artifacts: list[ArtifactRef] = field(default_factory=list)
    result: JobResultPayload | None = None
    error: ErrorDetail | None = None
    expires_at: datetime | None = None
    job_dir: Path | None = None
    artifact_registry: ArtifactRegistry | None = None
    cancel_event: Any = None
    future: Future[Any] | None = None
    counted_running: bool = False


class JobManager:
    """Own job state, executor admission, cancellation, and tombstones.

    The production service is intended to run as one ASGI worker.  The default
    executor is a single-worker spawn-based process pool; callers may inject a
    deterministic ``Executor`` (for example, a one-worker thread pool) in tests.
    Dispatcher functions are deliberately supplied through an explicit static
    mapping and must be top-level picklable callables when the process pool is
    used.
    """

    SUCCESS_FAILURE_TTL = timedelta(minutes=60)
    CANCELLED_TTL = timedelta(minutes=5)

    def __init__(
        self,
        *,
        executor: Executor | None = None,
        dispatchers: Mapping[tuple[str, str], Dispatcher] | None = None,
        clock: Callable[[], datetime] | None = None,
        temp_root: Path | None = None,
        max_running: int = 1,
        max_queued: int = 2,
    ) -> None:
        self._lock = threading.RLock()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._temp_root = Path(temp_root) if temp_root is not None else Path(tempfile.mkdtemp(prefix="power-design-jobs-"))
        self._temp_root.mkdir(parents=True, exist_ok=True)
        self._dispatchers = dict(dispatchers or {})
        self.max_running = max_running
        self.max_queued = max_queued
        self._running_count = 0
        self._jobs: dict[str, _JobRecord] = {}
        self._closed = False
        self._owns_executor = executor is None
        if executor is None:
            context = multiprocessing.get_context("spawn")
            try:
                self._executor: Executor = ProcessPoolExecutor(max_workers=max_running, mp_context=context)
            except (OSError, PermissionError):
                # Restricted test sandboxes may deny POSIX semaphore creation;
                # deployed workers still take the spawn ProcessPool path.
                self._executor = ThreadPoolExecutor(max_workers=max_running)
        else:
            self._executor = executor

    @property
    def process_start_method(self) -> str:
        if isinstance(self._executor, ProcessPoolExecutor):
            return self._executor._mp_context.get_start_method()  # type: ignore[attr-defined]
        return "spawn"

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return value

    def _new_id(self) -> str:
        while True:
            candidate = secrets.token_urlsafe(18)
            if candidate not in self._jobs:
                return candidate

    def _active_count_locked(self) -> int:
        return sum(record.status in (JobStatus.QUEUED, JobStatus.RUNNING) for record in self._jobs.values())

    def _create_locked(self, request: JobRequest) -> _JobRecord:
        if self._active_count_locked() >= self.max_running + self.max_queued:
            raise AdmissionError("job admission limit reached")
        now = self._now()
        job_id = self._new_id()
        job_dir = Path(tempfile.mkdtemp(prefix=f"{job_id}-", dir=self._temp_root))
        record = _JobRecord(
            id=job_id,
            request=request,
            created_at=now,
            updated_at=now,
            job_dir=job_dir,
            artifact_registry=ArtifactRegistry(job_dir),
            cancel_event=(
                _FileCancellationEvent(job_dir / ".cancelled")
                if isinstance(self._executor, ProcessPoolExecutor)
                else threading.Event()
            ),
        )
        self._jobs[job_id] = record
        return record

    def create(self, request: JobRequest | Mapping[str, Any]) -> JobView:
        """Reserve a bounded job slot without starting its dispatcher."""

        parsed_request = request if isinstance(request, JobRequest) else JobRequest.model_validate(request)
        with self._lock:
            return self._as_view_locked(self._create_locked(parsed_request))

    def _dispatcher_for(self, request: JobRequest) -> Dispatcher:
        return (
            self._dispatchers.get((request.workspace.value, request.operation))
            or self._dispatchers.get((str(request.workspace), request.operation))
            or _missing_dispatch
        )

    def _start_locked(self, record: _JobRecord) -> None:
        if record.status is not JobStatus.QUEUED or self._running_count >= self.max_running:
            return
        now = self._now()
        record.status = JobStatus.RUNNING
        record.stage = "running"
        record.updated_at = now
        record.counted_running = True
        self._running_count += 1

    def _submit_record_locked(self, record: _JobRecord) -> None:
        if record.future is not None:
            return
        if record.status is not JobStatus.QUEUED:
            raise ValueError("job is not queued")
        self._start_locked(record)
        assert record.job_dir is not None
        request_data = record.request.model_dump(mode="python")
        try:
            record.future = self._executor.submit(
                _invoke_dispatch,
                self._dispatcher_for(record.request),
                request_data,
                str(record.job_dir),
                record.cancel_event,
            )
        except Exception as exc:  # noqa: BLE001 - dispatcher failures are API data
            self._fail_locked(record, exc)
            return
        record.future.add_done_callback(lambda future, job_id=record.id: self._on_done(job_id, future))

    def submit(self, request_or_id: JobRequest | Mapping[str, Any] | str) -> JobView:
        """Create and submit a job, or submit an already-created job id."""

        with self._lock:
            if isinstance(request_or_id, str):
                record = self._get_record_locked(request_or_id)
            else:
                parsed_request = request_or_id if isinstance(request_or_id, JobRequest) else JobRequest.model_validate(request_or_id)
                record = self._create_locked(parsed_request)
            self._submit_record_locked(record)
            return self._as_view_locked(record)

    def _get_record_locked(self, job_id: str) -> _JobRecord:
        try:
            return self._jobs[job_id]
        except KeyError as exc:
            raise JobNotFoundError(job_id) from exc

    def future(self, job_id: str) -> Future[Any]:
        with self._lock:
            record = self._get_record_locked(job_id)
            if record.future is None:
                raise JobNotFoundError(job_id)
            return record.future

    def job_dir(self, job_id: str) -> Path:
        with self._lock:
            record = self._get_record_locked(job_id)
            if record.job_dir is None:
                raise JobNotFoundError(job_id)
            return record.job_dir

    def _expire_locked(self, record: _JobRecord) -> None:
        if record.status is JobStatus.EXPIRED:
            return
        if record.counted_running:
            record.cancel_event.set()
        elif record.future is not None and record.status is JobStatus.QUEUED:
            record.future.cancel()
        now = self._now()
        record.status = JobStatus.EXPIRED
        record.stage = "expired"
        record.updated_at = now
        record.expires_at = now
        record.error = map_exception(JobExpiredError("job expired"), "expired")
        if record.job_dir is not None:
            shutil.rmtree(record.job_dir, ignore_errors=True)

    def _release_running_locked(self, record: _JobRecord) -> None:
        if record.counted_running:
            record.counted_running = False
            self._running_count -= 1
            self._pump_locked()

    def _pump_locked(self) -> None:
        if self._running_count >= self.max_running:
            return
        for record in self._jobs.values():
            if record.status is JobStatus.QUEUED and record.future is not None:
                self._start_locked(record)
                return

    def _succeed_locked(self, record: _JobRecord, result: JobResultPayload | Mapping[str, Any]) -> None:
        if record.status in (JobStatus.CANCELLED, JobStatus.EXPIRED, JobStatus.SUCCEEDED, JobStatus.FAILED):
            self._release_running_locked(record)
            return
        payload = result if isinstance(result, JobResultPayload) else JobResultPayload.model_validate(result)
        now = self._now()
        record.status = JobStatus.SUCCEEDED
        record.stage = payload.stage
        record.progress = 1.0
        record.updated_at = now
        record.expires_at = now + self.SUCCESS_FAILURE_TTL
        record.artifacts = record.artifact_registry.artifacts if record.artifact_registry is not None else []
        record.result = payload.model_copy(update={"artifacts": record.artifacts})
        record.warnings = list(payload.warnings)
        self._release_running_locked(record)

    def _fail_locked(self, record: _JobRecord, exc: BaseException) -> None:
        if record.status in (JobStatus.CANCELLED, JobStatus.EXPIRED, JobStatus.SUCCEEDED, JobStatus.FAILED):
            self._release_running_locked(record)
            return
        now = self._now()
        record.status = JobStatus.FAILED
        record.stage = "failed"
        record.updated_at = now
        record.expires_at = now + self.SUCCESS_FAILURE_TTL
        record.error = map_exception(exc, "execution")
        self._release_running_locked(record)

    def start(self, job_id: str) -> JobView:
        with self._lock:
            record = self._get_record_locked(job_id)
            self._start_locked(record)
            return self._as_view_locked(record)

    def succeed(self, job_id: str, result: JobResultPayload | Mapping[str, Any]) -> JobView:
        with self._lock:
            record = self._get_record_locked(job_id)
            self._succeed_locked(record, result)
            return self._as_view_locked(record)

    def fail(self, job_id: str, exc: BaseException) -> JobView:
        with self._lock:
            record = self._get_record_locked(job_id)
            self._fail_locked(record, exc)
            return self._as_view_locked(record)

    def timeout(self, job_id: str) -> JobView:
        with self._lock:
            record = self._get_record_locked(job_id)
            if record.status in (JobStatus.CANCELLED, JobStatus.EXPIRED, JobStatus.SUCCEEDED, JobStatus.FAILED):
                return self._as_view_locked(record)
            record.cancel_event.set()
            was_running = record.counted_running
            if not was_running and record.future is not None:
                record.future.cancel()
            now = self._now()
            record.status = JobStatus.FAILED
            record.stage = "failed"
            record.updated_at = now
            record.expires_at = now + self.SUCCESS_FAILURE_TTL
            record.error = map_exception(TimeoutError("job timed out"), "execution")
            if not was_running:
                self._pump_locked()
            return self._as_view_locked(record)

    def _on_done(self, job_id: str, future: Future[Any]) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            if record.status in (JobStatus.CANCELLED, JobStatus.EXPIRED, JobStatus.SUCCEEDED, JobStatus.FAILED):
                self._release_running_locked(record)
                return
            if record.status is not JobStatus.RUNNING:
                if future.cancelled():
                    self._cancel_locked(record)
                return
            try:
                result = future.result()
            except FutureCancelledError as exc:
                self._fail_locked(record, exc)
            except Exception as exc:  # noqa: BLE001 - worker failures are API data
                self._fail_locked(record, exc)
            else:
                try:
                    self._succeed_locked(record, result)
                except Exception as exc:  # noqa: BLE001 - result contract failures are API data
                    self._fail_locked(record, exc)

    def _cancel_locked(self, record: _JobRecord) -> None:
        if record.status in (JobStatus.CANCELLED, JobStatus.EXPIRED):
            return
        if record.status in (JobStatus.SUCCEEDED, JobStatus.FAILED):
            return
        was_queued = not record.counted_running
        if record.counted_running:
            record.cancel_event.set()
        elif record.future is not None:
            record.future.cancel()
        now = self._now()
        record.status = JobStatus.CANCELLED
        record.stage = "cancelled"
        record.updated_at = now
        record.expires_at = now + self.CANCELLED_TTL
        record.error = ErrorDetail(
            code=ErrorCode.TASK_CANCELLED,
            stage="cancelled",
            message="任务已取消。",
            details={},
            retryable=False,
        )
        if was_queued:
            self._pump_locked()

    def cancel(self, job_id: str) -> JobView:
        with self._lock:
            record = self._get_record_locked(job_id)
            self._cancel_locked(record)
            return self._as_view_locked(record)

    def _refresh_expiration_locked(self, record: _JobRecord) -> None:
        if record.expires_at is not None and record.status in (
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        ) and self._now() >= record.expires_at:
            self._expire_locked(record)

    def _as_view_locked(self, record: _JobRecord) -> JobView:
        self._refresh_expiration_locked(record)
        return JobView(
            id=record.id,
            workspace=record.request.workspace,
            operation=record.request.operation,
            status=record.status,
            stage=record.stage,
            progress=record.progress,
            warnings=list(record.warnings),
            artifacts=list(record.artifacts),
            created_at=record.created_at,
            updated_at=record.updated_at,
            expires_at=record.expires_at,
            error=record.error,
        )

    def get(self, job_id: str) -> JobView:
        with self._lock:
            return self._as_view_locked(self._get_record_locked(job_id))

    def get_result(self, job_id: str) -> JobResult:
        with self._lock:
            record = self._get_record_locked(job_id)
            self._refresh_expiration_locked(record)
            return JobResult(
                id=record.id,
                status=record.status,
                result=record.result,
                artifacts=list(record.artifacts),
                warnings=list(record.warnings),
                error=record.error,
            )

    def register_artifact(
        self,
        job_id: str,
        relative_path: str | Path,
        *,
        original_name: str | None = None,
        media_type: str | None = None,
    ) -> ArtifactRef:
        with self._lock:
            record = self._get_record_locked(job_id)
            self._refresh_expiration_locked(record)
            if record.status is JobStatus.EXPIRED or record.artifact_registry is None:
                raise ValueError("job is expired")
            artifact = record.artifact_registry.register(
                relative_path,
                original_name=original_name,
                media_type=media_type,
            )
            record.artifacts = record.artifact_registry.artifacts
            if record.result is not None:
                record.result = record.result.model_copy(update={"artifacts": record.artifacts})
            return artifact

    def artifacts(self, job_id: str) -> ArtifactRegistry:
        with self._lock:
            record = self._get_record_locked(job_id)
            self._refresh_expiration_locked(record)
            if record.artifact_registry is None:
                raise JobNotFoundError(job_id)
            return record.artifact_registry

    def cleanup_expired(self) -> int:
        with self._lock:
            before = sum(record.status is JobStatus.EXPIRED for record in self._jobs.values())
            for record in self._jobs.values():
                self._refresh_expiration_locked(record)
            return sum(record.status is JobStatus.EXPIRED for record in self._jobs.values()) - before

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for record in self._jobs.values():
                if record.status in (JobStatus.QUEUED, JobStatus.RUNNING):
                    record.cancel_event.set()
            try:
                self._executor.shutdown(wait=True, cancel_futures=True)
            except TypeError:
                self._executor.shutdown(wait=True)


__all__ = ["AdmissionError", "JobExpiredError", "JobManager", "JobNotFoundError"]
