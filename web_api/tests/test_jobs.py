from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event

import pytest

from web_api.contracts import JobRequest, JobResultPayload, JobStatus
from web_api.jobs import AdmissionError, JobManager


class FakeClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 1, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, **kwargs: int) -> None:
        self.value += timedelta(**kwargs)


def _payload(request: JobRequest, _job_dir: Path, _cancel_event: object) -> JobResultPayload:
    return JobResultPayload(
        toolkit_version="7.5.0",
        algorithm_version="test-v1",
        workspace=request.workspace,
        operation=request.operation,
        config_snapshot=request.config,
        stage="complete",
        elapsed_s=0.0,
        metrics={"ok": True},
        feasibility=True,
    )


def _blocking_payload(request: JobRequest, _job_dir: Path, release: Event) -> JobResultPayload:
    release.wait(timeout=2.0)
    return _payload(request, _job_dir, release)


def _failing_dispatch(_request: JobRequest, _job_dir: Path, _cancel_event: object) -> JobResultPayload:
    raise ValueError("invalid parameter at /private/design/input.json")


def _failing_unc_dispatch(_request: JobRequest, _job_dir: Path, _cancel_event: object) -> JobResultPayload:
    raise ValueError(r"invalid parameter at \\server\share\file.json")


def _request() -> JobRequest:
    return JobRequest(workspace="llc", operation="system", config={"vin_nom_v": 400.0})


def test_manager_uses_spawn_and_completes_with_terminal_ttl(tmp_path: Path) -> None:
    clock = FakeClock()
    executor = ThreadPoolExecutor(max_workers=1)
    manager = JobManager(
        executor=executor,
        clock=clock,
        temp_root=tmp_path,
        dispatchers={("llc", "system"): _payload},
    )
    assert manager.process_start_method == "spawn"

    view = manager.submit(_request())
    manager.future(view.id).result(timeout=2.0)
    assert manager.get(view.id).status is JobStatus.SUCCEEDED
    assert manager.get_result(view.id).result is not None
    clock.advance(minutes=59, seconds=59)
    assert manager.get(view.id).status is JobStatus.SUCCEEDED
    clock.advance(seconds=1)
    assert manager.get(view.id).status is JobStatus.EXPIRED
    assert manager.get_result(view.id).status is JobStatus.EXPIRED
    manager.shutdown()


def test_manager_admits_one_running_and_two_queued_and_cancels_queued(tmp_path: Path) -> None:
    clock = FakeClock()
    release = Event()
    executor = ThreadPoolExecutor(max_workers=1)
    manager = JobManager(
        executor=executor,
        clock=clock,
        temp_root=tmp_path,
        dispatchers={("llc", "system"): lambda request, path, event: _blocking_payload(request, path, release)},
    )

    first = manager.submit(_request())
    second = manager.submit(_request())
    third = manager.submit(_request())
    with pytest.raises(AdmissionError):
        manager.submit(_request())
    assert manager.get(first.id).status is JobStatus.RUNNING
    assert manager.get(second.id).status is JobStatus.QUEUED
    assert manager.get(third.id).status is JobStatus.QUEUED

    cancelled = manager.cancel(second.id)
    assert cancelled.status is JobStatus.CANCELLED
    assert manager.get(second.id).status is JobStatus.CANCELLED
    release.set()
    manager.future(first.id).result(timeout=2.0)
    manager.future(third.id).result(timeout=2.0)
    assert manager.get(first.id).status is JobStatus.SUCCEEDED
    assert manager.get(third.id).status is JobStatus.SUCCEEDED
    manager.shutdown()


def test_late_completion_cannot_overwrite_cancelled_job(tmp_path: Path) -> None:
    release = Event()
    executor = ThreadPoolExecutor(max_workers=1)
    manager = JobManager(
        executor=executor,
        temp_root=tmp_path,
        dispatchers={("llc", "system"): lambda request, path, event: _blocking_payload(request, path, release)},
    )
    view = manager.submit(_request())
    assert manager.cancel(view.id).status is JobStatus.CANCELLED
    release.set()
    manager.future(view.id).result(timeout=2.0)
    assert manager.get(view.id).status is JobStatus.CANCELLED
    assert manager.get_result(view.id).error is not None
    manager.shutdown()


def test_late_completion_cannot_overwrite_timed_out_job(tmp_path: Path) -> None:
    release = Event()
    executor = ThreadPoolExecutor(max_workers=1)
    manager = JobManager(
        executor=executor,
        temp_root=tmp_path,
        dispatchers={("llc", "system"): lambda request, path, event: _blocking_payload(request, path, release)},
    )
    view = manager.submit(_request())
    assert manager.timeout(view.id).error is not None
    assert manager.get_result(view.id).error.code.value == "task_timeout"
    release.set()
    manager.future(view.id).result(timeout=2.0)
    assert manager.get(view.id).status is JobStatus.FAILED
    manager.shutdown()


def test_failed_job_has_stable_error_and_failure_ttl(tmp_path: Path) -> None:
    clock = FakeClock()
    executor = ThreadPoolExecutor(max_workers=1)
    manager = JobManager(
        executor=executor,
        clock=clock,
        temp_root=tmp_path,
        dispatchers={("llc", "system"): _failing_dispatch},
    )
    view = manager.submit(_request())
    with pytest.raises(ValueError):
        manager.future(view.id).result(timeout=2.0)
    result = manager.get_result(view.id)
    assert manager.get(view.id).status is JobStatus.FAILED
    assert result.error is not None
    assert result.error.code.value == "invalid_parameters"
    assert "/private" not in result.error.details["message"]
    clock.advance(minutes=60)
    manager.cleanup_expired()
    assert manager.get(view.id).status is JobStatus.EXPIRED
    manager.shutdown()


def test_failed_job_redacts_unc_paths(tmp_path: Path) -> None:
    executor = ThreadPoolExecutor(max_workers=1)
    manager = JobManager(
        executor=executor,
        temp_root=tmp_path,
        dispatchers={("llc", "system"): _failing_unc_dispatch},
    )
    view = manager.submit(_request())
    with pytest.raises(ValueError):
        manager.future(view.id).result(timeout=2.0)
    result = manager.get_result(view.id)
    assert result.error is not None
    assert "server" not in result.error.details["message"]
    assert "<path>" in result.error.details["message"]
    manager.shutdown()


def test_cancelled_job_uses_five_minute_tombstone_ttl(tmp_path: Path) -> None:
    clock = FakeClock()
    release = Event()

    def blocking(request: JobRequest, path: Path, _cancel_event: object) -> JobResultPayload:
        release.wait(timeout=2.0)
        return _payload(request, path, release)

    executor = ThreadPoolExecutor(max_workers=1)
    manager = JobManager(
        executor=executor,
        clock=clock,
        temp_root=tmp_path,
        dispatchers={("llc", "system"): blocking},
    )
    view = manager.submit(_request())
    assert manager.cancel(view.id).status is JobStatus.CANCELLED
    release.set()
    manager.future(view.id).result(timeout=2.0)
    clock.advance(minutes=4, seconds=59)
    assert manager.get(view.id).status is JobStatus.CANCELLED
    clock.advance(seconds=1)
    manager.cleanup_expired()
    assert manager.get(view.id).status is JobStatus.EXPIRED
    clock.advance(seconds=1)
    assert manager.get(view.id).status is JobStatus.EXPIRED
    manager.shutdown()
