from __future__ import annotations

import pickle
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event

import pytest

from web_api.contracts import JobRequest, JobResultPayload, JobStatus
from web_api.jobs import (
    AdapterOutput,
    AdmissionError,
    InternalArtifact,
    JobManager,
    bridge_adapter,
)


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


def _sample_adapter(
    operation: str,
    config: dict[str, object],
    export_options: dict[str, object],
    job_dir: Path,
) -> AdapterOutput:
    (job_dir / "report.txt").write_text("report", encoding="utf-8")
    payload = JobResultPayload(
        toolkit_version="7.5.0",
        algorithm_version="adapter-v1",
        workspace="llc",
        operation=operation,
        config_snapshot=config,
        stage="complete",
        elapsed_s=0.0,
        metrics={"ok": export_options.get("format") == "json"},
        feasibility=True,
    )
    return AdapterOutput(
        result=payload,
        artifacts=(InternalArtifact(path="report.txt", original_name="result.txt"),),
    )


class _ImmediateQueuedExecutor:
    def __init__(self) -> None:
        self.futures: list[Future[object]] = []

    def submit(self, function: object, *args: object, **kwargs: object) -> Future[object]:
        future: Future[object] = Future()
        if not self.futures:
            self.futures.append(future)
            return future
        try:
            future.set_result(function(*args, **kwargs))  # type: ignore[operator]
        except BaseException as exc:  # noqa: BLE001 - fake executor captures worker termination
            future.set_exception(exc)
        self.futures.append(future)
        return future

    def shutdown(self, **_kwargs: object) -> None:
        return None


def _request(export_options: dict[str, object] | None = None) -> JobRequest:
    return JobRequest(
        workspace="llc",
        operation="system",
        config={"vin_nom_v": 400.0},
        export_options=export_options or {},
    )


def test_manager_uses_spawn_and_completes_with_terminal_ttl(tmp_path: Path) -> None:
    clock = FakeClock()
    executor = ThreadPoolExecutor(max_workers=1)
    manager = JobManager(
        executor=executor,
        clock=clock,
        temp_root=tmp_path,
        dispatchers={("llc", "system"): _payload},
    )
    assert pickle.dumps(_payload)

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


def test_completed_queued_future_is_consumed_without_sticking_or_double_completion(tmp_path: Path) -> None:
    executor = _ImmediateQueuedExecutor()
    manager = JobManager(
        executor=executor,
        temp_root=tmp_path,
        dispatchers={("llc", "system"): _payload},
    )
    first = manager.submit(_request())
    second = manager.submit(_request())
    assert manager.get(first.id).status is JobStatus.RUNNING
    assert manager.get(second.id).status is JobStatus.SUCCEEDED
    executor.futures[0].set_result(_payload(_request(), tmp_path, Event()))
    assert manager.get(first.id).status is JobStatus.SUCCEEDED
    third = manager.submit(_request())
    assert third.status is JobStatus.SUCCEEDED
    manager.shutdown()


def test_adapter_bridge_registers_internal_paths_and_populates_public_artifacts(tmp_path: Path) -> None:
    assert pickle.dumps(bridge_adapter(_sample_adapter))
    executor = ThreadPoolExecutor(max_workers=1)
    manager = JobManager(
        executor=executor,
        temp_root=tmp_path,
        dispatchers={("llc", "system"): bridge_adapter(_sample_adapter)},
    )
    view = manager.submit(_request({"format": "json"}))
    manager.future(view.id).result(timeout=2.0)
    result = manager.get_result(view.id)
    assert result.result is not None
    assert len(result.artifacts) == 1
    assert result.artifacts[0].name == "result.txt"
    assert result.result.artifacts[0].id == result.artifacts[0].id
    assert result.result.metrics["ok"] is True
    assert all(not isinstance(value, Path) for value in result.result.model_dump(mode="python").values())
    manager.shutdown()


def test_expired_tombstone_scrubs_result_artifacts_and_internal_directory(tmp_path: Path) -> None:
    clock = FakeClock()
    executor = ThreadPoolExecutor(max_workers=1)
    manager = JobManager(
        executor=executor,
        clock=clock,
        temp_root=tmp_path,
        dispatchers={("llc", "system"): bridge_adapter(_sample_adapter)},
    )
    view = manager.submit(_request())
    manager.future(view.id).result(timeout=2.0)
    job_dir = manager.job_dir(view.id)
    clock.advance(minutes=60)
    manager.cleanup_expired()
    result = manager.get_result(view.id)
    assert result.status is JobStatus.EXPIRED
    assert result.result is None
    assert result.artifacts == []
    assert not job_dir.exists()
    manager.shutdown()


def test_adapter_envelope_accepts_absolute_inside_path_and_rejects_outside_path(tmp_path: Path) -> None:
    def inside_adapter(
        operation: str,
        config: dict[str, object],
        export_options: dict[str, object],
        job_dir: Path,
    ) -> dict[str, object]:
        path = job_dir / "inside.txt"
        path.write_text(export_options["marker"], encoding="utf-8")
        payload = _payload(JobRequest(workspace="llc", operation=operation, config=config), job_dir, Event())
        return {
            "result": payload.model_dump(mode="python"),
            "artifact_manifest": [{"path": str(path), "original_name": "inside.txt"}],
        }

    manager = JobManager(
        executor=ThreadPoolExecutor(max_workers=1),
        temp_root=tmp_path,
        dispatchers={("llc", "system"): bridge_adapter(inside_adapter)},
    )
    view = manager.submit(_request({"marker": "inside"}))
    manager.future(view.id).result(timeout=2.0)
    assert manager.get_result(view.id).artifacts[0].name == "inside.txt"
    manager.shutdown()

    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")

    def outside_adapter(
        operation: str,
        config: dict[str, object],
        export_options: dict[str, object],
        job_dir: Path,
    ) -> AdapterOutput:
        return AdapterOutput(
            result=_payload(JobRequest(workspace="llc", operation=operation, config=config), job_dir, Event()),
            artifacts=(InternalArtifact(path=outside),),
        )

    manager = JobManager(
        executor=ThreadPoolExecutor(max_workers=1),
        temp_root=tmp_path / "outside-case",
        dispatchers={("llc", "system"): bridge_adapter(outside_adapter)},
    )
    view = manager.submit(_request())
    manager.future(view.id).result(timeout=2.0)
    result = manager.get_result(view.id)
    assert result.error is not None
    assert result.artifacts == []
    manager.shutdown()


def test_direct_payload_manifest_is_not_accepted_as_public_artifact(tmp_path: Path) -> None:
    def ambiguous(_request: JobRequest, _job_dir: Path, _cancel_event: object) -> dict[str, object]:
        payload = _payload(_request, _job_dir, _cancel_event)
        value = payload.model_dump(mode="python")
        value["artifacts"] = [
            {
                "id": "unregistered",
                "name": "report.txt",
                "media_type": "text/plain",
                "size_bytes": 1,
                "sha256": "a" * 64,
            }
        ]
        return value

    manager = JobManager(
        executor=ThreadPoolExecutor(max_workers=1),
        temp_root=tmp_path,
        dispatchers={("llc", "system"): ambiguous},
    )
    view = manager.submit(_request())
    manager.future(view.id).result(timeout=2.0)
    result = manager.get_result(view.id)
    assert result.artifacts == []
    assert result.result is None
    manager.shutdown()


def test_result_identity_mismatch_fails_without_result_or_artifacts(tmp_path: Path) -> None:
    def mismatch(request: JobRequest, job_dir: Path, _cancel_event: object) -> JobResultPayload:
        payload = _payload(request, job_dir, _cancel_event)
        return payload.model_copy(update={"workspace": "vienna", "operation": "other"})

    manager = JobManager(
        executor=ThreadPoolExecutor(max_workers=1),
        temp_root=tmp_path,
        dispatchers={("llc", "system"): mismatch},
    )
    view = manager.submit(_request())
    manager.future(view.id).result(timeout=2.0)
    result = manager.get_result(view.id)
    assert result.result is None
    assert result.artifacts == []
    assert result.error is not None
    assert result.error.code.value == "internal_error"
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


def test_cancel_scrubs_running_and_queued_artifacts_before_late_completion(tmp_path: Path) -> None:
    release = Event()
    executor = ThreadPoolExecutor(max_workers=1)
    manager = JobManager(
        executor=executor,
        temp_root=tmp_path,
        dispatchers={("llc", "system"): lambda request, path, event: _blocking_payload(request, path, release)},
    )
    first = manager.create(_request())
    first_dir = manager.job_dir(first.id)
    (first_dir / "first.txt").write_text("first", encoding="utf-8")
    manager.register_artifact(first.id, "first.txt")
    manager.submit(first.id)
    second = manager.create(_request())
    second_dir = manager.job_dir(second.id)
    (second_dir / "second.txt").write_text("second", encoding="utf-8")
    manager.register_artifact(second.id, "second.txt")
    manager.submit(second.id)

    assert manager.cancel(first.id).artifacts == []
    assert manager.get_result(first.id).result is None
    assert manager.get_result(first.id).artifacts == []
    with pytest.raises((KeyError, ValueError)):
        manager.artifacts(first.id)
    with pytest.raises(ValueError):
        manager.register_artifact(first.id, "first.txt")
    assert manager.cancel(second.id).artifacts == []
    assert manager.get_result(second.id).result is None
    release.set()
    manager.future(first.id).result(timeout=2.0)
    assert manager.get_result(first.id).result is None
    assert manager.get_result(first.id).artifacts == []
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
    job_dir = manager.job_dir(view.id)
    assert manager.cancel(view.id).status is JobStatus.CANCELLED
    release.set()
    manager.future(view.id).result(timeout=2.0)
    clock.advance(minutes=4, seconds=59)
    assert manager.get(view.id).status is JobStatus.CANCELLED
    clock.advance(seconds=1)
    manager.cleanup_expired()
    assert manager.get(view.id).status is JobStatus.EXPIRED
    assert not job_dir.exists()
    clock.advance(seconds=1)
    assert manager.get(view.id).status is JobStatus.EXPIRED
    manager.shutdown()
