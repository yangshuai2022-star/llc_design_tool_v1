from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient

from web_api.contracts import JobRequest, JobResultPayload
from web_api.jobs import JobManager
from web_api.main import create_app


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


def _manager(tmp_path: Path) -> JobManager:
    return JobManager(
        executor=ThreadPoolExecutor(max_workers=1),
        temp_root=tmp_path,
        dispatchers={("llc", "system"): _payload},
    )


def test_job_routes_return_stable_statuses_and_location(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    with TestClient(create_app(manager)) as client:
        response = client.post(
            "/api/v1/jobs",
            json={"workspace": "llc", "operation": "system", "config": {}},
        )
        assert response.status_code == 202
        assert response.headers["location"].endswith(f"/api/v1/jobs/{response.json()['id']}")
        job_id = response.json()["id"]
        assert client.get(f"/api/v1/jobs/{job_id}").status_code == 200
        manager.future(job_id).result(timeout=2.0)
        assert client.get(f"/api/v1/jobs/{job_id}/result").status_code == 200
        assert client.get("/api/v1/jobs/unknown/result").status_code == 404


def test_pending_result_is_409_and_delete_is_idempotent(tmp_path: Path) -> None:
    from threading import Event

    release = Event()

    def blocking(request: JobRequest, path: Path, event: object) -> JobResultPayload:
        release.wait(timeout=2.0)
        return _payload(request, path, event)

    manager = JobManager(
        executor=ThreadPoolExecutor(max_workers=1),
        temp_root=tmp_path,
        dispatchers={("llc", "system"): blocking},
    )
    with TestClient(create_app(manager)) as client:
        job_id = client.post(
            "/api/v1/jobs",
            json={"workspace": "llc", "operation": "system", "config": {}},
        ).json()["id"]
        assert client.get(f"/api/v1/jobs/{job_id}/result").status_code == 409
        assert client.delete(f"/api/v1/jobs/{job_id}").status_code == 200
        assert client.delete(f"/api/v1/jobs/{job_id}").status_code == 200
        release.set()


def test_artifact_routes_download_only_registered_files(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    with TestClient(create_app(manager)) as client:
        job_id = client.post(
            "/api/v1/jobs",
            json={"workspace": "llc", "operation": "system", "config": {}},
        ).json()["id"]
        manager.future(job_id).result(timeout=2.0)
        job_dir = manager.job_dir(job_id)
        (job_dir / "report.txt").write_text("report", encoding="utf-8")
        artifact = manager.register_artifact(job_id, "report.txt")
        listing = client.get(f"/api/v1/jobs/{job_id}/artifacts")
        assert listing.status_code == 200
        assert listing.json()[0]["id"] == artifact.id
        download = client.get(f"/api/v1/jobs/{job_id}/artifacts/{artifact.id}")
        assert download.status_code == 200
        assert download.content == b"report"
        assert download.headers["cache-control"] == "no-store"
        assert download.headers["x-content-type-options"] == "nosniff"
