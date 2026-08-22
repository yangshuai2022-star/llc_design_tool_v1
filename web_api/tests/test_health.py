from web_api.main import create_app


def test_health_reports_matching_toolkit_version():
    from fastapi.testclient import TestClient

    response = TestClient(create_app()).get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "api_version": "1.0.0",
        "toolkit_version": "7.5.0",
    }
