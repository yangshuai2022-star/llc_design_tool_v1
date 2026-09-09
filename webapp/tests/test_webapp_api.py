from fastapi.testclient import TestClient

from webapp.app import app

client = TestClient(app)


def test_typed_request_rejects_unknown_spec_field():
    response = client.post("/api/llc/analyze", json={"spec": {"solver_loops": 999}})

    assert response.status_code == 422


def test_pdf_report_endpoint_returns_pdf():
    response = client.post("/api/llc/report", json={"spec": {}})

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF")


def test_excel_report_endpoint_returns_xlsx():
    response = client.post("/api/llc/report.xlsx", json={"spec": {}})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    assert response.content.startswith(b"PK")
