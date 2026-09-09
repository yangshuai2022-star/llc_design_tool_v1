"""FastAPI entry point for browser-based Power Design Toolkit operation."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from backend.api.control import router as control_router
from llc_design import __version__

from .reporting import build_excel_report, build_pdf_report
from .schemas import LLCAnalyzeRequest, OptimizeRequest, ReportRequest
from .service import analyze_llc, core_catalog, default_payload, optimize_llc

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"

app = FastAPI(
    title="Power Design Toolkit Web",
    version=__version__,
    description="Server-side Python LLC engineering calculation API and browser UI.",
)
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.include_router(control_router)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/llc/defaults")
def llc_defaults() -> dict[str, Any]:
    return default_payload()


@app.post("/api/llc/analyze")
def llc_analyze(request: LLCAnalyzeRequest) -> dict[str, Any]:
    try:
        return analyze_llc(request.spec.service_payload())
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        # Keep stack traces server-side.  The browser receives a concise error.
        raise HTTPException(status_code=500, detail=f"LLC calculation failed: {exc}") from exc


@app.get("/api/llc/cores")
def llc_cores() -> dict[str, Any]:
    return core_catalog()


@app.post("/api/llc/optimize")
def llc_optimize(request: OptimizeRequest) -> dict[str, Any]:
    try:
        return optimize_llc(request.service_payload())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLC optimization failed: {exc}") from exc


@app.post("/api/llc/report")
def llc_report(request: ReportRequest) -> Response:
    pdf = build_pdf_report(
        request.spec.service_payload(),
        project=request.project,
        engineer=request.engineer,
    )
    return Response(
        pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="llc_design_report.pdf"'},
    )


@app.post("/api/llc/report.xlsx")
def llc_report_xlsx(request: LLCAnalyzeRequest) -> Response:
    xlsx = build_excel_report(request.spec.service_payload())
    return Response(
        xlsx,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="llc_design_report.xlsx"'},
    )


@app.get("/{full_path:path}", include_in_schema=False)
def spa_fallback(full_path: str) -> FileResponse:
    """SPA fallback so hash routes (/llc, /control/loop, ...) load the shell."""
    if full_path.startswith(("api/", "static/")):
        raise HTTPException(status_code=404)
    return FileResponse(STATIC / "index.html")


def run() -> None:
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("webapp.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run()
