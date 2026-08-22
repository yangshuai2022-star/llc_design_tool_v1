"""FastAPI application factory for the web design service."""

from fastapi import FastAPI

from llc_design import __version__ as toolkit_version
from web_api import API_VERSION


def create_app() -> FastAPI:
    application = FastAPI(
        title="Power Design Toolkit API",
        version=API_VERSION,
    )

    @application.get("/api/v1/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "api_version": API_VERSION,
            "toolkit_version": toolkit_version,
        }

    return application


app = create_app()
