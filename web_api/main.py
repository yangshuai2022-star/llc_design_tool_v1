"""FastAPI application factory for the web design service."""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from llc_design import __version__ as toolkit_version
from web_api import API_VERSION
from web_api.jobs import JobManager
from web_api.routes import jobs_router


def create_app(manager: JobManager | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        job_manager = manager or JobManager()
        application.state.job_manager = job_manager
        stop = asyncio.Event()

        async def cleanup_loop() -> None:
            while not stop.is_set():
                try:
                    await asyncio.wait_for(stop.wait(), timeout=60.0)
                except asyncio.TimeoutError:
                    job_manager.cleanup_expired()

        task = asyncio.create_task(cleanup_loop())
        try:
            yield
        finally:
            stop.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            job_manager.shutdown()

    application = FastAPI(
        title="Power Design Toolkit API",
        version=API_VERSION,
        lifespan=lifespan,
    )
    application.state.job_manager = manager
    application.include_router(jobs_router)

    @application.get("/api/v1/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "api_version": API_VERSION,
            "toolkit_version": toolkit_version,
        }

    return application


app = create_app()
