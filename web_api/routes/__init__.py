"""HTTP route groups for the versioned web API."""

from .jobs import router as jobs_router

__all__ = ["jobs_router"]
