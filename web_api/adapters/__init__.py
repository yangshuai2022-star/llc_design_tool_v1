"""Workspace adapters used by the web API job runner."""

from .ttpl import (
    TTPLAdapter,
    build_inductor_request,
    build_ttpl_config,
    execute_ttpl_operation,
    run_ttpl_operation,
)

__all__ = [
    "TTPLAdapter",
    "build_inductor_request",
    "build_ttpl_config",
    "execute_ttpl_operation",
    "run_ttpl_operation",
]
