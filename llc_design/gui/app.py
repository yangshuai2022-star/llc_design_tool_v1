"""GUI entry point for the separated LLC/PFC engineering workspaces."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from ..core.config import load_spec
from ..core.spec import LLCDesignSpec
from . import theme
from .launcher import WorkspaceApplicationController


def run_gui(config_path: str | None = None) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Power Design Toolkit")
    theme.apply_app_theme(app)
    spec = load_spec(config_path) if config_path else LLCDesignSpec()
    controller = WorkspaceApplicationController(spec)
    if not controller.start():
        return 0
    # Keep the controller strongly reachable for the lifetime of the Qt event loop.
    app._power_design_controller = controller  # type: ignore[attr-defined]
    return int(app.exec())
