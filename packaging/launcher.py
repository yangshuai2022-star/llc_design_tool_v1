"""PyInstaller entry point for the Power Design Toolkit workspace GUI.

The ``--self-test`` path is intentionally part of the packaged executable.  CI
uses it after PyInstaller packaging so a release is not considered healthy just
because the output directory exists.
"""

from __future__ import annotations

import json
import os
import sys


def _run_packaged_self_test() -> int:
    """Exercise the packaged GUI/runtime without entering the Qt event loop."""

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtWidgets import QApplication

    from llc_design.core.spec import LLCDesignSpec
    from llc_design.gui import theme
    from llc_design.gui.launcher import WorkspaceApplicationController, WorkspaceSelectionDialog
    from llc_design.validation.provenance import validate_bundled_data

    app = QApplication.instance() or QApplication(["PowerDesignTool", "--self-test"])
    app.setApplicationName("Power Design Toolkit")
    theme.apply_app_theme(app)

    controller = WorkspaceApplicationController(LLCDesignSpec())
    selector = WorkspaceSelectionDialog()
    selector.close()

    report = validate_bundled_data()
    if not report.valid:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        for window in (
            controller.llc_window,
            controller.pfc_window,
            controller.control_window,
            controller.fra_window,
        ):
            window.close()
        return 2

    # Force every top-level workspace through construction/show/hide so missing
    # Qt plugins, packaged resources, optional imports and initialization errors
    # fail the release job instead of reaching users.
    for workspace in ("llc", "pfc", "control", "fra"):
        controller.show_workspace(workspace)
        app.processEvents()

    for window in (
        controller.llc_window,
        controller.pfc_window,
        controller.control_window,
        controller.fra_window,
    ):
        window.hide()
        window.close()
    app.processEvents()

    print("Power Design Toolkit packaged self-test: OK")
    return 0


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return _run_packaged_self_test()

    from llc_design.gui.app import run_gui

    return int(run_gui())


if __name__ == "__main__":
    sys.exit(main())
