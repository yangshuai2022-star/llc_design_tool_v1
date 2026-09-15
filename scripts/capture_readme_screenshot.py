"""Capture a real launcher screenshot for README/documentation.

This renders the actual Qt launcher offscreen. It is intentionally not a mock
or a hand-drawn replacement for the application UI.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from llc_design.gui import theme
from llc_design.gui.i18n_ui import apply_language, bind_window
from llc_design.gui.launcher import WorkspaceSelectionDialog
from llc_design.i18n import set_language


def capture(path: Path, language: str = "en") -> Path:
    app = QApplication.instance() or QApplication(["capture-readme-screenshot"])
    app.setApplicationName("Power Design Toolkit")

    # The selector contains a few QLabel strings that are translated at
    # construction time rather than rebound later. Select the requested
    # language *before* constructing the dialog so a generated English/Japanese/
    # Korean screenshot cannot silently retain Chinese title/subtitle text.
    set_language(language, notify=False)
    theme.apply_app_theme(app)

    dialog = WorkspaceSelectionDialog()
    bind_window(dialog)
    apply_language(language, app)
    dialog.show()
    app.processEvents()

    path.parent.mkdir(parents=True, exist_ok=True)
    pixmap = dialog.grab()
    if pixmap.isNull() or not pixmap.save(str(path), "PNG"):
        raise RuntimeError(f"failed to save launcher screenshot to {path}")

    dialog.close()
    app.processEvents()
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--language", default="en", choices=("zh-Hans", "en", "ja", "ko"))
    args = parser.parse_args()
    output = capture(args.output, args.language)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
