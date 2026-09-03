"""PyInstaller entry point for the Power Design Toolkit workspace GUI."""

from __future__ import annotations

import sys

from llc_design.gui.app import run_gui

if __name__ == "__main__":
    sys.exit(run_gui())
