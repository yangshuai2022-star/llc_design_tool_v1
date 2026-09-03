#!/usr/bin/env bash
# Power Design Toolkit V8.1 - Linux/WSL one-click GUI launcher
# Double-click in a file manager that runs .sh, or run from a terminal:
#   bash run_gui.sh
#
# Behaviour:
#   1. cd into this script's directory
#   2. use ./.venv if present, otherwise create it
#   3. install the toolkit with [gui] extras on first run (or if missing)
#   4. launch: python -m llc_design gui

set -u

# --- locate this script's directory (resolve symlinks) ---
SCRIPT_DIR="$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]:-$0}")")" && pwd)"
cd "$SCRIPT_DIR" || { echo "[ERROR] cannot cd to $SCRIPT_DIR"; exit 1; }

VENV_DIR="./.venv"
PY=""

# --- pick a base python3 (>=3.10) to bootstrap the venv ---
pick_python() {
    local cmd
    for cmd in python3 python; do
        if command -v "$cmd" >/dev/null 2>&1; then
            if "$cmd" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
                command -v "$cmd"
                return 0
            fi
        fi
    done
    return 1
}

# --- ensure a venv exists ---
if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "[INFO] Creating virtualenv at $VENV_DIR ..."
    BOOT_PY="$(pick_python)" || {
        echo "[ERROR] Python 3.10+ not found. Install python3 and python3-venv, then re-run."
        read -r -p "Press Enter to exit..." _ < /dev/tty
        exit 1
    }
    echo "[INFO] Using bootstrap interpreter: $BOOT_PY ($("$BOOT_PY" --version 2>&1))"
    "$BOOT_PY" -m venv "$VENV_DIR" || {
        echo "[ERROR] venv creation failed. On Debian/Ubuntu install: sudo apt install python3-venv"
        read -r -p "Press Enter to exit..." _ < /dev/tty
        exit 1
    }
    "$VENV_DIR/bin/python" -m pip install --upgrade pip >/dev/null
fi

PY="$VENV_DIR/bin/python"

# --- first run / missing deps: install toolkit + GUI extras ---
if ! "$PY" -c 'import llc_design, pfc_design, PySide6, pyqtgraph' >/dev/null 2>&1; then
    echo "[INFO] First run: installing power-design-toolkit with [gui] extras ..."
    if ! "$PY" -m pip install -e ".[gui]"; then
        echo "[ERROR] pip install failed. Check network/pip, then re-run this script."
        read -r -p "Press Enter to exit..." _ < /dev/tty
        exit 1
    fi
    if ! "$PY" -c 'import llc_design, pfc_design, PySide6, pyqtgraph' >/dev/null 2>&1; then
        echo "[ERROR] install reported success but imports still fail."
        read -r -p "Press Enter to exit..." _ < /dev/tty
        exit 1
    fi
fi

# --- optional: warn if no display (WSLg / X11) ---
if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
    echo "[WARN] No DISPLAY/WAYLAND_DISPLAY detected. The GUI may not appear."
    echo "       WSL users: ensure WSLg is enabled (default on Windows 11)."
fi

echo "[INFO] Launching Power Design Toolkit GUI ..."
exec "$PY" -m llc_design gui