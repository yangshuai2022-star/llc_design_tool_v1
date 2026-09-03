#!/bin/bash
# Power Design Toolkit V8.1 - macOS one-click GUI launcher (LLC + PFC)
# Place this file inside the project directory; double-click to open the GUI.

set -u

# --- resolve this script's directory (portable, no readlink -f) ---
SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
case "$SCRIPT_PATH" in
  /*) : ;;
  *) SCRIPT_PATH="$PWD/$SCRIPT_PATH" ;;
esac
while [ -L "$SCRIPT_PATH" ]; do
  dir="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)"
  SCRIPT_PATH="$(readlink -- "$SCRIPT_PATH")"
  case "$SCRIPT_PATH" in /*) : ;; *) SCRIPT_PATH="$dir/$SCRIPT_PATH" ;; esac
done
SCRIPT_DIR="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)"
cd "$SCRIPT_DIR" || { echo "[ERROR] cannot cd to $SCRIPT_DIR"; exit 1; }

VENV_DIR="./.venv"

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
        echo "[ERROR] Python 3.10+ not found. Install from https://www.python.org/downloads/macos/"
        read -r -p "Press Enter to exit..." _ < /dev/tty
        exit 1
    }
    echo "[INFO] Using bootstrap interpreter: $BOOT_PY ($("$BOOT_PY" --version 2>&1))"
    "$BOOT_PY" -m venv "$VENV_DIR" || {
        echo "[ERROR] venv creation failed. On macOS install: brew install python@3.12"
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
        echo "[ERROR] pip install failed. Check network or pip configuration."
        read -r -p "Press Enter to exit..." _ < /dev/tty
        exit 1
    fi
    if ! "$PY" -c 'import llc_design, pfc_design, PySide6, pyqtgraph' >/dev/null 2>&1; then
        echo "[ERROR] install reported success but imports still fail."
        read -r -p "Press Enter to exit..." _ < /dev/tty
        exit 1
    fi
fi

echo "[INFO] Launching Power Design Toolkit GUI ..."
exec "$PY" -m llc_design gui