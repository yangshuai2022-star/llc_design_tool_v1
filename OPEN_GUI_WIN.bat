@echo off
setlocal
title Power Design Toolkit V8.1 (LLC + PFC)
pushd "%~dp0"

chcp 65001 >nul

set "VENV=.venv"

rem --- 1. reuse existing .venv if present ---
if exist "%VENV%\Scripts\python.exe" goto :venv_ok

rem --- 2. otherwise create .venv with a Python >=3.10 ---
echo [INFO] Creating virtualenv at %VENV% ...
set "BOOTPY="
py -3 -c "import sys;sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>nul && set "BOOTPY=py -3"
if not defined BOOTPY (
    python -c "import sys;sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>nul && set "BOOTPY=python"
)
if not defined BOOTPY (
    echo [ERROR] Python 3.10+ not found.
    echo Install from https://www.python.org/downloads/windows/
    echo and check "Add python.exe to PATH" during setup.
    pause
    exit /b 1
)
%BOOTPY% -m venv "%VENV%"
if errorlevel 1 (
    echo [ERROR] venv creation failed.
    pause
    exit /b 1
)
"%VENV%\Scripts\python.exe" -m pip install --upgrade pip >nul

:venv_ok
set "PY=%VENV%\Scripts\python.exe"

rem --- 3. install toolkit + GUI extras if missing ---
"%PY%" -c "import llc_design, pfc_design, PySide6, pyqtgraph" >nul 2>nul
if not errorlevel 1 goto :launch

echo [INFO] First run: installing power-design-toolkit with [gui] extras ...
"%PY%" -m pip install -e ".[gui]"
if errorlevel 1 (
    echo [ERROR] Install failed. Check network or pip configuration.
    pause
    exit /b 1
)
"%PY%" -c "import llc_design, pfc_design, PySide6, pyqtgraph" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Imports still fail after install.
    pause
    exit /b 1
)

:launch
echo [INFO] Launching Power Design Toolkit GUI...
"%PY%" -m llc_design gui
set "RC=%errorlevel%"
popd
exit /b %RC%