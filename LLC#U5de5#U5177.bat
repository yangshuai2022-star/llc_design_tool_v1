@echo off
setlocal
title Power Design Tool V6 (LLC + PFC)
pushd "%~dp0"

chcp 65001 >nul

rem Locate Python 3: prefer the py launcher, fall back to python
set "PYCMD="
where py >nul 2>nul
if not errorlevel 1 py -3 -c "import sys" >nul 2>nul
if not errorlevel 1 set "PYCMD=py -3"
if not defined PYCMD (
    where python >nul 2>nul
    if not errorlevel 1 python -c "import sys" >nul 2>nul
    if not errorlevel 1 set "PYCMD=python"
)
if not defined PYCMD (
    echo [ERROR] Python 3 ^(>=3.10^) not found.
    echo Please install from https://www.python.org/downloads/windows/
    echo and check "Add python.exe to PATH" during setup.
    pause
    exit /b 1
)

rem Already installed? Skip install and launch directly.
%PYCMD% -c "import llc_design, pfc_design, PySide6" >nul 2>nul
if not errorlevel 1 goto :launch

echo [INFO] First run: installing power-design-toolkit with GUI dependencies...
%PYCMD% -m pip install -e ".[gui]"
if errorlevel 1 (
    echo [ERROR] Install failed. Check network or pip configuration.
    pause
    exit /b 1
)

:launch
echo [INFO] Launching Power Design Tool GUI...
%PYCMD% -m llc_design gui
exit /b %errorlevel%
