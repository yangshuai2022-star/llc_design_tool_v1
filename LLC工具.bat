@echo off
chcp 65001 >nul
setlocal
title 电源设计工具 V6 (LLC + PFC)
pushd "%~dp0"

rem 定位 Python: 优先 py -3, 其次 python
set PY=
py -3 -c "import sys" >nul 2>nul && set "PY=py -3"
if not defined PY (
    python -c "import sys" >nul 2>nul && set "PY=python"
)
if not defined PY (
    echo 错误: 未找到 Python 3 (需要 ^>=3.10)。
    echo 请安装: https://www.python.org/downloads/windows/ 并在安装时勾选 "Add python.exe to PATH"。
    pause
    exit /b 1
)

rem 首次运行: 自动安装本工具包 (含 GUI 依赖)
%PY% -c "import llc_design, pfc_design, PySide6" >nul 2>nul
if errorlevel 1 (
    echo 首次运行: 正在安装 power-design-toolkit (含 GUI 依赖) ...
    %PY% -m pip install -e ".[gui]"
    if errorlevel 1 (
        echo 安装失败, 请检查网络或 pip 配置。
        pause
        exit /b 1
    )
)

echo 正在启动电源设计工具图形界面...
%PY% -m llc_design gui
exit /b %errorlevel%
