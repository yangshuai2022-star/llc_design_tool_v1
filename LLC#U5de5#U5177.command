#!/bin/bash
# LLC 设计工具 V3 - macOS 一键启动图形界面
# 位置: 放在 llc_design_tool_v3 目录内, 双击即可打开 GUI

cd "$(dirname "$0")" || exit 1

# 定位 Python: 优先专用 venv, 其次系统 python3
PY=""
if [ -x "/Users/yangshuai/venvs/sci/bin/python" ]; then
    PY="/Users/yangshuai/venvs/sci/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
fi

if [ -z "$PY" ]; then
    echo "错误: 未找到 Python 3 (需要 >=3.10)。"
    read -r -p "按回车退出..."
    exit 1
fi

# 首次运行: 自动安装本工具包 (含 GUI 依赖)
if ! "$PY" -c "import llc_design, PySide6" >/dev/null 2>&1; then
    echo "首次运行: 正在安装 power-design-toolkit (含 GUI 依赖) ..."
    if ! "$PY" -m pip install -e ".[gui]"; then
        echo "安装失败, 请检查网络或 pip 配置。"
        read -r -p "按回车退出..."
        exit 1
    fi
fi

echo "正在启动 LLC 设计工具图形界面..."
"$PY" -m llc_design gui
exit $?
