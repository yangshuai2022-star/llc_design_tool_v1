"""Complete LLC digital voltage-loop design workbench.

V7.1 focuses on information architecture: a compact clickable signal-flow
schematic, a context-sensitive parameter inspector, and a result area that
keeps the Bode plot large.  The numerical model is unchanged.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from ...control.digital_loop import (
    ADCSamplingConfig,
    AnalogSenseConfig,
    CommandTimingConfig,
    ControllerKind,
    DigitalLoopAnalysis,
    FMLUTMode,
    FrequencyModulatorLUT,
    PIControllerConfig,
    PIFControllerConfig,
    PWMCountMode,
    TwoP2ZControllerConfig,
)
from ...control.linearize import ControlInputKind
from ...control.phase_budget import phase_budget
from .bode_cursor import (
    BodeCursor,
    BodeCursorMeasurement,
    BodeCursorTrace,
    format_frequency,
)
from .control_block_diagram import BlockSpec, ConnectionSpec, ControlBlockDiagram
from .sense_schematic import AnalogSenseSchematic
from .. import theme
from power_codegen import generate_llc_control_code


class DigitalLoopView(QWidget):
    """PI/PIF/2P2Z, FM LUT, sensing and complete open-loop analysis."""

    analysis_requested = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.result: DigitalLoopAnalysis | None = None
        self._bode_cursor: BodeCursor | None = None
        self._cursor_frequency_hz: float | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(7)

        diagram_header = QHBoxLayout()
        title = QLabel("LLC 数字控制信号链")
        title.setStyleSheet(f"font-size:15px;font-weight:600;color:{theme.active_theme().text};")
        diagram_header.addWidget(title)
        hint = QLabel("点击模块联动参数/Bode · 滚轮缩放 · 拖动平移 · 双击空白适应窗口")
        hint.setStyleSheet(f"color:{theme.active_theme().text_muted};")
        diagram_header.addWidget(hint)
        diagram_header.addStretch(1)

        self.fit_diagram_button = QPushButton("适应")
        self.fit_diagram_button.setToolTip("适应当前窗口（双击图中空白也可执行）")
        self.actual_diagram_button = QPushButton("100%")
        self.actual_diagram_button.setToolTip("恢复 100% 矢量比例")
        self.zoom_out_button = QPushButton("−")
        self.zoom_out_button.setFixedWidth(34)
        self.zoom_out_button.setToolTip("缩小控制框图")
        self.zoom_in_button = QPushButton("＋")
        self.zoom_in_button.setFixedWidth(34)
        self.zoom_in_button.setToolTip("放大控制框图")
        self.fullscreen_diagram_button = QPushButton("全屏框图")
        self.fullscreen_diagram_button.setToolTip("在独立大窗口中查看、缩放和选择控制模块")
        diagram_header.addWidget(self.fit_diagram_button)
        diagram_header.addWidget(self.actual_diagram_button)
        diagram_header.addWidget(self.zoom_out_button)
        diagram_header.addWidget(self.zoom_in_button)
        diagram_header.addWidget(self.fullscreen_diagram_button)

        self.inspector_toggle = QPushButton("隐藏环节参数")
        self.inspector_toggle.setCheckable(True)
        self.inspector_toggle.setChecked(True)
        self.inspector_toggle.setToolTip("隐藏/显示数字环路的局部参数检查器")
        diagram_header.addWidget(self.inspector_toggle)
        root.addLayout(diagram_header)

        self._diagram_blocks = [
            BlockSpec("sum", "Σ", "Vref − Vfb", 15, 20, 82, 62),
            BlockSpec("controller", "PI / PIF / 2P2Z", "C(z)", 126, 20, 168, 62, "controller"),
            BlockSpec("clamp", "PCMD Clamp", "0 … 1", 324, 20, 132, 62),
            BlockSpec("fm", "FM LUT", "PCMD → Fsw", 486, 20, 150, 62, "modulator"),
            BlockSpec("pwm", "PWM / ZOH", "Zero update + delay", 666, 20, 158, 62, "modulator"),
            BlockSpec("plant", "LLC Power Stage", "Gvf(s)", 854, 20, 165, 62, "plant"),
            BlockSpec("sense", "Divider / OpAmp / RC", "Analog sense", 650, 118, 194, 62, "sense"),
            BlockSpec("adc", "ADC / Averaging", "S/H + digital delay", 412, 118, 194, 62, "sense"),
        ]
        self._diagram_connections = [
            ConnectionSpec("sum", "controller", "e"),
            ConnectionSpec("controller", "clamp", "PCMD"),
            ConnectionSpec("clamp", "fm"),
            ConnectionSpec("fm", "pwm", "Fsw"),
            ConnectionSpec("pwm", "plant"),
            ConnectionSpec("plant", "sense", "Vout"),
            ConnectionSpec("sense", "adc", "Vsen"),
            ConnectionSpec("adc", "sum", "Vfb", feedback=True),
        ]

        self.diagram = ControlBlockDiagram()
        self.diagram.setMinimumHeight(190)
        self.diagram.setMaximumHeight(235)
        self.diagram.set_diagram(self._diagram_blocks, self._diagram_connections)
        self.diagram.block_selected.connect(self._diagram_selected)
        root.addWidget(self.diagram, 0)

        self.fit_diagram_button.clicked.connect(self.diagram.fit_to_view)
        self.actual_diagram_button.clicked.connect(self.diagram.actual_size)
        self.zoom_out_button.clicked.connect(self.diagram.zoom_out)
        self.zoom_in_button.clicked.connect(self.diagram.zoom_in)
        self.fullscreen_diagram_button.clicked.connect(self._open_diagram_fullscreen)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(True)
        self.splitter.setHandleWidth(5)
        self.splitter.addWidget(self._build_parameter_inspector())
        self.splitter.addWidget(self._build_result_panel())
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([355, 1150])
        root.addWidget(self.splitter, 1)
        self.inspector_toggle.toggled.connect(self._toggle_inspector)

        self._show_parameter_page("controller", update_diagram=True)

    def _open_diagram_fullscreen(self) -> None:
        """Open a large vector copy of the control chain for focused inspection."""
        dialog = QDialog(self)
        dialog.setWindowTitle("LLC 数字控制信号链 — 全屏查看")
        dialog.setModal(True)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        header = QHBoxLayout()
        heading = QLabel("LLC 数字控制信号链")
        heading.setStyleSheet(f"font-size:18px;font-weight:700;color:{theme.active_theme().text_strong};")
        header.addWidget(heading)
        note = QLabel("滚轮缩放 · 左键拖动平移 · 双击空白适应窗口 · 点击模块同步主界面")
        note.setStyleSheet(f"color:{theme.active_theme().text_muted};")
        header.addWidget(note)
        header.addStretch(1)

        big = ControlBlockDiagram(dialog)
        fit_button = QPushButton("适应窗口")
        actual_button = QPushButton("100%")
        minus_button = QPushButton("−")
        plus_button = QPushButton("＋")
        close_button = QPushButton("关闭")
        header.addWidget(fit_button)
        header.addWidget(actual_button)
        header.addWidget(minus_button)
        header.addWidget(plus_button)
        header.addWidget(close_button)
        layout.addLayout(header)

        big.setMinimumHeight(620)
        big.set_diagram(list(self._diagram_blocks), list(self._diagram_connections))
        if self.diagram.selected_key and big.has_block(self.diagram.selected_key):
            big.select_block(self.diagram.selected_key)
        layout.addWidget(big, 1)

        fit_button.clicked.connect(big.fit_to_view)
        actual_button.clicked.connect(big.actual_size)
        minus_button.clicked.connect(big.zoom_out)
        plus_button.clicked.connect(big.zoom_in)
        close_button.clicked.connect(dialog.accept)

        def sync_selection(key: str) -> None:
            self.diagram.select_block(key, emit=False)
            self._diagram_selected(key)

        big.block_selected.connect(sync_selection)
        dialog.resize(1500, 880)
        dialog.setWindowState(dialog.windowState() | Qt.WindowState.WindowMaximized)
        dialog.exec()

    @staticmethod
    def _detail_definition(key: str) -> tuple[str, list[BlockSpec], list[ConnectionSpec]]:
        """Return a compact, enlarged sub-diagram for the selected control block."""
        canonical = {"sum": "controller", "clamp": "controller"}.get(key, key)
        if canonical == "controller":
            return (
                "数字控制器 / PCMD 限幅",
                [
                    BlockSpec("err", "误差 e", "Vref − Vfb", 8, 24, 90, 64),
                    BlockSpec("cz", "PI / PIF / 2P2Z", "C(z)", 128, 24, 156, 64, "controller"),
                    BlockSpec("limit", "PCMD Clamp", "0 … 1", 314, 24, 126, 64),
                ],
                [ConnectionSpec("err", "cz"), ConnectionSpec("cz", "limit", "PCMD")],
            )
        if canonical == "fm":
            return (
                "FM LUT：PCMD → 实际开关频率",
                [
                    BlockSpec("pcmd", "PCMD", "0 … 1", 8, 24, 88, 64),
                    BlockSpec("lut", "FM LUT", "PCMD → TBPRD", 126, 24, 126, 64, "modulator"),
                    BlockSpec("tbprd", "TBPRD", "120 MHz timer", 282, 24, 118, 64),
                    BlockSpec("fsw", "Fsw", "local Kfm", 430, 24, 92, 64, "plant"),
                ],
                [
                    ConnectionSpec("pcmd", "lut"), ConnectionSpec("lut", "tbprd"),
                    ConnectionSpec("tbprd", "fsw"),
                ],
            )
        if canonical == "pwm":
            return (
                "PWM / ZOH / 更新延迟",
                [
                    BlockSpec("cmd", "Fsw Cmd", "controller output", 8, 24, 104, 64),
                    BlockSpec("compute", "Compute", "CLA delay", 142, 24, 108, 64, "modulator"),
                    BlockSpec("load", "Zero / GLD", "PWM update", 280, 24, 112, 64, "modulator"),
                    BlockSpec("zoh", "ZOH", "effective Fsw", 422, 24, 92, 64),
                ],
                [
                    ConnectionSpec("cmd", "compute"), ConnectionSpec("compute", "load"),
                    ConnectionSpec("load", "zoh"),
                ],
            )
        if canonical == "plant":
            return (
                "LLC 功率级小信号对象",
                [
                    BlockSpec("fin", "Fsw", "Hz", 28, 24, 90, 64),
                    BlockSpec("gvf", "LLC Gvf(s)", "frequency → Vout", 158, 24, 176, 64, "plant"),
                    BlockSpec("vout", "Vout", "output voltage", 374, 24, 112, 64),
                ],
                [ConnectionSpec("fin", "gvf"), ConnectionSpec("gvf", "vout")],
            )
        if canonical == "sense":
            return (
                "Vout 模拟采样前端",
                [
                    BlockSpec("vout", "Vout", "power output", 4, 24, 84, 64),
                    BlockSpec("div", "Divider", "Rup/Rlow/Cdiv", 112, 24, 116, 64, "sense"),
                    BlockSpec("amp", "OpAmp", "gain / GBW", 252, 24, 106, 64, "sense"),
                    BlockSpec("rc", "ADC RC", "RADC/CADC", 382, 24, 112, 64, "sense"),
                ],
                [
                    ConnectionSpec("vout", "div"), ConnectionSpec("div", "amp"),
                    ConnectionSpec("amp", "rc"),
                ],
            )
        if canonical == "adc":
            return (
                "ADC / 多 SOC / 递归平均",
                [
                    BlockSpec("pin", "ADC Pin", "analog", 4, 24, 88, 64),
                    BlockSpec("sh", "S/H + SOC", "acquisition", 116, 24, 112, 64, "sense"),
                    BlockSpec("conv", "Convert", "EOC delay", 252, 24, 106, 64, "sense"),
                    BlockSpec("avg", "Average", "SOC + recursive", 382, 24, 118, 64, "sense"),
                    BlockSpec("vfb", "Vfb", "feedback", 524, 24, 82, 64),
                ],
                [
                    ConnectionSpec("pin", "sh"), ConnectionSpec("sh", "conv"),
                    ConnectionSpec("conv", "avg"), ConnectionSpec("avg", "vfb"),
                ],
            )
        return (
            "当前环节",
            [BlockSpec("block", canonical, "selected block", 30, 24, 180, 64)],
            [],
        )

    def _update_detail_diagram(self, key: str) -> None:
        if not hasattr(self, "detail_diagram"):
            return
        title, blocks, connections = self._detail_definition(key)
        self.detail_title.setText(title)
        self.detail_diagram.set_diagram(blocks, connections)

    def _toggle_inspector(self, visible: bool) -> None:
        if visible:
            total = max(sum(self.splitter.sizes()), 900)
            self.splitter.setSizes([355, max(total - 355, 500)])
            self.inspector_toggle.setText("隐藏环节参数")
        else:
            total = max(sum(self.splitter.sizes()), 900)
            self.splitter.setSizes([0, total])
            self.inspector_toggle.setText("显示环节参数")

    @staticmethod
    def _double(minimum: float, maximum: float, decimals: int, value: float, suffix: str = "") -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(minimum, maximum)
        box.setDecimals(decimals)
        box.setValue(value)
        box.setSuffix(suffix)
        box.setKeyboardTracking(False)
        box.setMinimumWidth(118)
        return box

    @staticmethod
    def _configure_form(form: QFormLayout) -> None:
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(6)

    @staticmethod
    def _page(title: str, description: str) -> tuple[QWidget, QVBoxLayout]:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 10)
        layout.setSpacing(7)
        heading = QLabel(title)
        heading.setStyleSheet(f"font-size:14px;font-weight:600;color:{theme.active_theme().text};")
        layout.addWidget(heading)
        desc = QLabel(description)
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color:{theme.active_theme().text_muted};padding-bottom:3px;")
        layout.addWidget(desc)
        return content, layout

    @staticmethod
    def _scroll_page(content: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(content)
        return scroll

    def _build_parameter_inspector(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(320)
        panel.setMaximumWidth(470)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 6, 0)
        layout.setSpacing(6)

        header = QHBoxLayout()
        header.addWidget(QLabel("参数环节"))
        self.parameter_selector = QComboBox()
        for label, key in (
            ("工作点 / 功率级", "plant"),
            ("数字控制器 / Clamp", "controller"),
            ("FM LUT", "fm"),
            ("PWM / ZOH / 延迟", "pwm"),
            ("Vout 模拟采样", "sense"),
            ("ADC / Average", "adc"),
        ):
            self.parameter_selector.addItem(label, key)
        self.parameter_selector.currentIndexChanged.connect(self._parameter_selector_changed)
        header.addWidget(self.parameter_selector, 1)
        layout.addLayout(header)

        detail_box = QGroupBox("当前环节放大图")
        detail_layout = QVBoxLayout(detail_box)
        detail_layout.setContentsMargins(6, 7, 6, 7)
        detail_layout.setSpacing(5)
        self.detail_title = QLabel("数字控制器 / PCMD 限幅")
        self.detail_title.setStyleSheet(f"font-weight:600;color:{theme.active_theme().text};")
        detail_layout.addWidget(self.detail_title)
        self.detail_diagram = ControlBlockDiagram()
        self.detail_diagram.setMinimumHeight(150)
        self.detail_diagram.setMaximumHeight(190)
        detail_layout.addWidget(self.detail_diagram)
        detail_hint = QLabel("局部图为矢量图，可滚轮放大；顶部“全屏框图”用于查看完整闭环。")
        detail_hint.setWordWrap(True)
        detail_hint.setStyleSheet(f"color:{theme.active_theme().text_muted};font-size:11px;")
        detail_layout.addWidget(detail_hint)
        layout.addWidget(detail_box)

        self.parameter_stack = QStackedWidget()
        self._page_for_block: dict[str, int] = {}
        self._canonical_block: dict[str, str] = {
            "sum": "controller",
            "controller": "controller",
            "clamp": "controller",
            "fm": "fm",
            "pwm": "pwm",
            "plant": "plant",
            "sense": "sense",
            "adc": "adc",
        }

        # Work point / plant
        page, pl = self._page(
            "工作点与功率级",
            "这里决定 Gvf(s) 的线性化工作点。功率级本体来自 LLC 小信号模型。",
        )
        work = QGroupBox("工作点")
        form = QFormLayout(work); self._configure_form(form)
        self.vbus = self._double(20, 2000, 2, 400, " V")
        self.load_percent = self._double(1, 150, 1, 100, " %")
        self.sample_us = self._double(0.5, 1000, 3, 20, " µs")
        form.addRow("母线电压", self.vbus)
        form.addRow("负载", self.load_percent)
        form.addRow("控制周期", self.sample_us)
        pl.addWidget(work); pl.addStretch(1)
        self._add_parameter_page("plant", page)

        # Controller / clamp
        page, pl = self._page(
            "数字控制器",
            "控制器输出为 PCMD。PI/PIF 使用固件一致的 Tustin 积分与限幅；2P2Z 直接使用差分方程系数。",
        )
        controller = QGroupBox("C(z) 与输出限制")
        form = QFormLayout(controller); self._configure_form(form)
        self.controller_kind = QComboBox()
        self.controller_kind.addItem("PI（固件 Tustin）", ControllerKind.PI)
        self.controller_kind.addItem("PIF（PI + 输出 LPF）", ControllerKind.PIF)
        self.controller_kind.addItem("2P2Z（直接系数）", ControllerKind.TWO_P_TWO_Z)
        self.kp = self._double(1e-8, 1e6, 8, 0.01)
        self.ti_ms = self._double(0.001, 1e6, 6, 1.0, " ms")
        self.pif_fc = self._double(0, 1e6, 2, 3500, " Hz")
        self.out_min = self._double(-10, 10, 5, 0.0)
        self.out_max = self._double(-10, 10, 5, 1.0)
        self.b0 = self._double(-1e9, 1e9, 10, 0.0)
        self.b1 = self._double(-1e9, 1e9, 10, 0.0)
        self.b2 = self._double(-1e9, 1e9, 10, 0.0)
        self.a1 = self._double(-10, 10, 10, 0.0)
        self.a2 = self._double(-10, 10, 10, 0.0)
        form.addRow("类型", self.controller_kind)
        form.addRow("Kp", self.kp)
        form.addRow("Ti", self.ti_ms)
        form.addRow("PIF 截止频率", self.pif_fc)
        form.addRow("输出下限", self.out_min)
        form.addRow("输出上限", self.out_max)
        form.addRow("2P2Z b0", self.b0)
        form.addRow("2P2Z b1", self.b1)
        form.addRow("2P2Z b2", self.b2)
        form.addRow("2P2Z a1_den", self.a1)
        form.addRow("2P2Z a2_den", self.a2)
        self.controller_kind.currentIndexChanged.connect(self._update_controller_fields)
        pl.addWidget(controller); pl.addStretch(1)
        self._add_parameter_page("controller", page, aliases=("sum", "clamp"))

        # FM LUT
        page, pl = self._page(
            "FM 调制器",
            "PCMD 通过用户 LUT 映射到 TBPRD 或 Fsw；局部斜率 Kfm 会进入开环增益。",
        )
        fm = QGroupBox("PCMD → TBPRD / Frequency")
        form = QFormLayout(fm); self._configure_form(form)
        self.fm_mode = QComboBox()
        self.fm_mode.addItem("PCMD → TBPRD（固件一致）", FMLUTMode.PCMD_TO_TBPRD)
        self.fm_mode.addItem("PCMD → Frequency", FMLUTMode.PCMD_TO_FREQUENCY)
        self.auto_pcmd = QCheckBox("由功率级工作频率反求 PCMD")
        self.auto_pcmd.setChecked(True)
        self.pcmd = self._double(0, 1, 6, 0.5)
        self.pcmd.setEnabled(False)
        self.auto_pcmd.toggled.connect(lambda checked: self.pcmd.setEnabled(not checked))
        self.lut_text = QPlainTextEdit()
        self.lut_text.setPlainText(FrequencyModulatorLUT.firmware_default().to_text())
        self.lut_text.setMinimumHeight(235)
        self.lut_text.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        form.addRow("LUT 类型", self.fm_mode)
        form.addRow(self.auto_pcmd)
        form.addRow("手动 PCMD", self.pcmd)
        form.addRow("用户 LUT", self.lut_text)
        pl.addWidget(fm); pl.addStretch(1)
        self._add_parameter_page("fm", page)

        # PWM / timing
        page, pl = self._page(
            "PWM / ZOH / Update Delay",
            "TBCLK、计数方式、ZOH 和 CLA 计算延迟共同决定 PCMD 到实际开关频率的时序相位损失。",
        )
        timing = QGroupBox("PWM 时序")
        form = QFormLayout(timing); self._configure_form(form)
        self.timer_mhz = self._double(1, 2000, 3, 120, " MHz")
        self.count_mode = QComboBox()
        self.count_mode.addItem("Up-Down", PWMCountMode.UP_DOWN)
        self.count_mode.addItem("Up", PWMCountMode.UP)
        self.computation_us = self._double(0, 1000, 4, 1.0, " µs")
        self.include_zoh = QCheckBox("包含 Zero-Order Hold")
        self.include_zoh.setChecked(True)
        form.addRow("TBCLK", self.timer_mhz)
        form.addRow("计数模式", self.count_mode)
        form.addRow("CLA 计算时间", self.computation_us)
        form.addRow(self.include_zoh)
        note = QLabel("PWM 命令在 Counter-Zero / Global-Load 生效；分析会给出最小、标称、最大 Zero 等待延迟。")
        note.setWordWrap(True); note.setStyleSheet(f"color:{theme.active_theme().text_muted};")
        pl.addWidget(timing); pl.addWidget(note); pl.addStretch(1)
        self._add_parameter_page("pwm", page)

        # Analog sensing
        page, pl = self._page(
            "Vout 外部模拟采样链",
            "分压器、Rlow 并联电容、运放和 ADC 前 RC 均会进入反馈链幅值与相位。",
        )
        sense = QGroupBox("模拟前端")
        sense_layout = QVBoxLayout(sense)
        self.sense_schematic = AnalogSenseSchematic("LLC Vout sensing")
        self.sense_schematic.set_labels(
            source="Vout", front="Rup / Rlow / Cdiv", amp="Buffer / OpAmp",
            rc="RADC / CADC", adc="ADC / Average",
        )
        sense_layout.addWidget(self.sense_schematic)
        form = QFormLayout(); self._configure_form(form); sense_layout.addLayout(form)
        self.rup_k = self._double(0.001, 1e6, 4, 117.0, " kΩ")
        self.rlow_k = self._double(0.001, 1e6, 4, 1.6, " kΩ")
        self.cdiv_nf = self._double(0, 1e6, 4, 1.0, " nF")
        self.opamp_gain = self._double(0.001, 1e6, 5, 1.0)
        self.opamp_bw_khz = self._double(0, 1e9, 2, 0.0, " kHz")
        self.adc_r = self._double(0, 1e9, 3, 220.0, " Ω")
        self.adc_c_nf = self._double(0, 1e6, 4, 2.0, " nF")
        form.addRow("Rup", self.rup_k)
        form.addRow("Rlow", self.rlow_k)
        form.addRow("Rlow 对地电容", self.cdiv_nf)
        form.addRow("运放增益", self.opamp_gain)
        form.addRow("运放带宽（0=理想）", self.opamp_bw_khz)
        form.addRow("ADC 前串联电阻", self.adc_r)
        form.addRow("ADC 输入对地电容", self.adc_c_nf)
        pl.addWidget(sense); pl.addStretch(1)
        self._add_parameter_page("sense", page)

        # ADC
        page, pl = self._page(
            "ADC / Averaging",
            "ADC 采样窗口、转换时间、多 SOC 与递归平均决定反馈采样链的数字幅相。",
        )
        adc = QGroupBox("ADC 与数字平均")
        form = QFormLayout(adc); self._configure_form(form)
        self.adc_clock_mhz = self._double(1, 1000, 3, 60.0, " MHz")
        self.acq_ns = self._double(1, 100000, 2, 300.0, " ns")
        self.conversion_cycles = self._double(1, 100, 3, 13.0, " cycles")
        self.soc_count = QSpinBox(); self.soc_count.setRange(1, 16); self.soc_count.setValue(3)
        self.previous_weight = self._double(0, 0.999, 6, 0.25)
        form.addRow("ADCCLK", self.adc_clock_mhz)
        form.addRow("Acquisition window", self.acq_ns)
        form.addRow("转换周期", self.conversion_cycles)
        form.addRow("连续 SOC 数", self.soc_count)
        form.addRow("上一平均值权重", self.previous_weight)
        pl.addWidget(adc); pl.addStretch(1)
        self._add_parameter_page("adc", page)

        layout.addWidget(self.parameter_stack, 1)
        self.run_button = QPushButton("建立 / 更新完整数字电压环")
        self.run_button.setStyleSheet("font-weight:600;")
        self.run_button.clicked.connect(self._request)
        layout.addWidget(self.run_button)
        self.codegen_button = QPushButton("生成 C99 控制代码")
        self.codegen_button.setToolTip("生成 LLC 控制器 + PCMD/FM LUT 的平台无关 C99/float32 ControlStep；不生成 BSP")
        self.codegen_button.clicked.connect(self._generate_c99)
        layout.addWidget(self.codegen_button)
        self._update_controller_fields()
        return panel

    def _add_parameter_page(self, key: str, content: QWidget, aliases: tuple[str, ...] = ()) -> None:
        index = self.parameter_stack.addWidget(self._scroll_page(content))
        self._page_for_block[key] = index
        for alias in aliases:
            self._page_for_block[alias] = index

    def _build_result_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 0, 0, 0)
        layout.setSpacing(6)

        top = QHBoxLayout()
        top.addWidget(QLabel("Bode 视图"))
        self.plot_group = QComboBox()
        self.plot_group.addItem("开环稳定性（默认）", "open")
        self.plot_group.addItem("功率级与 FM", "plant")
        self.plot_group.addItem("数字控制器", "controller")
        self.plot_group.addItem("模拟与 ADC 采样", "sense")
        self.plot_group.addItem("开环延迟包络", "delay")
        self.plot_group.addItem("完整链路分解", "overview")
        self.plot_group.addItem("闭环 / 灵敏度 / 输出阻抗", "closed")
        self.plot_group.currentIndexChanged.connect(self.refresh)
        top.addWidget(self.plot_group)
        default_hint = QLabel("默认只看系统开环，减少无关曲线干扰")
        default_hint.setStyleSheet(f"color:{theme.active_theme().text_muted};")
        top.addWidget(default_hint)
        top.addStretch(1)
        layout.addLayout(top)

        self.result_tabs = QTabWidget()
        self.result_tabs.setDocumentMode(True)

        bode_page = QWidget()
        bode_layout = QVBoxLayout(bode_page)
        bode_layout.setContentsMargins(0, 0, 0, 0)
        bode_layout.setSpacing(5)
        status_row = QHBoxLayout()
        self.cursor_status = QLabel("光标：在 Bode 幅频/相频图内单击并拖动")
        self.cursor_status.setWordWrap(True)
        _t = theme.active_theme()
        self.cursor_status.setStyleSheet(
            f"QLabel {{padding:5px 7px;border:1px solid {_t.border_card};border-radius:5px;"
            f"background:{_t.card_bg};color:{_t.text};}}"
        )
        self.phase_budget_label = QLabel("Phase budget：运行后显示交越频率各环节 Gain / Phase")
        self.phase_budget_label.setWordWrap(True)
        self.phase_budget_label.setStyleSheet(
            f"QLabel {{padding:5px 7px;border:1px solid {_t.border_card};border-radius:5px;"
            f"background:{_t.card_bg_alt};color:{_t.text};}}"
        )
        status_row.addWidget(self.cursor_status, 1)
        status_row.addWidget(self.phase_budget_label, 1)
        bode_layout.addLayout(status_row)

        self.figure = Figure(figsize=(11, 7))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(430)
        bode_layout.addWidget(self.canvas, 1)
        self.result_tabs.addTab(bode_page, "Bode / 稳定性")

        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        fixed = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        fixed.setPointSize(10)
        self.text.setFont(fixed)
        self.result_tabs.addTab(self.text, "详细结果 / 差分方程")
        layout.addWidget(self.result_tabs, 1)
        return panel

    def _parameter_selector_changed(self, index: int) -> None:
        key = self.parameter_selector.itemData(index)
        if key:
            self._show_parameter_page(str(key), update_diagram=True)

    def _show_parameter_page(self, key: str, *, update_diagram: bool = False) -> None:
        canonical = self._canonical_block.get(key, key)
        index = self._page_for_block.get(key, self._page_for_block.get(canonical))
        if index is not None:
            self.parameter_stack.setCurrentIndex(index)
        for combo_index in range(self.parameter_selector.count()):
            if self.parameter_selector.itemData(combo_index) == canonical:
                if self.parameter_selector.currentIndex() != combo_index:
                    self.parameter_selector.blockSignals(True)
                    self.parameter_selector.setCurrentIndex(combo_index)
                    self.parameter_selector.blockSignals(False)
                break
        self._update_detail_diagram(canonical)
        if update_diagram:
            self.diagram.select_block(key if self.diagram.has_block(key) else canonical, emit=False)

    def _diagram_selected(self, key: str) -> None:
        self._show_parameter_page(key, update_diagram=False)
        mapping = {
            "sum": "controller",
            "controller": "controller",
            "clamp": "controller",
            "fm": "plant",
            "pwm": "delay",
            "plant": "plant",
            "sense": "sense",
            "adc": "sense",
        }
        target = mapping.get(key)
        if target is not None:
            for index in range(self.plot_group.count()):
                if self.plot_group.itemData(index) == target:
                    self.plot_group.setCurrentIndex(index)
                    break
        self.result_tabs.setCurrentIndex(0)

    def _update_controller_fields(self) -> None:
        kind = self.controller_kind.currentData()
        pi_enabled = kind in (ControllerKind.PI, ControllerKind.PIF)
        self.kp.setEnabled(pi_enabled)
        self.ti_ms.setEnabled(pi_enabled)
        self.pif_fc.setEnabled(kind == ControllerKind.PIF)
        for widget in (self.b0, self.b1, self.b2, self.a1, self.a2):
            widget.setEnabled(kind == ControllerKind.TWO_P_TWO_Z)

    def set_nominal_work_point(self, vbus_v: float) -> None:
        self.vbus.setValue(vbus_v)

    def set_busy(self, busy: bool) -> None:
        self.run_button.setEnabled(not busy)
        if hasattr(self, "codegen_button"):
            self.codegen_button.setEnabled(not busy)

    def _controller_config(self, sample_time_s: float):
        kind = self.controller_kind.currentData()
        if self.out_max.value() <= self.out_min.value():
            raise ValueError("控制器输出上限必须大于下限")
        common = {
            "sample_time_s": sample_time_s,
            "output_min": self.out_min.value(),
            "output_max": self.out_max.value(),
        }
        if kind == ControllerKind.PI:
            return PIControllerConfig(
                kp=self.kp.value(), ti_s=self.ti_ms.value() * 1e-3, **common)
        if kind == ControllerKind.PIF:
            return PIFControllerConfig(
                kp=self.kp.value(), ti_s=self.ti_ms.value() * 1e-3,
                lpf_cutoff_hz=self.pif_fc.value(), **common)
        return TwoP2ZControllerConfig(
            b0=self.b0.value(), b1=self.b1.value(), b2=self.b2.value(),
            a1=self.a1.value(), a2=self.a2.value(), **common)

    def _generate_c99(self) -> None:
        if self.result is None:
            QMessageBox.information(self, "C99 代码生成", "请先建立 / 更新完整数字电压环。")
            return
        directory = QFileDialog.getExistingDirectory(self, "选择 LLC C99 输出目录")
        if not directory:
            return
        try:
            result = generate_llc_control_code(self.result, Path(directory) / "llc_control_generated")
        except Exception as exc:
            QMessageBox.warning(self, "C99 代码生成失败", str(exc))
            return
        QMessageBox.information(
            self, "C99 代码生成完成",
            f"已生成：{result.directory}\n\n输出 PCMD / Fsw / TBPRD 等语义控制量，不生成 ePWM/ADC BSP。",
        )

    def _request(self) -> None:
        try:
            sample_time_s = self.sample_us.value() * 1e-6
            lut = FrequencyModulatorLUT.from_text(
                self.lut_text.toPlainText(),
                mode=self.fm_mode.currentData(),
                timer_clock_hz=self.timer_mhz.value() * 1e6,
                count_mode=self.count_mode.currentData(),
            )
            controller = self._controller_config(sample_time_s)
            analog = AnalogSenseConfig(
                rup_ohm=self.rup_k.value() * 1e3,
                rlow_ohm=self.rlow_k.value() * 1e3,
                divider_capacitance_f=self.cdiv_nf.value() * 1e-9,
                opamp_gain=self.opamp_gain.value(),
                opamp_bandwidth_hz=self.opamp_bw_khz.value() * 1e3,
                adc_series_resistance_ohm=self.adc_r.value(),
                adc_shunt_capacitance_f=self.adc_c_nf.value() * 1e-9,
                normalize_to_engineering_units=True,
            )
            adc = ADCSamplingConfig(
                control_sample_time_s=sample_time_s,
                adc_clock_hz=self.adc_clock_mhz.value() * 1e6,
                acquisition_time_s=self.acq_ns.value() * 1e-9,
                conversion_cycles=self.conversion_cycles.value(),
                soc_count=self.soc_count.value(),
                recursive_previous_weight=self.previous_weight.value(),
            )
            timing = CommandTimingConfig(
                computation_delay_s=self.computation_us.value() * 1e-6,
                include_zero_order_hold=self.include_zoh.isChecked(),
            )
            self.analysis_requested.emit({
                "small_signal": {
                    "vbus_v": self.vbus.value(),
                    "load_fraction": self.load_percent.value() / 100.0,
                    "sample_time_s": sample_time_s,
                    "control_input_kind": ControlInputKind.FREQUENCY_HZ,
                    "timer_clock_hz": self.timer_mhz.value() * 1e6,
                    "input_delay_samples": 0,
                },
                "loop": {
                    "controller_config": controller,
                    "fm_lut": lut,
                    "command_pu": None if self.auto_pcmd.isChecked() else self.pcmd.value(),
                    "analog_sense": analog,
                    "adc_sampling": adc,
                    "command_timing": timing,
                },
            })
        except Exception as exc:
            QMessageBox.warning(self, "数字环路参数错误", str(exc))

    def set_analysis(self, result: DigitalLoopAnalysis) -> None:
        self.result = result
        self.pcmd.setValue(result.fm_operating_point.command_pu)
        self._cursor_frequency_hz = result.margins_nominal_delay.critical_gain_crossover_hz
        self.refresh()

    def _cursor_changed(self, measurement: BodeCursorMeasurement) -> None:
        self._cursor_frequency_hz = measurement.frequency_hz
        lines = [
            f"光标：{format_frequency(measurement.frequency_hz)}",
            *[
                f"{value.label}: {value.gain_db:+.4g} dB / {value.phase_deg:+.5g}°"
                for value in measurement.values
            ],
        ]
        self.cursor_status.setText("\n".join(lines))
        if self.result is not None:
            labels = {
                "controller": "Controller C(z)",
                "fm_power_stage": "FM × Plant",
                "sense_total": "Sense / ADC",
                "open_loop_nominal": "Total open loop",
            }
            budget = phase_budget(
                self.result.frequencies_hz, self.result.responses, labels,
                measurement.frequency_hz,
                ["controller", "fm_power_stage", "sense_total", "open_loop_nominal"],
            )
            self.phase_budget_label.setText(
                "Phase budget @ " + format_frequency(measurement.frequency_hz) + "\n" +
                " | ".join(
                    f"{b.label}: {b.gain_db:+.2f} dB / {b.phase_deg:+.2f}°"
                    for b in budget
                )
            )

    @staticmethod
    def _magnitude(response: np.ndarray) -> np.ndarray:
        return 20.0 * np.log10(np.maximum(np.abs(response), 1e-300))

    @staticmethod
    def _phase(response: np.ndarray) -> np.ndarray:
        return np.unwrap(np.angle(response)) * 180.0 / math.pi

    def refresh(self) -> None:
        if self.result is None:
            return
        result = self.result
        f = result.frequencies_hz
        group = self.plot_group.currentData()
        groups: dict[str, list[tuple[str, str, str]]] = {
            "open": [
                ("系统开环 L", "open_loop_nominal", "-"),
            ],
            "overview": [
                ("Gvf 功率级", "power_stage", "-"),
                ("Kfm × Gvf", "fm_power_stage", "--"),
                ("C(z)", "controller", ":"),
                ("完整采样链", "sense_total", "-."),
                ("系统开环 L", "open_loop_nominal", "-"),
            ],
            "plant": [
                ("Gvf: fs → Vo", "power_stage", "-"),
                ("Gpcmd: PCMD → Vo", "fm_power_stage", "--"),
            ],
            "controller": [(result.controller.name, "controller", "-")],
            "sense": [
                ("分压 / 运放 / ADC RC（原始）", "sense_analog_raw", "-"),
                ("模拟链（标定后）", "sense_analog_calibrated", "--"),
                ("ADC 多 SOC + 递归平均", "adc_sampling", ":"),
                ("完整采样反馈链", "sense_total", "-."),
            ],
            "delay": [
                ("开环：最小 Zero 等待", "open_loop_minimum", "--"),
                ("开环：标称 Zero 等待", "open_loop_nominal", "-"),
                ("开环：最大 Zero 等待", "open_loop_maximum", ":"),
            ],
            "closed": [
                ("开环 L", "open_loop_nominal", "-"),
                ("闭环 T", "closed_loop_nominal", "--"),
                ("灵敏度 S", "sensitivity_nominal", ":"),
                ("闭环输出阻抗", "closed_loop_output_impedance", "-."),
            ],
        }
        if self._bode_cursor is not None:
            self._bode_cursor.disconnect()
            self._bode_cursor = None
        self.figure.clear()
        ax_mag = self.figure.add_subplot(211)
        ax_phase = self.figure.add_subplot(212, sharex=ax_mag)
        cursor_traces: list[BodeCursorTrace] = []
        for label, key, linestyle in groups[str(group)]:
            response = result.responses[key]
            gain_db = self._magnitude(response)
            phase_deg = self._phase(response)
            magnitude_line, = ax_mag.semilogx(
                f, gain_db, linestyle=linestyle, label=label)
            ax_phase.semilogx(
                f, phase_deg, linestyle=linestyle, label=label,
                color=magnitude_line.get_color())
            cursor_traces.append(BodeCursorTrace(
                label=label,
                frequencies_hz=f,
                gain_db=gain_db,
                phase_deg=phase_deg,
                color=magnitude_line.get_color(),
            ))
        ax_mag.axhline(0.0, linewidth=0.8, linestyle="--")
        margin = result.margins_nominal_delay
        if group in ("open", "overview", "delay", "closed") and margin.critical_gain_crossover_hz:
            ax_mag.axvline(margin.critical_gain_crossover_hz, linewidth=0.9, linestyle=":")
            ax_phase.axvline(margin.critical_gain_crossover_hz, linewidth=0.9, linestyle=":")
        ax_mag.set_ylabel("Magnitude (dB)")
        ax_phase.set_ylabel("Phase (deg)")
        ax_phase.set_xlabel("Perturbation frequency (Hz)")
        for axis in (ax_mag, ax_phase):
            axis.grid(True, which="both", alpha=0.28)
            axis.legend(fontsize=8, ncol=2, loc="best")
        self.figure.tight_layout(pad=0.8)
        initial_frequency = self._cursor_frequency_hz
        if initial_frequency is None and margin.critical_gain_crossover_hz is not None:
            initial_frequency = margin.critical_gain_crossover_hz
        self._bode_cursor = BodeCursor(
            self.canvas,
            ax_mag,
            ax_phase,
            cursor_traces,
            initial_frequency_hz=initial_frequency,
            on_changed=self._cursor_changed,
        )
        self.canvas.draw_idle()
        self._update_text()

    @staticmethod
    def _format_margin(label: str, margin) -> list[str]:
        crossover = "—" if margin.critical_gain_crossover_hz is None else f"{margin.critical_gain_crossover_hz:.6g} Hz"
        pm = "—" if margin.phase_margin_deg is None else f"{margin.phase_margin_deg:.5g} deg"
        gm = "—" if margin.gain_margin_db is None else f"{margin.gain_margin_db:.5g} dB"
        delay = "—" if margin.delay_margin_s is None else f"{margin.delay_margin_s*1e6:.5g} µs"
        return [f"{label}: fc={crossover}, PM={pm}, GM={gm}, delay margin={delay}"]

    def _update_text(self) -> None:
        if self.result is None:
            return
        r = self.result
        fm = r.fm_operating_point
        analog = r.analog_sense
        adc = r.adc_sampling
        discrete = r.discrete_approximation
        poles = ", ".join(
            f"{p.real:.5g}{p.imag:+.5g}j |z|={abs(p):.5g}"
            for p in discrete.closed_loop_poles
        ) or "—"
        text = [
            "LLC 完整数字电压环",
            "=" * 88,
            f"工作点: Vbus={r.small_signal.operating_point.vbus_v:.4g} V, "
            f"Load={r.small_signal.operating_point.load_fraction*100:.3g}%, "
            f"fs={r.small_signal.operating_point.switching_frequency_hz/1e3:.7g} kHz",
            f"控制器: {r.controller.name}",
            f"C(z) numerator={np.array2string(r.controller.numerator, precision=9, separator=', ')}",
            f"C(z) denominator={np.array2string(r.controller.denominator, precision=9, separator=', ')}",
            f"差分方程: {r.controller.difference_equation(precision=9)}",
            "",
            "FM 调制器",
            "-" * 88,
            f"PCMD={fm.command_pu:.8g}, TBPRD={fm.tbprd_counts:.8g}, LUT fs={fm.frequency_hz/1e3:.8g} kHz",
            f"Kfm={fm.gain_hz_per_pu:.9g} Hz/pu",
            f"Kfm(left/right)={fm.left_gain_hz_per_pu:.9g} / {fm.right_gain_hz_per_pu:.9g} Hz/pu",
            f"PCMD headroom low/high={fm.command_headroom_low:.6g} / {fm.command_headroom_high:.6g}",
            "",
            "模拟与 ADC 链",
            "-" * 88,
            f"Divider DC gain={analog.divider_gain:.10g}; calibration gain={analog.effective_calibration_gain:.10g}",
            f"Rup||Rlow={analog.divider_thevenin_ohm:.8g} Ω; divider pole={analog.divider_pole_hz/1e3:.8g} kHz",
            f"ADC RC pole={analog.adc_rc_pole_hz/1e3:.8g} kHz",
            f"SOC sample offsets={np.array2string(adc.sample_offsets_s*1e6, precision=6)} µs",
            f"SOC4 EOC delay≈{adc.eoc_delay_s*1e6:.7g} µs; recursive filter weight={adc.recursive_previous_weight:.6g}",
            "",
            "稳定性",
            "-" * 88,
        ]
        text += self._format_margin("最小 PWM-Zero 等待", r.margins_minimum_delay)
        text += self._format_margin("标称 PWM-Zero 等待", r.margins_nominal_delay)
        text += self._format_margin("最大 PWM-Zero 等待", r.margins_maximum_delay)
        text += [
            f"离散近似延迟={discrete.integer_delay_samples}+{discrete.fractional_delay_samples:.6g} sample",
            f"离散闭环极点稳定={discrete.stable}",
            f"离散闭环极点: {poles}",
            f"综合结论: {'LIKELY STABLE' if r.likely_stable else 'CHECK / UNSTABLE'}",
            "",
            "模型边界与警告",
            "-" * 88,
        ]
        text.extend(f"- {warning}" for warning in r.warnings)
        self.text.setPlainText("\n".join(text))


__all__ = ["DigitalLoopView"]
