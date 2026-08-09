"""PySide6 PFC double-loop, sensing and waveform workbench."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from llc_design.control.digital_loop import (
    ControllerKind,
    PIControllerConfig,
    PIFControllerConfig,
    TwoP2ZControllerConfig,
)
from llc_design.gui.widgets.control_block_diagram import BlockSpec, ConnectionSpec, ControlBlockDiagram
from llc_design.gui.widgets.sense_schematic import AnalogSenseSchematic
from llc_design.control.phase_budget import phase_budget
from llc_design.gui.widgets.bode_cursor import (
    BodeCursorMeasurement,
    format_frequency,
)

from pfc_design.control import (
    ADCTimingConfig,
    DigitalFilterConfig,
    ExternalSenseConfig,
    LoadModel,
    PFCControlLabAnalysis,
    PFCControlLabConfig,
    PFCFirmwareAlgorithmConfig,
    PFCLineCycleWaveforms,
    tune_pfc_current_loop,
    PFCPowerStageConfig,
    PFCSwitchingWaveforms,
)
from pfc_design.control.waveforms import PFC_PWM_STATE_NAMES

from .bode_panel import PFCBodeCurve, SelectableBodePanel
from power_codegen import generate_ttpl_control_code


class PFCControlLabView(QWidget):
    """PFC current inner-loop, bus-voltage outer-loop and waveform workspace."""

    analysis_requested = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.result: tuple[
            PFCControlLabAnalysis,
            PFCLineCycleWaveforms,
            PFCSwitchingWaveforms,
        ] | None = None
        self._cursor_frequency_hz: float | None = None
        self._syncing_cursor = False

        root = QVBoxLayout(self)
        header = QHBoxLayout()
        title = QLabel("PFC Control Lab — 电流内环 / 母线电压外环 / 采样链 / 波形")
        title.setStyleSheet("font-size: 18px; font-weight: 600; padding: 3px;")
        header.addWidget(title)
        header.addStretch(1)
        self.run_button = QPushButton("运行 PFC 完整分析")
        self.run_button.clicked.connect(self._request)
        header.addWidget(self.run_button)
        self.codegen_button = QPushButton("生成 C99 控制代码")
        self.codegen_button.setToolTip("基于当前已验证环路生成平台无关 C99/float32 ControlStep + ISR 模板；不生成 BSP")
        self.codegen_button.clicked.connect(self._generate_c99)
        header.addWidget(self.codegen_button)
        root.addLayout(header)

        diagram_bar = QHBoxLayout()
        diagram_title = QLabel("TTPL PFC 双环控制信号链")
        diagram_title.setStyleSheet("font-size:15px;font-weight:600;color:#344054;")
        diagram_bar.addWidget(diagram_title)
        diagram_hint = QLabel("点击模块联动参数/Bode · 滚轮缩放 · 拖动平移 · 双击空白适应窗口")
        diagram_hint.setStyleSheet("color:#667085;")
        diagram_bar.addWidget(diagram_hint)
        diagram_bar.addStretch(1)
        self.fit_diagram_button = QPushButton("适应")
        self.actual_diagram_button = QPushButton("100%")
        self.zoom_out_button = QPushButton("−"); self.zoom_out_button.setFixedWidth(34)
        self.zoom_in_button = QPushButton("＋"); self.zoom_in_button.setFixedWidth(34)
        self.fullscreen_diagram_button = QPushButton("全屏框图")
        for button in (self.fit_diagram_button, self.actual_diagram_button, self.zoom_out_button, self.zoom_in_button, self.fullscreen_diagram_button):
            diagram_bar.addWidget(button)
        root.addLayout(diagram_bar)

        # Two clearly separated rows: slow voltage path above, fast current path
        # below.  Feed-forward and sensing are placed underneath their summing
        # points so every connection is orthogonal and the two feedback returns
        # do not cut through the main signal path.
        self._diagram_blocks = [
            BlockSpec("vsum", "Σ Vbus", "Vref − Vbus", 20, 18, 105, 62),
            BlockSpec("voltage_controller", "Voltage PI / PIF / 2P2Z", "Cv(z) · 10 kHz", 155, 18, 190, 62, "controller"),
            BlockSpec("vff", "Vrms² Feedforward", "V-loop normalization", 375, 18, 175, 62, "modulator"),
            BlockSpec("amc", "AMC / Gcmd", "conductance · 25 kHz", 580, 18, 160, 62, "modulator"),
            BlockSpec("iref", "Iref Generator", "Gcmd × |Vac|", 770, 18, 165, 62, "modulator"),

            BlockSpec("isum", "Σ Current", "Iref − |iL|", 770, 118, 130, 62),
            BlockSpec("current_controller", "Current PI / PIF / 2P2Z", "Ci(z) · 50 kHz", 930, 118, 200, 62, "controller"),
            BlockSpec("indu_comp", "Current Gain", "Kindu = 0.7 … 1", 1160, 118, 160, 62, "modulator"),
            BlockSpec("duty", "Σ Duty", "Dff + ΔD", 1350, 118, 125, 62),
            BlockSpec("pwm", "PWM / Min Pulse", "ZOH + update delay", 1505, 118, 175, 62, "modulator"),
            BlockSpec("plant", "TTPL Boost Plant", "Gid(s) / bus energy", 1710, 118, 185, 62, "plant"),

            BlockSpec("vac_sense", "Vac Sense / ADC", "Iref / DFF / Vrms", 545, 225, 175, 62, "sense"),
            BlockSpec("duty_ff", "Duty Feedforward", "1 − |Vac| / Vbus_set", 1340, 225, 195, 62, "modulator"),
            BlockSpec("current_sense", "iL Sense / ADC", "Sensor + RC + ADC", 1695, 225, 190, 62, "sense"),
            BlockSpec("vbus_sense", "Vbus Sense / ADC", "Divider + RC + ADC", 1695, 325, 190, 62, "sense"),
        ]
        self._diagram_connections = [
            ConnectionSpec("vsum", "voltage_controller"),
            ConnectionSpec("voltage_controller", "vff"),
            ConnectionSpec("vff", "amc"),
            ConnectionSpec("amc", "iref", "Gcmd"),
            ConnectionSpec("iref", "isum", "Iref"),
            ConnectionSpec("isum", "current_controller", "ei"),
            ConnectionSpec("current_controller", "indu_comp", "Duty PI"),
            ConnectionSpec("indu_comp", "duty", "ΔD"),
            ConnectionSpec("duty_ff", "duty", "Dff"),
            ConnectionSpec("duty", "pwm"),
            ConnectionSpec("pwm", "plant"),
            ConnectionSpec("plant", "current_sense", "iL"),
            ConnectionSpec("current_sense", "isum", "iL meas", feedback=True),
            ConnectionSpec("plant", "vbus_sense", "Vbus"),
            ConnectionSpec("vbus_sense", "vsum", "Vbus meas", feedback=True),
            ConnectionSpec("vac_sense", "iref", "|Vac|"),
            ConnectionSpec("vac_sense", "duty_ff", "|Vac|"),
            ConnectionSpec("vac_sense", "vff", "Vrms"),
        ]
        self.diagram = ControlBlockDiagram()
        self.diagram.setMinimumHeight(220)
        self.diagram.setMaximumHeight(285)
        self.diagram.set_diagram(self._diagram_blocks, self._diagram_connections)
        self.diagram.block_selected.connect(self._diagram_selected)
        root.addWidget(self.diagram)
        self.fit_diagram_button.clicked.connect(self.diagram.fit_to_view)
        self.actual_diagram_button.clicked.connect(self.diagram.actual_size)
        self.zoom_out_button.clicked.connect(self.diagram.zoom_out)
        self.zoom_in_button.clicked.connect(self.diagram.zoom_in)
        self.fullscreen_diagram_button.clicked.connect(self._open_diagram_fullscreen)

        splitter = QSplitter()
        splitter.addWidget(self._build_input_panel())
        splitter.addWidget(self._build_result_panel())
        splitter.setSizes([400, 1180])
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, 1)

    def _open_diagram_fullscreen(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("TTPL PFC 双环控制信号链 — 全屏")
        dialog.setModal(True)
        layout = QVBoxLayout(dialog)
        bar = QHBoxLayout()
        title = QLabel("TTPL PFC：电压外环 + 电流内环 + Duty Feedforward")
        title.setStyleSheet("font-size:18px;font-weight:700;color:#101828;")
        bar.addWidget(title); bar.addStretch(1)
        big = ControlBlockDiagram(dialog)
        buttons = []
        for text, slot in (("适应窗口", big.fit_to_view), ("100%", big.actual_size), ("−", big.zoom_out), ("＋", big.zoom_in)):
            b = QPushButton(text); b.clicked.connect(slot); bar.addWidget(b); buttons.append(b)
        close = QPushButton("关闭"); close.clicked.connect(dialog.accept); bar.addWidget(close)
        layout.addLayout(bar)
        big.set_diagram(list(self._diagram_blocks), list(self._diagram_connections))
        if self.diagram.selected_key and big.has_block(self.diagram.selected_key):
            big.select_block(self.diagram.selected_key)
        big.block_selected.connect(lambda key: (self.diagram.select_block(key, emit=False), self._diagram_selected(key)))
        layout.addWidget(big, 1)
        dialog.resize(1600, 900)
        dialog.setWindowState(dialog.windowState() | Qt.WindowState.WindowMaximized)
        dialog.exec()

    @staticmethod
    def _spin(minimum, maximum, decimals, value, suffix="") -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setDecimals(decimals)
        widget.setValue(value)
        widget.setSuffix(suffix)
        widget.setKeyboardTracking(False)
        return widget

    @staticmethod
    def _integer_spin(minimum: int, maximum: int, value: int) -> QSpinBox:
        widget = QSpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        widget.setKeyboardTracking(False)
        return widget

    def _build_input_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        self.input_tabs = QTabWidget()
        self.input_tabs.addTab(self._build_power_stage_tab(), "功率级/波形")
        self.input_tabs.addTab(self._build_controller_tab(), "双环控制器")
        self.input_tabs.addTab(self._build_sensing_tab(), "外置滤波/ADC")
        layout.addWidget(self.input_tabs)

        note = QLabel(
            "Bode 默认只显示系统开环 Li/Lv。每个 Bode 页面上方可独立开启或关闭任意传递函数。"
        )
        note.setWordWrap(True)
        note.setStyleSheet("padding: 6px; border: 1px solid #aaa; background: #f7f7f7;")
        layout.addWidget(note)
        layout.addStretch(1)
        scroll.setWidget(content)
        scroll.setMinimumWidth(385)
        return scroll

    def _build_power_stage_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        group = QGroupBox("单相 TTPL CCM PFC")
        form = QFormLayout(group)
        self.vin_rms = self._spin(20, 400, 2, 230, " V")
        self.line_hz = self._spin(40, 70, 2, 50, " Hz")
        self.vbus = self._spin(100, 1000, 2, 400, " V")
        self.pout = self._spin(10, 100000, 1, 3300, " W")
        self.fsw = self._spin(5, 500, 2, 50, " kHz")
        self.inductance = self._spin(1, 10000, 2, 220, " µH")
        self.dcr = self._spin(0, 5000, 3, 55, " mΩ")
        self.cbus = self._spin(1, 100000, 1, 1320, " µF")
        self.cbus_esr = self._spin(0, 1000, 3, 35, " mΩ")
        self.angle = self._spin(1, 179, 1, 60, "°")
        self.duty_min = self._spin(0, 0.9, 5, 0.01)
        self.duty_max = self._spin(0.01, 1.0, 5, 0.98)
        self.min_pulse_us = self._spin(0, 100, 4, 0, " µs")
        self.deadtime_ns = self._spin(0, 5000, 2, 100, " ns")
        self.switch_cycles = self._integer_spin(1, 20, 2)
        self.switch_samples = self._integer_spin(100, 5000, 800)
        self.load_model = QComboBox()
        self.load_model.addItem("恒功率 LLC 负载", LoadModel.CONSTANT_POWER)
        self.load_model.addItem("电阻负载", LoadModel.RESISTIVE)
        for label, widget in (
            ("输入 RMS", self.vin_rms),
            ("电网频率", self.line_hz),
            ("母线电压", self.vbus),
            ("输出功率", self.pout),
            ("开关频率", self.fsw),
            ("Boost 电感", self.inductance),
            ("电感/通路等效电阻", self.dcr),
            ("母线电容", self.cbus),
            ("母线电容 ESR", self.cbus_esr),
            ("电流环分析相位", self.angle),
            ("Duty 最小", self.duty_min),
            ("Duty 最大", self.duty_max),
            ("最小有效脉宽", self.min_pulse_us),
            ("高频桥臂死区", self.deadtime_ns),
            ("局部开关周期数", self.switch_cycles),
            ("每开关周期采样点", self.switch_samples),
            ("后级负载", self.load_model),
        ):
            form.addRow(label, widget)
        layout.addWidget(group)

        waveform_note = QGroupBox("波形范围")
        note_layout = QVBoxLayout(waveform_note)
        note_layout.addWidget(QLabel(
            "• AC 周期页始终显示稳态仿真的最后一个完整 AC 周期。\n"
            "• 开关周期页显示上面设定的局部 PWM 周期数。\n"
            "• 仿真内部保留多个 AC 周期用于控制器与母线状态收敛。"
        ))
        layout.addWidget(waveform_note)
        layout.addStretch(1)
        return page

    def _controller_group(self, title: str, *, voltage: bool):
        group = QGroupBox(title)
        form = QFormLayout(group)
        kind = QComboBox()
        kind.addItem("PI", ControllerKind.PI)
        kind.addItem("PIF（PI + 输出 LPF）", ControllerKind.PIF)
        kind.addItem("2P2Z", ControllerKind.TWO_P_TWO_Z)
        # The current-loop GUI starts from the conservative V7.1.5 design
        # baseline instead of the low-margin legacy firmware values.  The
        # original firmware pair remains available through the restore button
        # below, and the one-click tuner recomputes a recommendation whenever
        # L/R/sensing/delay settings change.
        kp = self._spin(0, 1000, 8, 0.1 if voltage else 0.00854059)
        ti = self._spin(0.001, 10000, 5, 2.0 if voltage else 0.64608, " ms")
        fc = self._spin(0.1, 100000, 2, 1000 if voltage else 5000, " Hz")
        b0 = self._spin(-1e6, 1e6, 9, 0)
        b1 = self._spin(-1e6, 1e6, 9, 0)
        b2 = self._spin(-1e6, 1e6, 9, 0)
        a1 = self._spin(-10, 10, 9, 0)
        a2 = self._spin(-10, 10, 9, 0)
        for label, widget in (
            ("类型", kind), ("Kp", kp), ("Ti", ti), ("PIF 截止", fc),
            ("b0", b0), ("b1", b1), ("b2", b2),
            ("a1_den", a1), ("a2_den", a2),
        ):
            form.addRow(label, widget)

        def update() -> None:
            current = kind.currentData()
            is_pi = current in (ControllerKind.PI, ControllerKind.PIF)
            kp.setEnabled(is_pi)
            ti.setEnabled(is_pi)
            fc.setEnabled(current == ControllerKind.PIF)
            for item in (b0, b1, b2, a1, a2):
                item.setEnabled(current == ControllerKind.TWO_P_TWO_Z)

        kind.currentIndexChanged.connect(update)
        update()
        return group, {
            "kind": kind, "kp": kp, "ti": ti, "fc": fc,
            "b0": b0, "b1": b1, "b2": b2, "a1": a1, "a2": a2,
        }

    def _build_controller_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        current_group, self.current_ctrl = self._controller_group(
            "电感电流内环（50 kHz）", voltage=False)
        voltage_group, self.voltage_ctrl = self._controller_group(
            "母线电压外环（10 kHz）", voltage=True)
        layout.addWidget(current_group)

        tune_group = QGroupBox("电流环稳定设计起点")
        tune_layout = QVBoxLayout(tune_group)
        tune_buttons = QHBoxLayout()
        self.autotune_current_button = QPushButton("一键稳定整定并应用")
        self.autotune_current_button.setToolTip("按当前 Boost L/R、采样链、50 kHz 数字延迟与 indu_comp 包络自动设计 PI")
        self.restore_firmware_current_button = QPushButton("恢复固件原始 PI")
        self.restore_firmware_current_button.setToolTip("恢复 Kp=0.01, Ti=0.075 ms；该组参数保留用于对照，不代表当前模型下有足够相位裕量")
        tune_buttons.addWidget(self.autotune_current_button)
        tune_buttons.addWidget(self.restore_firmware_current_button)
        tune_layout.addLayout(tune_buttons)
        self.autotune_status = QLabel(
            "默认使用稳定设计基线。修改 L/R、采样滤波或延迟后，建议重新执行一键整定。"
        )
        self.autotune_status.setWordWrap(True)
        self.autotune_status.setStyleSheet("padding:6px;border:1px solid #b2ddff;background:#eff8ff;color:#175cd3;")
        tune_layout.addWidget(self.autotune_status)
        self.autotune_current_button.clicked.connect(self._auto_tune_current_loop)
        self.restore_firmware_current_button.clicked.connect(self._restore_firmware_current_pi)
        layout.addWidget(tune_group)
        layout.addWidget(voltage_group)

        fw_group = QGroupBox("AMC / 前馈 / 调度")
        form = QFormLayout(fw_group)
        self.vff_gain = self._spin(1e-6, 10, 8, 0.01)
        self.gcmd_max = self._spin(0.001, 10, 5, 0.18, " A/V")
        self.indu_gain = self._spin(0, 10, 5, 0.085)
        self.current_delay_us = self._spin(0, 100, 3, 11, " µs")
        self.amc_delay_us = self._spin(0, 100, 3, 20, " µs")
        self.voltage_delay_us = self._spin(0, 100, 3, 4, " µs")
        for label, widget in (
            ("Vrms 前馈 K", self.vff_gain),
            ("gcmd 上限", self.gcmd_max),
            ("indu_comp 斜率", self.indu_gain),
            ("电流控制+PWM 延迟", self.current_delay_us),
            ("AMC 更新延迟", self.amc_delay_us),
            ("电压计算延迟", self.voltage_delay_us),
        ):
            form.addRow(label, widget)
        layout.addWidget(fw_group)
        layout.addStretch(1)
        return page

    def _sense_group(self, title: str, defaults: tuple[float, ...], *, source_label: str, front_label: str):
        gain, bw_khz, source_r, source_c_nf, out_r, out_c_nf, sample_khz = defaults
        group = QGroupBox(title)
        form = QFormLayout(group)
        schematic = AnalogSenseSchematic(title=f"{title}：硬件与数字采样链")
        schematic.set_labels(
            source=source_label,
            front=front_label,
            amp="Buffer / OpAmp\nGain + GBW",
            rc="External RC\nRout / Cadc",
            adc="ADC S/H\nFilter / Scale",
        )
        form.addRow(schematic)
        widgets = {
            "gain": self._spin(1e-9, 1000, 9, gain, " V/unit"),
            "bw": self._spin(0.1, 100000, 2, bw_khz, " kHz"),
            "source_r": self._spin(0, 1e7, 3, source_r, " Ω"),
            "source_c": self._spin(0, 1e6, 4, source_c_nf, " nF"),
            "out_r": self._spin(0, 1e7, 3, out_r, " Ω"),
            "out_c": self._spin(0, 1e6, 4, out_c_nf, " nF"),
            "sample": self._spin(0.1, 1000, 3, sample_khz, " kHz"),
            "alpha": self._spin(0.000001, 1, 6, 1.0),
        }
        for label, key in (
            ("前端增益", "gain"), ("运放带宽", "bw"),
            ("前级等效 R", "source_r"), ("前级对地 C", "source_c"),
            ("ADC 串联 R", "out_r"), ("ADC 对地 C", "out_c"),
            ("采样率", "sample"), ("数字 LPF α", "alpha"),
        ):
            form.addRow(label, widgets[key])
        return group, widgets

    def _build_sensing_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        current, self.current_sense = self._sense_group(
            "电感电流采样", (0.03, 2000, 0, 0, 220, 2, 50),
            source_label="iL", front_label="Current Sensor\nShunt / Hall")
        vac, self.vac_sense = self._sense_group(
            "AC 电压采样", (1 / 150, 1000, 2000, 1, 220, 2, 50),
            source_label="Vac", front_label="HV Divider\nRup / Rlow / C")
        vbus_gain = 1600 / (117000 + 1600)
        vbus_r = 117000 * 1600 / (117000 + 1600)
        vbus, self.vbus_sense = self._sense_group(
            "母线电压采样", (vbus_gain, 1000, vbus_r, 1, 220, 2, 10),
            source_label="Vbus", front_label="117k / 1.6k\n+ 1 nF")
        for group in (current, vac, vbus):
            layout.addWidget(group)
        layout.addStretch(1)
        return page

    def _build_result_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.cursor_status = QLabel(
            "Bode 光标：运行分析后，在幅频或相频图中单击、拖动。"
        )
        self.cursor_status.setWordWrap(True)
        self.cursor_status.setStyleSheet(
            "QLabel {padding:5px;border:1px solid #aaa;background:#f7f7f7;}"
        )
        layout.addWidget(self.cursor_status)
        self.phase_budget_label = QLabel("Phase budget：点击控制框图环节或拖动 Bode 光标查看该频点的幅相贡献。")
        self.phase_budget_label.setWordWrap(True)
        self.phase_budget_label.setStyleSheet("QLabel {padding:5px;border:1px solid #cfd4dc;background:#fbfcfe;}")
        layout.addWidget(self.phase_budget_label)

        self.tabs = QTabWidget()
        self.current_bode = SelectableBodePanel("PFC 电感电流内环")
        self.voltage_bode = SelectableBodePanel("PFC 母线电压外环")
        self.sense_bode = SelectableBodePanel("PFC 三路外部滤波与 ADC")
        for panel_widget in (self.current_bode, self.voltage_bode, self.sense_bode):
            panel_widget.cursor_changed.connect(self._cursor_changed)
        self.tabs.addTab(self.current_bode, "电流环 Bode")
        self.tabs.addTab(self.voltage_bode, "电压环 Bode")
        self.tabs.addTab(self.sense_bode, "采样链 Bode")

        self.ac_overview_figure, self.ac_overview_canvas = self._waveform_tab(
            "完整 AC 周期")
        self.ac_control_figure, self.ac_control_canvas = self._waveform_tab(
            "AC 控制细节")
        self.switch_figure, self.switch_canvas = self._waveform_tab(
            "局部开关周期")
        self.zero_figure, self.zero_canvas = self._waveform_tab(
            "Zero Crossing Analyzer")
        self.harmonic_figure, self.harmonic_canvas = self._waveform_tab(
            "PF / THD / Harmonics")

        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        fixed = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        fixed.setPointSize(10)
        self.summary.setFont(fixed)
        self.tabs.addTab(self.summary, "分析摘要")
        layout.addWidget(self.tabs, 1)
        return panel

    def _waveform_tab(self, title: str):
        page = QWidget()
        page_layout = QVBoxLayout(page)
        hint = QLabel(title)
        hint.setStyleSheet("font-weight: 600; padding: 3px;")
        figure = Figure(figsize=(11, 8))
        canvas = FigureCanvasQTAgg(figure)
        page_layout.addWidget(hint)
        page_layout.addWidget(canvas, 1)
        self.tabs.addTab(page, title)
        return figure, canvas

    def _diagram_selected(self, key: str) -> None:
        # Parameter panel and Bode trace use the same semantic key.
        if key in {"voltage_controller", "current_controller", "vff", "amc", "iref", "duty_ff", "indu_comp", "duty"}:
            self.input_tabs.setCurrentIndex(1)
        elif key in {"current_sense", "vac_sense", "vbus_sense"}:
            self.input_tabs.setCurrentIndex(2)
        else:
            self.input_tabs.setCurrentIndex(0)

        if key == "current_controller":
            self.tabs.setCurrentWidget(self.current_bode); self.current_bode.focus_curve("controller_ci")
        elif key == "current_sense":
            self.tabs.setCurrentWidget(self.current_bode); self.current_bode.focus_curve("sense_hi")
        elif key == "indu_comp":
            self.tabs.setCurrentWidget(self.current_bode); self.current_bode.focus_curve("indu_comp_gain")
        elif key == "pwm":
            self.tabs.setCurrentWidget(self.current_bode); self.current_bode.focus_curve("pwm_zoh")
        elif key == "plant":
            self.tabs.setCurrentWidget(self.current_bode); self.current_bode.focus_curve("plant_gid")
        elif key == "voltage_controller":
            self.tabs.setCurrentWidget(self.voltage_bode); self.voltage_bode.focus_curve("controller_cv")
        elif key in {"vff", "amc", "iref"}:
            self.tabs.setCurrentWidget(self.voltage_bode); self.voltage_bode.focus_curve("amc_vff")
        elif key == "duty_ff":
            # Duty feed-forward is an additive disturbance-decoupling path, not
            # part of the return ratio. Keep the stability view on Li rather
            # than falsely presenting it as a loop-gain transfer function.
            self.tabs.setCurrentWidget(self.current_bode); self.current_bode.show_open_loop_only()
        elif key == "vbus_sense":
            self.tabs.setCurrentWidget(self.voltage_bode); self.voltage_bode.focus_curve("sense_hv")
        elif key == "vac_sense":
            self.tabs.setCurrentWidget(self.sense_bode); self.sense_bode.focus_curve("vac_total", keep_open_loop=False)

    def _restore_firmware_current_pi(self) -> None:
        index = self.current_ctrl["kind"].findData(ControllerKind.PI)
        if index >= 0:
            self.current_ctrl["kind"].setCurrentIndex(index)
        self.current_ctrl["kp"].setValue(0.01)
        self.current_ctrl["ti"].setValue(0.075)
        self.autotune_status.setText(
            "已恢复固件原始 PI：Kp=0.01, Ti=0.075 ms。该值用于固件对照；请运行 Bode 检查当前硬件/延迟下的实际裕量。"
        )
        self.autotune_status.setStyleSheet("padding:6px;border:1px solid #fec84b;background:#fffaeb;color:#b54708;")

    def _auto_tune_current_loop(self) -> None:
        try:
            config = self._config()
            tuned = tune_pfc_current_loop(config)
            index = self.current_ctrl["kind"].findData(ControllerKind.PI)
            if index >= 0:
                self.current_ctrl["kind"].setCurrentIndex(index)
            self.current_ctrl["kp"].setValue(tuned.controller.kp)
            self.current_ctrl["ti"].setValue(tuned.controller.ti_s * 1e3)
            worst = tuned.worst_point
            worst_text = ""
            if worst is not None:
                worst_text = (
                    f"；最差点≈{worst.vin_rms_v:.1f} Vrms / {100*worst.load_ratio:.0f}% load / "
                    f"{worst.line_angle_deg:.0f}° / Kindu={worst.indu_comp:.3f}"
                )
            self.autotune_status.setText(
                f"{tuned.message}  Kp={tuned.controller.kp:.7g}, "
                f"Ti={tuned.controller.ti_s*1e3:.5g} ms；"
                f"fc≈{(tuned.nominal_crossover_hz or 0):.1f} Hz，"
                f"PM≈{(tuned.nominal_phase_margin_deg or 0):.1f}°，"
                f"最差 PM≈{(tuned.worst_phase_margin_deg or 0):.1f}°{worst_text}"
            )
            style = (
                "padding:6px;border:1px solid #75e0a7;background:#ecfdf3;color:#067647;"
                if tuned.accepted else
                "padding:6px;border:1px solid #fec84b;background:#fffaeb;color:#b54708;"
            )
            self.autotune_status.setStyleSheet(style)
            # One-click means both apply and immediately run the full analysis,
            # so the user sees the new Bode/waveforms without a second action.
            self._request()
        except Exception as exc:
            self.autotune_status.setText(f"自动整定失败：{exc}")
            self.autotune_status.setStyleSheet("padding:6px;border:1px solid #fda29b;background:#fef3f2;color:#b42318;")

    def set_busy(self, busy: bool) -> None:
        self.run_button.setEnabled(not busy)
        # Freeze inputs while the worker is using a snapshot of them.  Without
        # this, changing line frequency/plant values mid-run can make the GUI
        # render a result using a different parameter set than the solver used.
        if hasattr(self, "input_tabs"):
            self.input_tabs.setEnabled(not busy)
        if hasattr(self, "autotune_current_button"):
            self.autotune_current_button.setEnabled(not busy)
        if hasattr(self, "codegen_button"):
            self.codegen_button.setEnabled(not busy)

    def _controller(self, fields, sample_rate, output_min, output_max):
        kind = fields["kind"].currentData()
        ts = 1.0 / sample_rate
        if kind == ControllerKind.PI:
            return PIControllerConfig(
                fields["kp"].value(), fields["ti"].value() * 1e-3,
                ts, output_min, output_max)
        if kind == ControllerKind.PIF:
            return PIFControllerConfig(
                fields["kp"].value(), fields["ti"].value() * 1e-3,
                fields["fc"].value(), ts, output_min, output_max)
        return TwoP2ZControllerConfig(
            fields["b0"].value(), fields["b1"].value(), fields["b2"].value(),
            fields["a1"].value(), fields["a2"].value(),
            ts, output_min, output_max)

    def _sense(self, title, fields, sample_rate):
        timing = ADCTimingConfig(
            sample_rate_hz=sample_rate,
            adc_clock_hz=60e6,
            acquisition_time_s=300e-9,
            conversion_cycles=13,
            # Do not double-count computation/PWM delay inside the sensor.
            # The firmware/PWM model below owns those delays explicitly.
            computation_delay_s=0.0,
            pwm_update_delay_s=0.0,
            digital_filter=DigitalFilterConfig(fields["alpha"].value()),
        )
        return ExternalSenseConfig(
            name=title,
            front_end_gain_v_per_unit=fields["gain"].value(),
            amplifier_gain=1.0,
            amplifier_bandwidth_hz=fields["bw"].value() * 1e3,
            source_resistance_ohm=fields["source_r"].value(),
            shunt_capacitance_f=fields["source_c"].value() * 1e-9,
            output_resistance_ohm=fields["out_r"].value(),
            adc_capacitance_f=fields["out_c"].value() * 1e-9,
            normalize_to_engineering_units=True,
            timing=timing,
        )

    def _config(self) -> PFCControlLabConfig:
        stage = PFCPowerStageConfig(
            vin_rms_v=self.vin_rms.value(),
            line_frequency_hz=self.line_hz.value(),
            bus_voltage_v=self.vbus.value(),
            output_power_w=self.pout.value(),
            switching_frequency_hz=self.fsw.value() * 1e3,
            boost_inductance_h=self.inductance.value() * 1e-6,
            equivalent_series_resistance_ohm=self.dcr.value() * 1e-3,
            bus_capacitance_f=self.cbus.value() * 1e-6,
            bus_cap_esr_ohm=self.cbus_esr.value() * 1e-3,
            load_model=self.load_model.currentData(),
            line_angle_deg=self.angle.value(),
            duty_min=self.duty_min.value(),
            duty_max=self.duty_max.value(),
            minimum_effective_pulse_s=self.min_pulse_us.value() * 1e-6,
            deadtime_s=self.deadtime_ns.value() * 1e-9,
        )
        fw = PFCFirmwareAlgorithmConfig(
            vac_rms_feedforward_gain=self.vff_gain.value(),
            gcmd_max_a_per_v=self.gcmd_max.value(),
            indu_comp_gain=self.indu_gain.value(),
            current_computation_delay_s=self.current_delay_us.value() * 1e-6,
            current_pwm_update_delay_s=0.0,
            amc_update_delay_s=self.amc_delay_us.value() * 1e-6,
            voltage_computation_delay_s=self.voltage_delay_us.value() * 1e-6,
        )
        return PFCControlLabConfig(
            power_stage=stage,
            firmware=fw,
            current_controller=self._controller(
                self.current_ctrl, 50e3, -2.0, 0.98),
            voltage_controller=self._controller(
                self.voltage_ctrl, 10e3, -1.0, 40.0),
            current_sense=self._sense(
                "PFC inductor current", self.current_sense,
                self.current_sense["sample"].value() * 1e3),
            vac_sense=self._sense(
                "AC input voltage", self.vac_sense,
                self.vac_sense["sample"].value() * 1e3),
            vbus_sense=self._sense(
                "PFC bus voltage", self.vbus_sense,
                self.vbus_sense["sample"].value() * 1e3),
            switching_cycles=self.switch_cycles.value(),
            switching_samples_per_cycle=self.switch_samples.value(),
        )

    def _generate_c99(self) -> None:
        if self.result is None:
            QMessageBox.information(self, "C99 代码生成", "请先运行 PFC 完整分析；建议先执行一键稳定整定。")
            return
        directory = QFileDialog.getExistingDirectory(self, "选择 TTPL C99 输出目录")
        if not directory:
            return
        try:
            analysis = self.result[0]
            result = generate_ttpl_control_code(analysis, Path(directory) / "ttpl_control_generated")
        except Exception as exc:
            QMessageBox.warning(self, "C99 代码生成失败", str(exc))
            return
        QMessageBox.information(
            self, "C99 代码生成完成",
            f"已生成：{result.directory}\n\n仅包含控制算法 / ControlStep / ISR 模板，不包含 ADC、PWM、GPIO 或中断 BSP 配置。",
        )

    def _request(self) -> None:
        try:
            config = self._config()
            config.validate()
            self.analysis_requested.emit(config)
        except Exception as exc:
            QMessageBox.warning(self, "PFC Control Lab 参数错误", str(exc))

    def set_result(self, result) -> None:
        self.result = result
        analysis, line_cycle, switching = result
        pm = analysis.current_loop.margins.phase_margin_deg
        fc = analysis.current_loop.margins.critical_gain_crossover_hz
        if pm is not None and pm < 45.0:
            self.autotune_status.setText(
                f"当前电流环裕量不足：PM={pm:.1f}°, fc={0 if fc is None else fc:.1f} Hz。建议点击“一键稳定整定并应用”。"
            )
            self.autotune_status.setStyleSheet("padding:6px;border:1px solid #fda29b;background:#fef3f2;color:#b42318;")
        else:
            # A Bode margin alone is not enough for a useful starting point.
            # Confirm that the settled AC cycle and the local switching
            # reconstruction remain finite/bounded before advertising the
            # current controller as a usable baseline.
            metrics = line_cycle.metrics
            current_scale = max(metrics.input_current_rms_a, 1e-6)
            tracking_ratio = metrics.current_error_rms_a / current_scale
            sw_current = np.asarray(switching.signals.get("inductor_current", []), dtype=float)
            time_ok = (
                sw_current.size > 0
                and np.all(np.isfinite(sw_current))
                and np.all(np.isfinite(line_cycle.signals["i_inductor"]))
                and tracking_ratio < 0.35
            )
            if pm is not None and pm >= 45.0 and time_ok:
                self.autotune_status.setText(
                    f"稳定性检查通过：PM={pm:.1f}°, fc={0 if fc is None else fc:.1f} Hz；"
                    f"AC 电流误差 RMS={metrics.current_error_rms_a:.3f} A "
                    f"({100*tracking_ratio:.1f}% Irms)，开关电流连续且有界。可在此基础上继续调优。"
                )
                self.autotune_status.setStyleSheet("padding:6px;border:1px solid #75e0a7;background:#ecfdf3;color:#067647;")
            elif pm is not None and pm >= 45.0:
                self.autotune_status.setText(
                    f"线性 Bode 已稳定（PM={pm:.1f}°），但时域跟踪仍需检查："
                    f"current error={100*tracking_ratio:.1f}% Irms。"
                )
                self.autotune_status.setStyleSheet("padding:6px;border:1px solid #fec84b;background:#fffaeb;color:#b54708;")
        self._cursor_frequency_hz = (
            analysis.current_loop.margins.critical_gain_crossover_hz
        )
        self._set_bode_results(analysis)
        self._plot_ac_overview(line_cycle, analysis)
        self._plot_ac_control(line_cycle, analysis)
        self._plot_switching(switching)
        self._plot_zero_crossing(line_cycle, analysis)
        self._plot_harmonics(line_cycle)
        self._show_summary(analysis, line_cycle, switching)

    def _set_bode_results(self, analysis: PFCControlLabAnalysis) -> None:
        f = analysis.frequencies_hz
        ci = analysis.current_loop.responses
        self.current_bode.set_curves(f, [
            PFCBodeCurve("plant_gid", "Gid 功率级", ci["plant_gid"]),
            PFCBodeCurve("controller_ci", "Ci 数字控制器", ci["controller_ci"]),
            PFCBodeCurve("indu_comp_gain", "Kindu 电感电流补偿", ci["indu_comp_gain"]),
            PFCBodeCurve("pwm_zoh", "PWM/ZOH/延迟", ci["pwm_zoh"]),
            PFCBodeCurve("sense_hi_analog", "Hi 模拟滤波", ci["sense_hi_analog"]),
            PFCBodeCurve("sense_hi_adc", "Hi ADC/数字链", ci["sense_hi_adc"]),
            PFCBodeCurve("sense_hi", "Hi 完整采样链", ci["sense_hi"]),
            PFCBodeCurve("forward_current", "电流环前向通道", ci["forward_current"]),
            PFCBodeCurve(
                "open_current", "Li 电流系统开环", ci["open_current"],
                default_visible=True, is_open_loop=True),
            PFCBodeCurve(
                "closed_current_measured", "Ti 电流闭环（测量）",
                ci["closed_current_measured"]),
            PFCBodeCurve(
                "closed_current_actual", "Ti 电流闭环（实际）",
                ci["closed_current_actual"]),
            PFCBodeCurve(
                "sensitivity_current", "Si 电流灵敏度",
                ci["sensitivity_current"]),
        ], margins=analysis.current_loop.margins, open_loop_key="open_current")

        cv = analysis.voltage_loop.responses
        self.voltage_bode.set_curves(f, [
            PFCBodeCurve(
                "current_closed_for_outer", "已闭合电流内环",
                cv["current_closed_for_outer"]),
            PFCBodeCurve("amc_vff", "AMC/Vrms 前馈", cv["amc_vff"]),
            PFCBodeCurve("bus_plant_gvg", "母线功率级 Gvg", cv["bus_plant_gvg"]),
            PFCBodeCurve("controller_cv", "Cv 数字控制器", cv["controller_cv"]),
            PFCBodeCurve("sense_hv_analog", "Hv 模拟滤波", cv["sense_hv_analog"]),
            PFCBodeCurve("sense_hv_adc", "Hv ADC/数字链", cv["sense_hv_adc"]),
            PFCBodeCurve("sense_hv", "Hv 完整采样链", cv["sense_hv"]),
            PFCBodeCurve("forward_voltage", "电压环前向通道", cv["forward_voltage"]),
            PFCBodeCurve(
                "open_voltage", "Lv 电压系统开环", cv["open_voltage"],
                default_visible=True, is_open_loop=True),
            PFCBodeCurve("closed_voltage", "Tv 电压闭环", cv["closed_voltage"]),
            PFCBodeCurve(
                "sensitivity_voltage", "Sv 电压灵敏度",
                cv["sensitivity_voltage"]),
        ], margins=analysis.voltage_loop.margins, open_loop_key="open_voltage")

        self.sense_bode.set_curves(f, [
            PFCBodeCurve(
                "current_total", "电流采样总链",
                analysis.current_sense_response.total,
                default_visible=True),
            PFCBodeCurve(
                "current_analog", "电流采样模拟部分",
                analysis.current_sense_response.calibrated_analog),
            PFCBodeCurve(
                "vac_total", "Vac 采样总链",
                analysis.vac_sense_response.total,
                default_visible=True),
            PFCBodeCurve(
                "vac_analog", "Vac 采样模拟部分",
                analysis.vac_sense_response.calibrated_analog),
            PFCBodeCurve(
                "vbus_total", "Vbus 采样总链",
                analysis.vbus_sense_response.total,
                default_visible=True),
            PFCBodeCurve(
                "vbus_analog", "Vbus 采样模拟部分",
                analysis.vbus_sense_response.calibrated_analog),
        ])

        for panel in (self.current_bode, self.voltage_bode, self.sense_bode):
            panel.set_cursor_frequency(self._cursor_frequency_hz)

    def _cursor_changed(self, measurement: BodeCursorMeasurement) -> None:
        if self._syncing_cursor:
            return
        self._syncing_cursor = True
        try:
            self._cursor_frequency_hz = measurement.frequency_hz
            sender = self.sender()
            for panel in (self.current_bode, self.voltage_bode, self.sense_bode):
                if panel is not sender:
                    panel.set_cursor_frequency(measurement.frequency_hz)
            self.cursor_status.setText("\n".join([
                f"光标频率：{format_frequency(measurement.frequency_hz)}",
                *[
                    f"{value.label}: Gain={value.gain_db:+.6g} dB, "
                    f"Phase={value.phase_deg:+.7g}°"
                    for value in measurement.values
                ],
            ]))
            if self.result is not None:
                analysis = self.result[0]
                if sender is self.current_bode:
                    responses = analysis.current_loop.responses
                    labels = {"controller_ci":"Ci", "indu_comp_gain":"Kindu", "pwm_zoh":"PWM/ZOH", "plant_gid":"Gid", "sense_hi":"Current Sense", "open_current":"Li"}
                    keys = ["controller_ci","indu_comp_gain","pwm_zoh","plant_gid","sense_hi","open_current"]
                elif sender is self.voltage_bode:
                    responses = analysis.voltage_loop.responses
                    labels = {"controller_cv":"Cv", "amc_vff":"AMC/VFF", "current_closed_for_outer":"Closed current loop", "bus_plant_gvg":"Bus plant", "sense_hv":"Vbus Sense", "open_voltage":"Lv"}
                    keys = ["controller_cv","amc_vff","current_closed_for_outer","bus_plant_gvg","sense_hv","open_voltage"]
                else:
                    responses, labels, keys = {}, {}, []
                if keys:
                    budget = phase_budget(analysis.frequencies_hz, responses, labels, measurement.frequency_hz, keys)
                    self.phase_budget_label.setText(
                        "Phase budget @ " + format_frequency(measurement.frequency_hz) + "\n" +
                        " | ".join(f"{b.label}: {b.gain_db:+.2f} dB / {b.phase_deg:+.2f}°" for b in budget)
                    )
        finally:
            self._syncing_cursor = False

    @staticmethod
    def _last_ac_cycle(result: PFCLineCycleWaveforms, line_hz: float):
        dt = float(np.mean(np.diff(result.time_s)))
        count = max(int(round(1.0 / line_hz / dt)), 2)
        start = max(len(result.time_s) - count, 0)
        selection = slice(start, len(result.time_s))
        time_ms = (result.time_s[selection] - result.time_s[start]) * 1e3
        signals = {key: np.asarray(value)[selection] for key, value in result.signals.items()}
        return time_ms, signals

    @staticmethod
    def _style_axes(axes, xlabel: str) -> None:
        for axis in axes:
            axis.grid(True, alpha=0.3)
            handles, labels = axis.get_legend_handles_labels()
            if handles:
                axis.legend(fontsize=8, ncol=min(4, len(handles)), loc="best")
        axes[-1].set_xlabel(xlabel)

    def _plot_ac_overview(
        self,
        result: PFCLineCycleWaveforms,
        analysis: PFCControlLabAnalysis,
    ) -> None:
        figure = self.ac_overview_figure
        figure.clear()
        axes = [figure.add_subplot(611)]
        for index in range(2, 7):
            axes.append(figure.add_subplot(610 + index, sharex=axes[0]))
        time_ms, s = self._last_ac_cycle(
            result, analysis.config.power_stage.line_frequency_hz)

        axes[0].plot(time_ms, s["vac"], label="Vac actual")
        axes[0].plot(time_ms, s["vac_measured"], label="Vac measured")
        axes[0].set_ylabel("Vac (V)")

        axes[1].plot(time_ms, s["i_input_signed"], label="Iin actual")
        axes[1].plot(time_ms, s["i_ref"], label="Iref")
        axes[1].plot(time_ms, s["i_measured_signed"], label="I measured")
        axes[1].set_ylabel("Current (A)")

        axes[2].plot(time_ms, s["current_error"], label="Current error")
        axes[2].axhline(0.0, linewidth=0.8)
        axes[2].set_ylabel("Ei (A)")

        axes[3].plot(time_ms, s["vbus"], label="Vbus actual")
        axes[3].plot(time_ms, s["vbus_measured"], label="Vbus measured")
        axes[3].axhline(
            analysis.config.power_stage.bus_voltage_v,
            linestyle="--", linewidth=0.8, label="Vbus command")
        axes[3].set_ylabel("Vbus (V)")

        axes[4].plot(time_ms, s["input_power"], label="Input power")
        axes[4].axhline(
            analysis.config.power_stage.output_power_w,
            linestyle="--", linewidth=0.8, label="Load power")
        axes[4].set_ylabel("Power (W)")

        axes[5].plot(time_ms, s["bus_cap_current"], label="Bus capacitor current")
        axes[5].plot(time_ms, s["boost_output_current"], label="Boost output current")
        axes[5].plot(time_ms, s["load_current"], label="Load current")
        axes[5].set_ylabel("Bus I (A)")

        self._style_axes(axes, "Time in final AC period (ms)")
        metrics = result.metrics
        figure.suptitle(
            "PFC 完整一个 AC 周期 — "
            f"PF={metrics.power_factor:.5f}, THD={metrics.current_thd_percent:.4f}%"
        )
        figure.tight_layout()
        self.ac_overview_canvas.draw_idle()

    def _plot_ac_control(
        self,
        result: PFCLineCycleWaveforms,
        analysis: PFCControlLabAnalysis,
    ) -> None:
        figure = self.ac_control_figure
        figure.clear()
        axes = [figure.add_subplot(611)]
        for index in range(2, 7):
            axes.append(figure.add_subplot(610 + index, sharex=axes[0]))
        time_ms, s = self._last_ac_cycle(
            result, analysis.config.power_stage.line_frequency_hz)

        axes[0].plot(time_ms, s["vac_rms_estimate"], label="Vac RMS estimate")
        axes[0].axhline(
            analysis.config.power_stage.vin_rms_v,
            linestyle="--", linewidth=0.8, label="Vin RMS nominal")
        axes[0].set_ylabel("Vrms (V)")

        axes[1].plot(time_ms, s["vloop"], label="Voltage-loop output")
        axes[1].plot(time_ms, s["voltage_error"], label="Vbus error")
        axes[1].set_ylabel("V-loop")

        axes[2].plot(time_ms, s["gcmd"], label="gcmd AMC")
        axes[2].set_ylabel("gcmd (A/V)")

        axes[3].plot(time_ms, s["duty_ff"], label="Duty FF")
        axes[3].plot(time_ms, s["duty_pi"], label="Duty PI")
        axes[3].plot(time_ms, s["duty_total"], label="Duty total")
        axes[3].plot(time_ms, s["effective_duty_min"], label="Effective duty min")
        axes[3].set_ylabel("Duty (pu)")

        axes[4].plot(time_ms, s["vbus_ripple"], label="Vbus ripple from command")
        axes[4].plot(time_ms, s["minimum_pulse_active"], label="Min-pulse active")
        axes[4].set_ylabel("Ripple/logic")

        axes[5].plot(time_ms, s["current_update_strobe"], label="Current 50 kHz")
        axes[5].plot(time_ms, s["amc_update_strobe"], label="AMC 25 kHz")
        axes[5].plot(time_ms, s["voltage_update_strobe"], label="Voltage 10 kHz")
        axes[5].set_ylabel("Update")

        self._style_axes(axes, "Time in final AC period (ms)")
        figure.suptitle("PFC 一个 AC 周期内的双环控制量与多速率更新")
        figure.tight_layout()
        self.ac_control_canvas.draw_idle()

    def _plot_switching(self, result: PFCSwitchingWaveforms) -> None:
        figure = self.switch_figure
        figure.clear()
        axes = [figure.add_subplot(511)]
        for index in range(2, 6):
            axes.append(figure.add_subplot(510 + index, sharex=axes[0]))
        time_us = result.time_s * 1e6
        s = result.signals

        axes[0].plot(time_us, s["hf_high_gate"], label="HF high gate")
        axes[0].plot(time_us, s["hf_low_gate"], label="HF low gate")
        axes[0].plot(time_us, s["lf_polarity_gate"], label="LF polarity")
        axes[0].set_ylabel("Gate")

        axes[1].plot(time_us, s["switch_node_voltage"], label="Switch node")
        axes[1].plot(time_us, s["inductor_voltage"], label="Inductor voltage")
        axes[1].set_ylabel("Voltage (V)")

        axes[2].plot(time_us, s["inductor_current"], label="Inductor current (continuous)")
        if "inductor_current_average" in s:
            axes[2].plot(time_us, s["inductor_current_average"], linestyle="--", label="AC-workpoint average")
        axes[2].plot(time_us, s["boost_output_current"], label="Boost output current")
        axes[2].set_ylabel("Current (A)")

        axes[3].plot(time_us, s["high_side_current"], label="HF high current")
        axes[3].plot(time_us, s["low_side_current"], label="HF low current")
        axes[3].set_ylabel("Switch I (A)")

        axes[4].plot(time_us, s["bus_cap_current"], label="Bus capacitor current")
        axes[4].plot(time_us, s["duty_command"], label="Duty command")
        axes[4].set_ylabel("Bus I / Duty")

        period_us = 1e6 / result.switching_frequency_hz
        cycles = int(round(result.time_s[-1] * result.switching_frequency_hz)) + 1
        for boundary in range(1, cycles):
            x = boundary * period_us
            for axis in axes:
                axis.axvline(x, linestyle=":", linewidth=0.7)
        self._style_axes(axes, "Time (µs)")
        figure.suptitle(
            f"PFC 局部开关周期波形 — AC phase={result.line_angle_deg:.2f}°, "
            f"fs={result.switching_frequency_hz / 1e3:.5g} kHz"
        )
        figure.tight_layout()
        self.switch_canvas.draw_idle()

    def _plot_zero_crossing(self, result: PFCLineCycleWaveforms, analysis: PFCControlLabAnalysis) -> None:
        figure = self.zero_figure
        figure.clear()
        axes = [figure.add_subplot(511)]
        for index in range(2, 6):
            axes.append(figure.add_subplot(510 + index, sharex=axes[0]))
        time_ms, s = self._last_ac_cycle(result, analysis.config.power_stage.line_frequency_hz)
        # Show both zero crossings in the final line cycle, with state-code overlays.
        axes[0].plot(time_ms, s["vac"], label="Vac")
        axes[0].plot(time_ms, s["vac_measured"], label="Vac measured")
        axes[0].axhline(15.0, linestyle=":", linewidth=0.8); axes[0].axhline(-15.0, linestyle=":", linewidth=0.8)
        axes[0].set_ylabel("Vac (V)")
        axes[1].plot(time_ms, s["i_input_signed"], label="Iac")
        axes[1].plot(time_ms, s["i_ref"], label="Iref")
        axes[1].set_ylabel("Current (A)")
        axes[2].plot(time_ms, s["duty_ff"], label="Duty FF")
        axes[2].plot(time_ms, s["duty_pi"], label="Duty PI")
        axes[2].plot(time_ms, s["duty_total"], label="Duty total")
        axes[2].set_ylabel("Duty")
        axes[3].plot(time_ms, s["lf_gate_state"], label="LF bridge state")
        axes[3].plot(time_ms, s["zc_deadband_fraction"], label="ZC deadband fraction")
        axes[3].plot(time_ms, s["current_pi_reset_strobe"], label="PI reset")
        axes[3].set_ylabel("State")
        axes[4].step(time_ms, s["pwm_state_code"], where="post", label="PWM state code")
        axes[4].plot(time_ms, s["minimum_pulse_active"], label="Min pulse active")
        axes[4].set_ylabel("State code")
        self._style_axes(axes, "Time in final AC period (ms)")
        legend = ", ".join(f"{k}={v}" for k,v in PFC_PWM_STATE_NAMES.items())
        figure.suptitle("TTPL Zero-Crossing Analyzer — " + legend, fontsize=9)
        figure.tight_layout(); self.zero_canvas.draw_idle()

    def _plot_harmonics(self, result: PFCLineCycleWaveforms) -> None:
        figure = self.harmonic_figure
        figure.clear()
        ax = figure.add_subplot(111)
        m = result.metrics
        orders = np.asarray(m.harmonic_orders)
        amps = np.asarray(m.harmonic_current_rms_a)
        ax.bar(orders, amps)
        ax.set_xlabel("Harmonic order")
        ax.set_ylabel("Current RMS (A)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_title(
            f"PF={m.power_factor:.6f}, displacement={m.displacement_factor:.6f}, "
            f"distortion={m.distortion_factor:.6f}, THD={m.current_thd_percent:.4f}%"
        )
        figure.tight_layout(); self.harmonic_canvas.draw_idle()

    def _show_summary(
        self,
        analysis: PFCControlLabAnalysis,
        line: PFCLineCycleWaveforms,
        switching: PFCSwitchingWaveforms,
    ) -> None:
        ci = analysis.current_loop.margins
        cv = analysis.voltage_loop.margins
        m = line.metrics
        text = [
            "PFC Control Lab 双环摘要",
            "=" * 78,
            f"工作点: Vin={analysis.config.power_stage.vin_rms_v:.3f} Vrms, "
            f"Vbus={analysis.config.power_stage.bus_voltage_v:.3f} V, "
            f"P={analysis.config.power_stage.output_power_w:.3f} W",
            f"电流环相位: {analysis.operating_point.line_angle_deg:.2f}°, "
            f"Iref={analysis.operating_point.current_reference_a:.6f} A, "
            f"indu_comp={analysis.operating_point.indu_comp:.6f}",
            "",
            "Bode 默认显示策略:",
            "- 电流环页面：默认仅 Li 系统开环",
            "- 电压环页面：默认仅 Lv 系统开环",
            "- 闭环、灵敏度、控制器、功率级和采样分量由复选框按需显示",
            "",
            f"电流环: fc={ci.critical_gain_crossover_hz}, PM={ci.phase_margin_deg}, "
            f"GM={ci.gain_margin_db}, stable={analysis.current_loop.likely_stable}",
            f"电压环: fc={cv.critical_gain_crossover_hz}, PM={cv.phase_margin_deg}, "
            f"GM={cv.gain_margin_db}, stable={analysis.voltage_loop.likely_stable}",
            "",
            f"PF={m.power_factor:.7f}, displacement={m.displacement_factor:.7f}, "
            f"distortion={m.distortion_factor:.7f}, THD={m.current_thd_percent:.5f}%",
            f"I1 RMS={m.fundamental_current_rms_a:.6f} A, min-pulse occupancy={m.minimum_pulse_fraction*100:.4f}%",
            f"Zero-cross current-error RMS={m.zero_cross_current_error_rms_a:.6f} A",
            f"Vbus avg={m.bus_voltage_average_v:.5f} V, "
            f"ripple pp={m.bus_voltage_ripple_pp_v:.5f} V",
            f"Icap rms={m.bus_capacitor_current_rms_a:.5f} A, "
            f"current error rms={m.current_error_rms_a:.5f} A",
            f"AC 波形显示：最后 1 个完整 AC 周期；内部仿真 "
            f"{analysis.config.waveform_line_cycles} 周期用于收敛",
            f"局部开关波形：{analysis.config.switching_cycles} 周期，"
            f"每周期 {analysis.config.switching_samples_per_cycle} 点",
            "",
            "注意：半波同步观察器未纳入本模块；Vac 直接来自实际采样链。",
        ]
        if analysis.warnings or line.warnings:
            text += ["", "警告:"] + [
                f"- {warning}" for warning in (*analysis.warnings, *line.warnings)
            ]
        self.summary.setPlainText("\n".join(text))


__all__ = ["PFCControlLabView"]
