"""Three-phase Vienna PFC full design/control workspace (V7)."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QCheckBox,
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
from llc_design.control.phase_budget import phase_budget
from llc_design.gui.widgets.bode_cursor import BodeCursorMeasurement, format_frequency
from llc_design.gui.widgets.control_block_diagram import (
    BlockSpec,
    ConnectionSpec,
    ControlBlockDiagram,
)
from llc_design.gui.widgets.sense_schematic import AnalogSenseSchematic
from llc_design.gui import theme
from pfc_design.control.config import (
    ADCTimingConfig,
    DigitalFilterConfig,
    ExternalSenseConfig,
    LoadModel,
)
from power_codegen import generate_vienna_control_code
from pfc_design.vienna import (
    ViennaControlLabAnalysis,
    ViennaControlLabConfig,
    ViennaFirmwareConfig,
    ViennaLineCycleWaveforms,
    ViennaPowerStageConfig,
    ViennaSwitchingWaveforms,
)
from .bode_panel import PFCBodeCurve, SelectableBodePanel
from .inductor_design import PFCInductorDesignEditor, PFCInductorDesignResultView


class ViennaControlLabView(QWidget):
    """Interactive Vienna design page.

    Main control remains a DC-voltage outer loop plus three ABC stationary-frame
    current loops.  Split-bus midpoint balance is deliberately shown as an
    auxiliary path, not as a third main PFC loop.
    """

    analysis_requested = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.result = None
        self._syncing_cursor = False

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(7)

        # Same compact interaction bar as the LLC/TTPL workspaces.  The old
        # Vienna page used a separate 18 px title row plus a second diagram
        # toolbar, wasting vertical space and making Windows look inconsistent.
        diagram_bar = QHBoxLayout()
        diagram_bar.setSpacing(6)
        diagram_title = QLabel("Vienna PFC 双环 / 中点平衡")
        diagram_title.setStyleSheet(f"font-size:15px;font-weight:600;color:{theme.active_theme().text};")
        diagram_bar.addWidget(diagram_title)
        diagram_hint = QLabel("Vdc 外环 + ABC 电流内环 + Midpoint Balance · 点击模块联动参数/Bode")
        diagram_hint.setStyleSheet(f"color:{theme.active_theme().text_muted};")
        diagram_bar.addWidget(diagram_hint)
        diagram_bar.addStretch(1)

        self.run_button = QPushButton("运行分析")
        self.run_button.clicked.connect(self._request)
        self.codegen_button = QPushButton("生成 C99")
        self.codegen_button.setToolTip("生成 Vienna C99/float32 ControlStep + ISR 模板；不生成 BSP")
        self.codegen_button.clicked.connect(self._generate_c99)
        diagram_bar.addWidget(self.run_button)
        diagram_bar.addWidget(self.codegen_button)

        self.fit_diagram_button = QPushButton("适应")
        self.actual_diagram_button = QPushButton("100%")
        self.zoom_out_button = QPushButton("−"); self.zoom_out_button.setFixedWidth(34)
        self.zoom_in_button = QPushButton("＋"); self.zoom_in_button.setFixedWidth(34)
        self.fullscreen_diagram_button = QPushButton("全屏")
        self.diagram_toggle = QPushButton("隐藏框图")
        self.diagram_toggle.setCheckable(True); self.diagram_toggle.setChecked(True)
        self.inspector_toggle = QPushButton("隐藏参数")
        self.inspector_toggle.setCheckable(True); self.inspector_toggle.setChecked(True)
        for button in (
            self.fit_diagram_button, self.actual_diagram_button, self.zoom_out_button,
            self.zoom_in_button, self.fullscreen_diagram_button, self.diagram_toggle,
            self.inspector_toggle,
        ):
            diagram_bar.addWidget(button)
        root.addLayout(diagram_bar)

        # One clean main row plus support/sensing rows.  The previous V7 layout
        # mixed feed-forward, sensing and balance into the forward path and made
        # several lines cross.  This layout intentionally mirrors the LLC/TTPL
        # vector-diagram interaction model.
        self._diagram_blocks = [
            BlockSpec("vsum", "Σ Vdc", "Vdc* − Vdc", 20, 0, 86, 52),
            BlockSpec("voltage_controller", "Voltage PI / PIF / 2P2Z", "Cv(z) · 10 kHz", 126, 0, 154, 52, "controller"),
            BlockSpec("gcmd", "Gcmd", "conductance", 300, 0, 110, 52, "modulator"),
            BlockSpec("references", "3φ Current Ref", "Gcmd × Vabc", 430, 0, 146, 52, "modulator"),

            BlockSpec("current_controller", "ABC Current PI/PIF/2P2Z", "CiA/B/C · 50 kHz", 430, 64, 174, 52, "controller"),
            BlockSpec("feedforward", "Vabc + R/L FF", "voltage-drop FF", 624, 64, 138, 52, "modulator"),
            BlockSpec("modulator", "Vienna Modulator", "THI / D0 / min pulse", 782, 64, 150, 52, "modulator"),
            BlockSpec("plant", "Vienna Plant", "Iabc / Vdc±", 952, 64, 138, 52, "plant"),

            BlockSpec("bus_sense", "Vdc± Sense / ADC", "bus + midpoint", 20, 128, 146, 52, "sense"),
            BlockSpec("phase_voltage_sense", "Vabc Sense / ADC", "reference / FF", 260, 128, 150, 52, "sense"),
            BlockSpec("current_sense", "Iabc Sense / ADC", "current feedback", 624, 128, 150, 52, "sense"),
            BlockSpec("balance", "Midpoint Balance", "ΔVdc → Cbal(z)", 794, 128, 150, 52, "aux"),
        ]
        self._diagram_connections = [
            ConnectionSpec("vsum", "voltage_controller"),
            ConnectionSpec("voltage_controller", "gcmd"),
            ConnectionSpec("gcmd", "references", "Gcmd"),
            ConnectionSpec("references", "current_controller", "Iabc*"),
            ConnectionSpec("current_controller", "modulator", "Δmabc"),
            ConnectionSpec("modulator", "plant", "mabc"),
            ConnectionSpec("phase_voltage_sense", "references", "Vabc"),
            ConnectionSpec("phase_voltage_sense", "feedforward", "Vabc"),
            ConnectionSpec("references", "feedforward", "Iabc*"),
            ConnectionSpec("feedforward", "modulator", "FF"),
            ConnectionSpec("current_sense", "current_controller", "Iabc meas", True),
            ConnectionSpec("bus_sense", "vsum", "Vdc meas", True),
            ConnectionSpec("bus_sense", "balance", "ΔVdc"),
            ConnectionSpec("balance", "modulator", "balance"),
        ]
        self.diagram = ControlBlockDiagram()
        self.diagram.setMinimumHeight(150)
        self.diagram.setMaximumHeight(195)
        self.diagram.set_diagram(self._diagram_blocks, self._diagram_connections)
        self.diagram.block_selected.connect(self._diagram_selected)
        root.addWidget(self.diagram)
        self.fit_diagram_button.clicked.connect(self.diagram.fit_to_view)
        self.actual_diagram_button.clicked.connect(self.diagram.actual_size)
        self.zoom_out_button.clicked.connect(self.diagram.zoom_out)
        self.zoom_in_button.clicked.connect(self.diagram.zoom_in)
        self.fullscreen_diagram_button.clicked.connect(self._open_diagram_fullscreen)
        self.diagram_toggle.toggled.connect(self._toggle_diagram)

        self.splitter = QSplitter()
        self.splitter.setChildrenCollapsible(True)
        self.splitter.setHandleWidth(5)
        self.splitter.addWidget(self._build_inputs())
        self.splitter.addWidget(self._build_results())
        self.splitter.setSizes([345, 1200])
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        root.addWidget(self.splitter, 1)
        self.inspector_toggle.toggled.connect(self._toggle_inspector)
        self.input_tabs.currentChanged.connect(self._input_tab_changed)

    def _toggle_diagram(self, visible: bool) -> None:
        self.diagram.setVisible(bool(visible))
        self.diagram_toggle.setText("隐藏框图" if visible else "显示框图")
        if visible:
            self.diagram.fit_to_view()

    def _toggle_inspector(self, visible: bool) -> None:
        total = max(sum(self.splitter.sizes()), 900)
        if visible:
            width = min(345, max(300, total // 4))
            self.splitter.setSizes([width, max(total - width, 500)])
            self.inspector_toggle.setText("隐藏参数")
        else:
            self.splitter.setSizes([0, total])
            self.inspector_toggle.setText("显示参数")

    def _open_diagram_fullscreen(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Three-Phase Vienna PFC 控制信号链 — 全屏")
        dialog.setModal(True)
        layout = QVBoxLayout(dialog)
        bar = QHBoxLayout()
        title = QLabel("Vienna：Vdc 外环 + ABC 电流内环 + Midpoint Balance")
        title.setStyleSheet(f"font-size:18px;font-weight:700;color:{theme.active_theme().text_strong};")
        bar.addWidget(title); bar.addStretch(1)
        big = ControlBlockDiagram(dialog)
        for text, slot in (("适应窗口", big.fit_to_view), ("100%", big.actual_size), ("−", big.zoom_out), ("＋", big.zoom_in)):
            button = QPushButton(text); button.clicked.connect(slot); bar.addWidget(button)
        close = QPushButton("关闭"); close.clicked.connect(dialog.accept); bar.addWidget(close)
        layout.addLayout(bar)
        big.set_diagram(list(self._diagram_blocks), list(self._diagram_connections))
        if self.diagram.selected_key and big.has_block(self.diagram.selected_key):
            big.select_block(self.diagram.selected_key)
        big.block_selected.connect(
            lambda key: (self.diagram.select_block(key, emit=False), self._diagram_selected(key)))
        layout.addWidget(big, 1)
        dialog.resize(1600, 900)
        dialog.setWindowState(dialog.windowState() | Qt.WindowState.WindowMaximized)
        dialog.exec()

    @staticmethod
    def _spin(lo, hi, dec, val, suffix=""):
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setDecimals(dec)
        w.setValue(val)
        w.setSuffix(suffix)
        w.setKeyboardTracking(False)
        return w

    def _build_inputs(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)
        self.input_tabs = QTabWidget()
        self.input_tabs.addTab(self._stage_tab(), "功率级")
        self.input_tabs.addTab(self._control_tab(), "双环 / Balance")
        self.input_tabs.addTab(self._sense_tab(), "8 路采样链")
        self.inductor_editor = PFCInductorDesignEditor(
            "vienna", self._inductor_context, self._apply_inductor_design)
        self._inductor_input_index = self.input_tabs.addTab(self.inductor_editor, "电感设计")
        self.inductor_editor.design_completed.connect(self._inductor_design_ready)
        layout.addWidget(self.input_tabs)
        layout.addStretch(1)
        scroll.setWidget(container)
        scroll.setMinimumWidth(300)
        scroll.setMaximumWidth(430)
        return scroll

    def _stage_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        group = QGroupBox("Three-Phase Vienna Power Stage")
        form = QFormLayout(group)
        self.vll = self._spin(50, 1000, 2, 400, " V")
        self.line_hz = self._spin(40, 70, 2, 50, " Hz")
        self.vdc = self._spin(100, 1500, 2, 700, " V")
        self.pout = self._spin(100, 200000, 1, 10000, " W")
        self.fsw = self._spin(5, 500, 2, 65, " kHz")
        self.lphase = self._spin(1, 10000, 2, 600, " µH")
        self.rphase = self._spin(0, 5000, 2, 80, " mΩ")
        self.cplus = self._spin(1, 100000, 1, 680, " µF")
        self.cminus = self._spin(1, 100000, 1, 680, " µF")
        self.modlim = self._spin(.1, 1, 4, .96)
        self.minpulse = self._spin(0, 100, 4, 0, " µs")
        self.sw_angle = self._spin(0, 360, 2, 30, "°")
        self.mid_init = self._spin(-100, 100, 3, 4, " V")
        self.load = QComboBox()
        self.load.addItem("恒功率后级", LoadModel.CONSTANT_POWER)
        self.load.addItem("电阻负载", LoadModel.RESISTIVE)
        for label, widget in [
            ("输入线电压 RMS", self.vll), ("电网频率", self.line_hz),
            ("总 DC Bus", self.vdc), ("输出功率", self.pout),
            ("开关频率", self.fsw), ("每相 Boost L", self.lphase),
            ("每相串联 R", self.rphase), ("上母线电容", self.cplus),
            ("下母线电容", self.cminus), ("调制度上限", self.modlim),
            ("最小有效脉宽", self.minpulse), ("开关波形相位", self.sw_angle),
            ("初始中点不平衡 ΔV", self.mid_init), ("负载模型", self.load),
        ]:
            form.addRow(label, widget)
        layout.addWidget(group)
        layout.addStretch(1)
        return page

    def _controller_group(self, title, defaults):
        kp0, ti_ms0, outmin, outmax = defaults
        group = QGroupBox(title)
        form = QFormLayout(group)
        kind = QComboBox()
        kind.addItem("PI", ControllerKind.PI)
        kind.addItem("PIF", ControllerKind.PIF)
        kind.addItem("2P2Z", ControllerKind.TWO_P_TWO_Z)
        fields = {
            "kind": kind,
            "kp": self._spin(0, 1e4, 8, kp0),
            "ti": self._spin(.001, 1e5, 5, ti_ms0, " ms"),
            "fc": self._spin(.1, 1e6, 2, 3000, " Hz"),
            "b0": self._spin(-1e6, 1e6, 9, 0), "b1": self._spin(-1e6, 1e6, 9, 0),
            "b2": self._spin(-1e6, 1e6, 9, 0), "a1": self._spin(-10, 10, 9, 0),
            "a2": self._spin(-10, 10, 9, 0), "min": outmin, "max": outmax,
        }
        for label, key in [
            ("类型", "kind"), ("Kp", "kp"), ("Ti", "ti"), ("PIF Fc", "fc"),
            ("b0", "b0"), ("b1", "b1"), ("b2", "b2"),
            ("a1_den", "a1"), ("a2_den", "a2"),
        ]:
            form.addRow(label, fields[key])

        def update():
            k = kind.currentData()
            pi = k in (ControllerKind.PI, ControllerKind.PIF)
            fields["kp"].setEnabled(pi)
            fields["ti"].setEnabled(pi)
            fields["fc"].setEnabled(k == ControllerKind.PIF)
            for key in ("b0", "b1", "b2", "a1", "a2"):
                fields[key].setEnabled(k == ControllerKind.TWO_P_TWO_Z)

        kind.currentIndexChanged.connect(update)
        update()
        return group, fields

    def _control_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        gc, self.cc = self._controller_group("ABC 相电流内环 50 kHz", (.02, .25, -.45, .45))
        gv, self.vc = self._controller_group("DC 总线电压外环 10 kHz", (2e-5, 80, 0, .30))
        gb, self.bc = self._controller_group("DC Midpoint Balance 10 kHz（辅助）", (.001, 20, -.08, .08))
        layout.addWidget(gc)
        layout.addWidget(gv)
        layout.addWidget(gb)

        group = QGroupBox("多速率 / Vienna Modulator")
        form = QFormLayout(group)
        self.balance_limit = self._spin(.001, .5, 4, .08)
        self.mid_gain = self._spin(.1, 500, 3, 30, " A/pu")
        self.current_delay = self._spin(0, 100, 3, 9, " µs")
        self.third_harmonic = QCheckBox("启用 Third-Harmonic / common-mode injection")
        self.third_harmonic.setChecked(True)
        self.inductor_ff = QCheckBox("启用 R·I + L·dI/dt 电感压降前馈")
        self.inductor_ff.setChecked(True)
        form.addRow("Balance injection limit", self.balance_limit)
        form.addRow("Midpoint local plant gain", self.mid_gain)
        form.addRow("Current compute+PWM delay", self.current_delay)
        form.addRow(self.third_harmonic)
        form.addRow(self.inductor_ff)
        layout.addWidget(group)
        layout.addStretch(1)
        return page

    def _sense_group(self, title, gain, sample_khz, *, source_label, front_label):
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
        fields = {
            "gain": self._spin(1e-9, 1000, 9, gain, " V/unit"),
            "bw": self._spin(.1, 1e5, 2, 1500, " kHz"),
            "r": self._spin(0, 1e7, 2, 220, " Ω"),
            "c": self._spin(0, 1e6, 4, 2, " nF"),
            "sample": self._spin(.1, 1000, 3, sample_khz, " kHz"),
            "alpha": self._spin(.000001, 1, 6, 1),
        }
        for label, key in [
            ("前端增益", "gain"), ("运放带宽", "bw"), ("ADC 串联 R", "r"),
            ("ADC 对地 C", "c"), ("采样率", "sample"), ("数字 LPF α", "alpha"),
        ]:
            form.addRow(label, fields[key])
        return group, fields

    def _sense_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.lock_abc = QCheckBox("锁定 ABC 三相参数（硬件滤波共用；关闭后启用增益/偏置失配诊断）")
        self.lock_abc.setChecked(True)
        layout.addWidget(self.lock_abc)

        gi, self.si = self._sense_group(
            "Ia / Ib / Ic Current Sense", .03, 50,
            source_label="Ia / Ib / Ic", front_label="3 × Current Sensor")
        gv, self.sv = self._sense_group(
            "Va / Vb / Vc Voltage Sense", 1 / 180, 50,
            source_label="Va / Vb / Vc", front_label="3 × HV Divider")
        gb, self.sb = self._sense_group(
            "Vdc+ / Vdc- Split Bus Sense", 1600 / (117000 + 1600), 10,
            source_label="Vdc+ / Vdc-", front_label="2 × HV Divider")
        layout.addWidget(gi)
        layout.addWidget(gv)
        layout.addWidget(gb)

        self.mismatch_group = QGroupBox("通道失配诊断（仅在 ABC 解锁时生效）")
        form = QFormLayout(self.mismatch_group)
        self.i_gain = [self._spin(80, 120, 4, 100, "%") for _ in range(3)]
        self.i_off = [self._spin(-10, 10, 6, 0, " A") for _ in range(3)]
        self.v_gain = [self._spin(80, 120, 4, 100, "%") for _ in range(3)]
        self.v_off = [self._spin(-20, 20, 6, 0, " V") for _ in range(3)]
        self.bus_gain = [self._spin(80, 120, 4, 100, "%") for _ in range(2)]
        self.bus_off = [self._spin(-20, 20, 6, 0, " V") for _ in range(2)]
        for idx, ph in enumerate("ABC"):
            row = QWidget(); r = QHBoxLayout(row); r.setContentsMargins(0, 0, 0, 0)
            r.addWidget(QLabel("Gain")); r.addWidget(self.i_gain[idx]); r.addWidget(QLabel("Offset")); r.addWidget(self.i_off[idx])
            form.addRow(f"I{ph}", row)
        for idx, ph in enumerate("ABC"):
            row = QWidget(); r = QHBoxLayout(row); r.setContentsMargins(0, 0, 0, 0)
            r.addWidget(QLabel("Gain")); r.addWidget(self.v_gain[idx]); r.addWidget(QLabel("Offset")); r.addWidget(self.v_off[idx])
            form.addRow(f"V{ph}", row)
        for idx, name in enumerate(("Vdc+", "Vdc-")):
            row = QWidget(); r = QHBoxLayout(row); r.setContentsMargins(0, 0, 0, 0)
            r.addWidget(QLabel("Gain")); r.addWidget(self.bus_gain[idx]); r.addWidget(QLabel("Offset")); r.addWidget(self.bus_off[idx])
            form.addRow(name, row)
        self.mismatch_group.setEnabled(False)
        self.lock_abc.toggled.connect(lambda checked: self.mismatch_group.setEnabled(not checked))
        layout.addWidget(self.mismatch_group)
        layout.addStretch(1)
        return page

    def _inductor_context(self) -> dict[str, float]:
        return {
            "input_rms_v": self.vll.value(),
            "bus_voltage_v": self.vdc.value(),
            "output_power_w": self.pout.value(),
            "switching_frequency_hz": self.fsw.value() * 1e3,
            "target_inductance_uh": self.lphase.value(),
            "recommended_core_count": 5,
        }

    def _inductor_design_ready(self, result) -> None:
        self.inductor_result_view.set_result(result)
        self.tabs.setCurrentIndex(self._inductor_result_index)

    def _apply_inductor_design(self, result) -> None:
        self.lphase.setValue(result.l_full_load_peak_uh)
        self.rphase.setValue(result.rdc_hot_ohm * 1e3)
        QMessageBox.information(
            self, "Vienna 相电感参数已应用",
            f"Phase L = {result.l_full_load_peak_uh:.3f} µH\n"
            f"Phase DCR(@{result.request.copper_temperature_c:.0f}°C) = {result.rdc_hot_ohm*1e3:.3f} mΩ\n\n"
            "已写回 Vienna 功率级；建议重新运行完整分析。")

    def _input_tab_changed(self, index: int) -> None:
        if index == getattr(self, "_inductor_input_index", -1) and hasattr(self, "tabs"):
            self.tabs.setCurrentIndex(self._inductor_result_index)

    def _build_results(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        status_row = QHBoxLayout()
        status_row.setSpacing(6)
        self.cursor = QLabel("Bode 光标：主环路默认仅显示系统开环；点击框图环节可聚焦对应传递函数。")
        self.cursor.setWordWrap(True)
        _t = theme.active_theme()
        self.cursor.setStyleSheet(f"QLabel {{padding:5px 7px;border:1px solid {_t.border_card};border-radius:5px;background:{_t.card_bg};color:{_t.text};}}")
        self.budget = QLabel("Phase budget：拖动光标查看当前频率的幅相贡献。")
        self.budget.setWordWrap(True)
        self.budget.setStyleSheet(f"QLabel {{padding:5px 7px;border:1px solid {_t.border_card};border-radius:5px;background:{_t.card_bg_alt};color:{_t.text};}}")
        status_row.addWidget(self.cursor, 1); status_row.addWidget(self.budget, 2)
        layout.addLayout(status_row)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.bi = SelectableBodePanel("Vienna Phase-A Current Loop")
        self.bv = SelectableBodePanel("Vienna DC Voltage Loop")
        self.bb = SelectableBodePanel("Vienna Midpoint Balance Loop")
        self.bs = SelectableBodePanel("Vienna Sensor Chains")
        for bode in (self.bi, self.bv, self.bb, self.bs):
            bode.cursor_changed.connect(self._cursor_changed)
        self.tabs.addTab(self.bi, "Current Bode")
        self.tabs.addTab(self.bv, "Vdc Bode")
        self.tabs.addTab(self.bb, "Balance Bode")
        self.tabs.addTab(self.bs, "Sampling Bode")
        self.inductor_result_view = PFCInductorDesignResultView("Vienna PFC Phase Inductor")
        self._inductor_result_index = self.tabs.addTab(self.inductor_result_view, "电感设计")

        self.fac, self.cac = self._figtab("3φ AC Cycle")
        self.fbus, self.cbus = self._figtab("Split Bus / Control")
        self.fsw_fig, self.csw = self._figtab("Switching Workpoint")
        self.fsec, self.csec = self._figtab("Sector Analyzer")
        self.fharm, self.charm = self._figtab("PF / THD")
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.tabs.addTab(self.summary, "Summary")
        layout.addWidget(self.tabs, 1)
        return page

    def _figtab(self, title):
        page = QWidget()
        layout = QVBoxLayout(page)
        fig = Figure(figsize=(11, 8))
        canvas = FigureCanvasQTAgg(fig)
        layout.addWidget(canvas)
        self.tabs.addTab(page, title)
        return fig, canvas

    def _ctrl(self, fields, rate):
        kind = fields["kind"].currentData()
        ts = 1.0 / rate
        if kind == ControllerKind.PI:
            return PIControllerConfig(fields["kp"].value(), fields["ti"].value() * 1e-3, ts, fields["min"], fields["max"])
        if kind == ControllerKind.PIF:
            return PIFControllerConfig(fields["kp"].value(), fields["ti"].value() * 1e-3, fields["fc"].value(), ts, fields["min"], fields["max"])
        return TwoP2ZControllerConfig(
            fields["b0"].value(), fields["b1"].value(), fields["b2"].value(),
            fields["a1"].value(), fields["a2"].value(), ts, fields["min"], fields["max"])

    def _sense(self, title, fields):
        timing = ADCTimingConfig(
            sample_rate_hz=fields["sample"].value() * 1e3,
            adc_clock_hz=60e6,
            acquisition_time_s=300e-9,
            # ADC/SOC belongs to the sensing chain.  Controller computation and
            # PWM-update delays are modelled once in the firmware/PWM block;
            # keeping ADCTiming defaults here would count the same delay twice.
            computation_delay_s=0.0,
            pwm_update_delay_s=0.0,
            digital_filter=DigitalFilterConfig(fields["alpha"].value()),
        )
        return ExternalSenseConfig(
            name=title,
            front_end_gain_v_per_unit=fields["gain"].value(),
            amplifier_bandwidth_hz=fields["bw"].value() * 1e3,
            output_resistance_ohm=fields["r"].value(),
            adc_capacitance_f=fields["c"].value() * 1e-9,
            timing=timing,
        )

    @staticmethod
    def _percent_tuple(fields):
        return tuple(float(w.value()) / 100.0 for w in fields)

    @staticmethod
    def _value_tuple(fields):
        return tuple(float(w.value()) for w in fields)

    def _config(self):
        stage = ViennaPowerStageConfig(
            line_line_rms_v=self.vll.value(), line_frequency_hz=self.line_hz.value(),
            bus_voltage_v=self.vdc.value(), output_power_w=self.pout.value(),
            switching_frequency_hz=self.fsw.value() * 1e3,
            boost_inductance_h=self.lphase.value() * 1e-6,
            phase_series_resistance_ohm=self.rphase.value() * 1e-3,
            upper_bus_capacitance_f=self.cplus.value() * 1e-6,
            lower_bus_capacitance_f=self.cminus.value() * 1e-6,
            load_model=self.load.currentData(), modulation_limit=self.modlim.value(),
            minimum_effective_pulse_s=self.minpulse.value() * 1e-6,
        )
        fw = ViennaFirmwareConfig(
            current_computation_delay_s=self.current_delay.value() * 1e-6,
            pwm_update_delay_s=0.0,
            balance_injection_limit=self.balance_limit.value(),
            midpoint_current_gain_a_per_pu=self.mid_gain.value(),
            third_harmonic_injection_enabled=self.third_harmonic.isChecked(),
            inductor_voltage_drop_feedforward_enabled=self.inductor_ff.isChecked(),
        )
        locked = self.lock_abc.isChecked()
        return ViennaControlLabConfig(
            power_stage=stage,
            firmware=fw,
            current_controller=self._ctrl(self.cc, 50e3),
            voltage_controller=self._ctrl(self.vc, 10e3),
            balance_controller=self._ctrl(self.bc, 10e3),
            phase_current_sense=self._sense("Vienna current", self.si),
            phase_voltage_sense=self._sense("Vienna phase voltage", self.sv),
            split_bus_sense=self._sense("Vienna split bus", self.sb),
            switching_line_angle_deg=self.sw_angle.value(),
            initial_midpoint_imbalance_v=self.mid_init.value(),
            phase_current_gain_scale=(1.0, 1.0, 1.0) if locked else self._percent_tuple(self.i_gain),
            phase_current_offset_a=(0.0, 0.0, 0.0) if locked else self._value_tuple(self.i_off),
            phase_voltage_gain_scale=(1.0, 1.0, 1.0) if locked else self._percent_tuple(self.v_gain),
            phase_voltage_offset_v=(0.0, 0.0, 0.0) if locked else self._value_tuple(self.v_off),
            split_bus_gain_scale=(1.0, 1.0) if locked else self._percent_tuple(self.bus_gain),
            split_bus_offset_v=(0.0, 0.0) if locked else self._value_tuple(self.bus_off),
        )

    def _generate_c99(self):
        if self.result is None:
            QMessageBox.information(self, "C99 代码生成", "请先运行 Vienna 完整分析。")
            return
        directory = QFileDialog.getExistingDirectory(self, "选择 Vienna C99 输出目录")
        if not directory:
            return
        try:
            analysis = self.result[0]
            result = generate_vienna_control_code(analysis, Path(directory) / "vienna_control_generated")
        except Exception as exc:
            QMessageBox.warning(self, "C99 代码生成失败", str(exc))
            return
        QMessageBox.information(
            self, "C99 代码生成完成",
            f"已生成：{result.directory}\n\n输出 duty A/B/C 等语义控制命令，不生成 PWM/ADC/GPIO BSP。",
        )

    def _request(self):
        try:
            config = self._config()
            config.validate()
            self.analysis_requested.emit(config)
        except Exception as exc:
            QMessageBox.warning(self, "Vienna 参数错误", str(exc))

    def set_busy(self, busy):
        self.run_button.setEnabled(not busy)
        if hasattr(self, "input_tabs"):
            self.input_tabs.setEnabled(not busy)
        if hasattr(self, "codegen_button"):
            self.codegen_button.setEnabled(not busy)

    def set_result(self, result):
        self.result = result
        analysis, line, switching = result
        f = analysis.frequencies_hz

        ci = analysis.current_loop.responses
        self.bi.set_curves(f, [
            PFCBodeCurve("plant_gid", "Gid", ci["plant_gid"]),
            PFCBodeCurve("controller_ci", "Ci", ci["controller_ci"]),
            PFCBodeCurve("pwm_zoh", "PWM/ZOH", ci["pwm_zoh"]),
            PFCBodeCurve("sense_hi", "I Sense", ci["sense_hi"]),
            PFCBodeCurve("forward_current", "Current forward", ci["forward_current"]),
            PFCBodeCurve("open_current", "Li Phase-A Open Loop", ci["open_current"], True, True),
            PFCBodeCurve("closed_current_actual", "Ti", ci["closed_current_actual"]),
            PFCBodeCurve("sensitivity_current", "Si", ci["sensitivity_current"]),
        ], margins=analysis.current_loop.margins, open_loop_key="open_current")

        cv = analysis.voltage_loop.responses
        self.bv.set_curves(f, [
            PFCBodeCurve("controller_cv", "Cv", cv["controller_cv"]),
            PFCBodeCurve("current_closed_for_outer", "Closed current", cv["current_closed_for_outer"]),
            PFCBodeCurve("bus_plant_gvg", "Bus plant", cv["bus_plant_gvg"]),
            PFCBodeCurve("sense_hv", "Vdc sense", cv["sense_hv"]),
            PFCBodeCurve("forward_voltage", "Voltage forward", cv["forward_voltage"]),
            PFCBodeCurve("open_voltage", "Vdc Open Loop", cv["open_voltage"], True, True),
            PFCBodeCurve("closed_voltage", "Tv", cv["closed_voltage"]),
            PFCBodeCurve("sensitivity_voltage", "Sv", cv["sensitivity_voltage"]),
        ], margins=analysis.voltage_loop.margins, open_loop_key="open_voltage")

        cb = analysis.balance_loop.responses
        self.bb.set_curves(f, [
            PFCBodeCurve("controller_cb", "Cbal", cb["controller_cb"]),
            PFCBodeCurve("balance_plant", "Midpoint plant", cb["balance_plant"]),
            PFCBodeCurve("sense_balance", "Split bus sense", cb["sense_balance"]),
            PFCBodeCurve("forward_balance", "Balance forward", cb["forward_balance"]),
            PFCBodeCurve("open_balance", "Balance Open Loop", cb["open_balance"], True, True),
            PFCBodeCurve("closed_balance", "Tbal", cb["closed_balance"]),
            PFCBodeCurve("sensitivity_balance", "Sbal", cb["sensitivity_balance"]),
        ], margins=analysis.balance_loop.margins, open_loop_key="open_balance")

        self.bs.set_curves(f, [
            PFCBodeCurve("current_total", "Iabc total sense", analysis.current_sense_response.total),
            PFCBodeCurve("current_analog", "Iabc analog", analysis.current_sense_response.calibrated_analog),
            PFCBodeCurve("phase_voltage_total", "Vabc total sense", analysis.phase_voltage_sense_response.total),
            PFCBodeCurve("phase_voltage_analog", "Vabc analog", analysis.phase_voltage_sense_response.calibrated_analog),
            PFCBodeCurve("split_bus_total", "Vdc± total sense", analysis.split_bus_sense_response.total),
            PFCBodeCurve("split_bus_analog", "Vdc± analog", analysis.split_bus_sense_response.calibrated_analog),
        ])
        self.bs._set_all(False)

        self._plot_ac(line)
        self._plot_bus(line)
        self._plot_switch(switching)
        self._plot_sector(line)
        self._plot_harm(line)
        self._summary(analysis, line, switching)

    def _last(self, line):
        period = 1.0 / self.line_hz.value()
        start_time = float(line.time_s[-1]) - period
        start = int(np.searchsorted(line.time_s, start_time, side="left"))
        t = (line.time_s[start:] - line.time_s[start]) * 1e3
        return t, {k: np.asarray(v)[start:] for k, v in line.signals.items()}

    @staticmethod
    def _style(axes, xlabel):
        for ax in axes:
            ax.grid(True, alpha=.3)
            ax.legend(fontsize=8, ncol=3)
        axes[-1].set_xlabel(xlabel)

    def _plot_ac(self, line):
        t, s = self._last(line)
        f = self.fac; f.clear()
        axes = [f.add_subplot(411)]
        axes += [f.add_subplot(412, sharex=axes[0]), f.add_subplot(413, sharex=axes[0]), f.add_subplot(414, sharex=axes[0])]
        for ph in "abc":
            axes[0].plot(t, s[f"v{ph}"], label=f"V{ph}")
            axes[0].plot(t, s[f"v{ph}_meas"], linestyle=":", alpha=.65, label=f"V{ph} meas")
            axes[1].plot(t, s[f"i{ph}"], label=f"I{ph}")
            axes[1].plot(t, s[f"i{ph}_ref"], linestyle="--", label=f"I{ph} ref")
            axes[2].plot(t, s[f"mod_{ph}"], label=f"m{ph}")
            axes[2].plot(t, s[f"duty_{ph}"], linestyle="--", label=f"D0 {ph}")
        axes[3].plot(t, s["input_power_total"], label="3φ Pin")
        axes[3].axhline(self.pout.value(), linestyle="--", label="Pout")
        axes[0].set_ylabel("V")
        axes[1].set_ylabel("A")
        axes[2].set_ylabel("mod / D0")
        axes[3].set_ylabel("W")
        self._style(axes, "Time in final AC period (ms)")
        f.suptitle("Vienna 完整一个三相 AC 周期：测量 / 参考 / 调制 / 功率")
        f.tight_layout(); self.cac.draw_idle()

    def _plot_bus(self, line):
        t, s = self._last(line)
        f = self.fbus; f.clear()
        axes = [f.add_subplot(511)]
        axes += [f.add_subplot(512, sharex=axes[0]), f.add_subplot(513, sharex=axes[0]), f.add_subplot(514, sharex=axes[0]), f.add_subplot(515, sharex=axes[0])]
        axes[0].plot(t, s["vdc"], label="Vdc")
        axes[0].plot(t, s["vdc_plus"], label="Vdc+")
        axes[0].plot(t, s["vdc_minus"], label="Vdc-")
        axes[1].plot(t, s["vdc_delta"], label="Vdc+ - Vdc-")
        axes[2].plot(t, s["gcmd"], label="Gcmd")
        axes[2].plot(t, s["balance_output"], label="Balance out")
        axes[3].plot(t, s["third_harmonic_injection"], label="Third-harmonic/common-mode")
        for ph in "abc":
            axes[3].plot(t, s[f"inductor_ff_{ph}"], alpha=.7, label=f"FF {ph}")
        axes[4].plot(t, s["midpoint_current"], label="Midpoint current")
        axes[4].plot(t, s["bus_series_current"], label="Bus series current")
        self._style(axes, "Time in final AC period (ms)")
        f.suptitle("Vienna Split DC Bus / Voltage Loop / Balance / Feedforward")
        f.tight_layout(); self.cbus.draw_idle()

    def _plot_switch(self, sw):
        t = sw.time_s * 1e6
        s = sw.signals
        f = self.fsw_fig; f.clear()
        axes = [f.add_subplot(611)]
        axes += [f.add_subplot(612, sharex=axes[0]), f.add_subplot(613, sharex=axes[0]), f.add_subplot(614, sharex=axes[0]), f.add_subplot(615, sharex=axes[0]), f.add_subplot(616, sharex=axes[0])]
        for ph in "abc":
            axes[0].plot(t, s[f"gate_{ph}"], label=f"Center gate {ph}")
            axes[1].plot(t, s[f"vconv_{ph}"], label=f"Vconv {ph}")
            axes[2].plot(t, s[f"current_{ph}"], label=f"I{ph}")
            axes[3].plot(t, s[f"upper_diode_{ph}"], label=f"D+ {ph}")
            axes[3].plot(t, -s[f"lower_diode_{ph}"], linestyle="--", label=f"D- {ph}")
            axes[5].plot(t, s[f"duty_{ph}"], label=f"D0 {ph}")
        axes[4].plot(t, s["midpoint_current"], label="Imid")
        axes[4].plot(t, s["upper_cap_current"], label="IC+")
        axes[4].plot(t, s["lower_cap_current"], label="IC-")
        axes[0].set_ylabel("Gate")
        axes[1].set_ylabel("V")
        axes[2].set_ylabel("A")
        axes[3].set_ylabel("Diode A")
        axes[4].set_ylabel("Bus A")
        axes[5].set_ylabel("D0")
        self._style(axes, "Time (µs)")
        src = "" if sw.source_time_s is None else f", source t={sw.source_time_s*1e3:.4f} ms"
        f.suptitle(f"Vienna switching derived from AC workpoint @ {sw.line_angle_deg:.2f}°{src}")
        f.tight_layout(); self.csw.draw_idle()

    def _plot_sector(self, line):
        t, s = self._last(line)
        f = self.fsec; f.clear()
        axes = [f.add_subplot(311)]
        axes += [f.add_subplot(312, sharex=axes[0]), f.add_subplot(313, sharex=axes[0])]
        axes[0].step(t, s["sector"], where="post", label="Sector")
        axes[0].set_yticks(range(1, 7))
        for ph in "abc":
            axes[1].plot(t, s[f"mod_{ph}"], label=f"m{ph}")
            axes[1].plot(t, s[f"duty_{ph}"], linestyle="--", label=f"D0 {ph}")
        axes[2].plot(t, s["midpoint_current"], label="Imid")
        axes[2].plot(t, np.sign(s["va"]), label="sign Va")
        axes[2].plot(t, np.sign(s["vb"]), label="sign Vb")
        axes[2].plot(t, np.sign(s["vc"]), label="sign Vc")
        self._style(axes, "Time in final AC period (ms)")
        f.suptitle("Vienna Sector / phase polarity / zero-state duty / midpoint current")
        f.tight_layout(); self.csec.draw_idle()

    def _plot_harm(self, line):
        f = self.fharm; f.clear()
        ax = f.add_subplot(111)
        m = line.metrics
        orders = np.asarray(m.harmonic_orders, dtype=float)
        width = .24
        ax.bar(orders - width, m.phase_a_harmonic_rms_a, width=width, label="Ia")
        ax.bar(orders, m.phase_b_harmonic_rms_a, width=width, label="Ib")
        ax.bar(orders + width, m.phase_c_harmonic_rms_a, width=width, label="Ic")
        ax.set_xlabel("Harmonic order")
        ax.set_ylabel("RMS current (A)")
        ax.grid(True, axis="y", alpha=.3)
        ax.legend()
        ax.set_title(
            f"PF={m.overall_power_factor:.6f}; THD A/B/C={tuple(round(x,4) for x in m.phase_current_thd_percent)}%; "
            f"unbalance={m.current_unbalance_percent:.4f}%")
        f.tight_layout(); self.charm.draw_idle()

    def _summary(self, analysis, line, switching):
        m = line.metrics
        cfg = analysis.config
        self.summary.setPlainText("\n".join([
            "VIENNA PFC V7 SUMMARY",
            "=" * 88,
            f"Vll={cfg.power_stage.line_line_rms_v:.2f} V, Vdc={cfg.power_stage.bus_voltage_v:.2f} V, P={cfg.power_stage.output_power_w:.1f} W",
            "Main control: DC voltage outer loop + three ABC stationary-frame current inner loops",
            "Auxiliary control: split-bus midpoint balance",
            f"Third-harmonic/common-mode injection: {cfg.firmware.third_harmonic_injection_enabled}",
            f"Inductor R/L feedforward: {cfg.firmware.inductor_voltage_drop_feedforward_enabled}",
            "",
            f"Current loop PM={analysis.current_loop.margins.phase_margin_deg}, fc={analysis.current_loop.margins.critical_gain_crossover_hz}",
            f"Voltage loop PM={analysis.voltage_loop.margins.phase_margin_deg}, fc={analysis.voltage_loop.margins.critical_gain_crossover_hz}",
            f"Balance loop PM={analysis.balance_loop.margins.phase_margin_deg}, fc={analysis.balance_loop.margins.critical_gain_crossover_hz}",
            "",
            f"PF overall={m.overall_power_factor:.7f}",
            f"PF phase A/B/C={m.phase_power_factor}",
            f"Displacement A/B/C={m.phase_displacement_factor}",
            f"Distortion A/B/C={m.phase_distortion_factor}",
            f"THD A/B/C={m.phase_current_thd_percent} %",
            f"Current RMS A/B/C={m.phase_current_rms_a} A; unbalance={m.current_unbalance_percent:.5f}%",
            f"Vdc avg={m.bus_voltage_average_v:.4f} V, ripple pp={m.bus_voltage_ripple_pp_v:.4f} V",
            f"Midpoint ΔV avg={m.midpoint_delta_average_v:.5f} V, pp={m.midpoint_delta_pp_v:.5f} V, Imid RMS={m.midpoint_current_rms_a:.5f} A",
            f"Switching reconstruction source time={switching.source_time_s}",
            *( ["", "Warnings:"] + ["- " + w for w in (*analysis.warnings, *line.warnings)] if analysis.warnings or line.warnings else [] ),
        ]))

    def _diagram_selected(self, key):
        if key in {"voltage_controller", "gcmd", "references", "current_controller", "balance", "feedforward", "modulator"}:
            self.input_tabs.setCurrentIndex(1)
        elif key in {"current_sense", "phase_voltage_sense", "bus_sense"}:
            self.input_tabs.setCurrentIndex(2)
        else:
            self.input_tabs.setCurrentIndex(0)

        if key == "current_controller":
            self.tabs.setCurrentWidget(self.bi); self.bi.focus_curve("controller_ci")
        elif key == "current_sense":
            self.tabs.setCurrentWidget(self.bs); self.bs.focus_curve("current_total", keep_open_loop=False)
        elif key == "phase_voltage_sense":
            self.tabs.setCurrentWidget(self.bs); self.bs.focus_curve("phase_voltage_total", keep_open_loop=False)
        elif key == "voltage_controller":
            self.tabs.setCurrentWidget(self.bv); self.bv.focus_curve("controller_cv")
        elif key == "bus_sense":
            self.tabs.setCurrentWidget(self.bs); self.bs.focus_curve("split_bus_total", keep_open_loop=False)
        elif key == "balance":
            self.tabs.setCurrentWidget(self.bb); self.bb.focus_curve("controller_cb")
        elif key == "plant":
            self.tabs.setCurrentWidget(self.bi); self.bi.focus_curve("plant_gid")
        elif key in {"feedforward", "modulator", "references", "gcmd"}:
            # Feed-forward / modulation are nonlinear or common-mode paths and
            # should not be falsely presented as a return-ratio component.
            self.tabs.setCurrentWidget(self.bi); self.bi.show_open_loop_only()

    def _cursor_changed(self, measurement: BodeCursorMeasurement):
        if self._syncing_cursor:
            return
        self._syncing_cursor = True
        try:
            sender = self.sender()
            self.cursor.setText("\n".join([
                f"光标频率：{format_frequency(measurement.frequency_hz)}",
                *[f"{x.label}: Gain={x.gain_db:+.5g} dB, Phase={x.phase_deg:+.6g}°" for x in measurement.values],
            ]))
            if not self.result or sender is self.bs:
                self.budget.setText("Sampling-chain view: no loop phase-margin budget.")
                return
            analysis = self.result[0]
            if sender is self.bi:
                responses = analysis.current_loop.responses
                labels = {"controller_ci":"Ci", "pwm_zoh":"PWM", "plant_gid":"Plant", "sense_hi":"Sense", "open_current":"Li"}
                keys = list(labels)
            elif sender is self.bv:
                responses = analysis.voltage_loop.responses
                labels = {"controller_cv":"Cv", "current_closed_for_outer":"Ti", "bus_plant_gvg":"Bus", "sense_hv":"Sense", "open_voltage":"Lv"}
                keys = list(labels)
            else:
                responses = analysis.balance_loop.responses
                labels = {"controller_cb":"Cbal", "balance_plant":"Plant", "sense_balance":"Sense", "open_balance":"Lbal"}
                keys = list(labels)
            budget = phase_budget(analysis.frequencies_hz, responses, labels, measurement.frequency_hz, keys)
            self.budget.setText(" | ".join(f"{x.label}: {x.gain_db:+.2f}dB/{x.phase_deg:+.2f}°" for x in budget))
        finally:
            self._syncing_cursor = False


__all__ = ["ViennaControlLabView"]
