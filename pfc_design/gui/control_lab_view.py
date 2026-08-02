"""PySide6 PFC double-loop, sensing and waveform workbench."""

from __future__ import annotations

import math

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PySide6.QtCore import Signal
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
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
    PFCPowerStageConfig,
    PFCSwitchingWaveforms,
)

from .bode_panel import PFCBodeCurve, SelectableBodePanel


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
        root.addLayout(header)

        splitter = QSplitter()
        splitter.addWidget(self._build_input_panel())
        splitter.addWidget(self._build_result_panel())
        splitter.setSizes([400, 1180])
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, 1)

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
        kp = self._spin(0, 1000, 8, 0.1 if voltage else 0.01)
        ti = self._spin(0.001, 10000, 5, 2.0 if voltage else 0.075, " ms")
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

    def _sense_group(self, title: str, defaults: tuple[float, ...]):
        gain, bw_khz, source_r, source_c_nf, out_r, out_c_nf, sample_khz = defaults
        group = QGroupBox(title)
        form = QFormLayout(group)
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
            "电感电流采样", (0.03, 2000, 0, 0, 220, 2, 50))
        vac, self.vac_sense = self._sense_group(
            "AC 电压采样", (1 / 150, 1000, 2000, 1, 220, 2, 50))
        vbus_gain = 1600 / (117000 + 1600)
        vbus_r = 117000 * 1600 / (117000 + 1600)
        vbus, self.vbus_sense = self._sense_group(
            "母线电压采样", (vbus_gain, 1000, vbus_r, 1, 220, 2, 10))
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

    def set_busy(self, busy: bool) -> None:
        self.run_button.setEnabled(not busy)

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
        self._cursor_frequency_hz = (
            analysis.current_loop.margins.critical_gain_crossover_hz
        )
        self._set_bode_results(analysis)
        self._plot_ac_overview(line_cycle, analysis)
        self._plot_ac_control(line_cycle, analysis)
        self._plot_switching(switching)
        self._show_summary(analysis, line_cycle, switching)

    def _set_bode_results(self, analysis: PFCControlLabAnalysis) -> None:
        f = analysis.frequencies_hz
        ci = analysis.current_loop.responses
        self.current_bode.set_curves(f, [
            PFCBodeCurve("plant_gid", "Gid 功率级", ci["plant_gid"]),
            PFCBodeCurve("controller_ci", "Ci 数字控制器", ci["controller_ci"]),
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

        axes[2].plot(time_us, s["inductor_current"], label="Inductor current")
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
            f"THD={m.current_thd_percent:.5f}%",
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
