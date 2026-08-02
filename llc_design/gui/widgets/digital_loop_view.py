"""Complete LLC digital voltage-loop design workbench."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from PySide6.QtCore import Signal
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QCheckBox,
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
    DelayEnvelope,
    DigitalLoopAnalysis,
    FMLUTMode,
    FrequencyModulatorLUT,
    PIControllerConfig,
    PIFControllerConfig,
    PWMCountMode,
    TwoP2ZControllerConfig,
)
from ...control.linearize import ControlInputKind
from .bode_cursor import (
    BodeCursor,
    BodeCursorMeasurement,
    BodeCursorTrace,
    format_frequency,
)


class DigitalLoopView(QWidget):
    """PI/PIF/2P2Z, FM LUT, sensing and complete open-loop analysis."""

    analysis_requested = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.result: DigitalLoopAnalysis | None = None
        self._bode_cursor: BodeCursor | None = None
        self._cursor_frequency_hz: float | None = None
        root = QHBoxLayout(self)
        splitter = QSplitter()
        splitter.addWidget(self._build_control_panel())
        splitter.addWidget(self._build_result_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([470, 1250])
        root.addWidget(splitter)

    @staticmethod
    def _double(minimum: float, maximum: float, decimals: int, value: float, suffix: str = "") -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(minimum, maximum)
        box.setDecimals(decimals)
        box.setValue(value)
        box.setSuffix(suffix)
        box.setKeyboardTracking(False)
        return box

    def _build_control_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)

        work = QGroupBox("工作点与采样")
        form = QFormLayout(work)
        self.vbus = self._double(20, 2000, 2, 400, " V")
        self.load_percent = self._double(1, 150, 1, 100, " %")
        self.sample_us = self._double(0.5, 1000, 3, 20, " µs")
        form.addRow("母线电压", self.vbus)
        form.addRow("负载", self.load_percent)
        form.addRow("控制周期", self.sample_us)
        layout.addWidget(work)

        controller = QGroupBox("数字控制器")
        form = QFormLayout(controller)
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
        layout.addWidget(controller)

        fm = QGroupBox("FM：PCMD → 开关频率")
        form = QFormLayout(fm)
        self.fm_mode = QComboBox()
        self.fm_mode.addItem("PCMD → TBPRD（固件一致）", FMLUTMode.PCMD_TO_TBPRD)
        self.fm_mode.addItem("PCMD → Frequency", FMLUTMode.PCMD_TO_FREQUENCY)
        self.timer_mhz = self._double(1, 2000, 3, 120, " MHz")
        self.count_mode = QComboBox()
        self.count_mode.addItem("Up-Down", PWMCountMode.UP_DOWN)
        self.count_mode.addItem("Up", PWMCountMode.UP)
        self.auto_pcmd = QCheckBox("由功率级工作频率反求 PCMD")
        self.auto_pcmd.setChecked(True)
        self.pcmd = self._double(0, 1, 6, 0.5)
        self.pcmd.setEnabled(False)
        self.auto_pcmd.toggled.connect(lambda checked: self.pcmd.setEnabled(not checked))
        self.lut_text = QPlainTextEdit()
        self.lut_text.setPlainText(FrequencyModulatorLUT.firmware_default().to_text())
        self.lut_text.setMinimumHeight(190)
        fixed = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        self.lut_text.setFont(fixed)
        form.addRow("LUT 类型", self.fm_mode)
        form.addRow("TBCLK", self.timer_mhz)
        form.addRow("计数模式", self.count_mode)
        form.addRow(self.auto_pcmd)
        form.addRow("手动 PCMD", self.pcmd)
        form.addRow("用户 LUT", self.lut_text)
        layout.addWidget(fm)

        sense = QGroupBox("外部模拟采样链")
        form = QFormLayout(sense)
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
        form.addRow("运放小信号带宽（0=理想）", self.opamp_bw_khz)
        form.addRow("ADC 前串联电阻", self.adc_r)
        form.addRow("ADC 输入对地电容", self.adc_c_nf)
        layout.addWidget(sense)

        adc = QGroupBox("ADC、CLA 与 PWM 延迟")
        form = QFormLayout(adc)
        self.adc_clock_mhz = self._double(1, 1000, 3, 60.0, " MHz")
        self.acq_ns = self._double(1, 100000, 2, 300.0, " ns")
        self.conversion_cycles = self._double(1, 100, 3, 13.0, " cycles")
        self.soc_count = QSpinBox(); self.soc_count.setRange(1, 16); self.soc_count.setValue(3)
        self.previous_weight = self._double(0, 0.999, 6, 0.25)
        self.computation_us = self._double(0, 1000, 4, 1.0, " µs")
        form.addRow("ADCCLK", self.adc_clock_mhz)
        form.addRow("Acquisition window", self.acq_ns)
        form.addRow("转换周期（可按芯片修正）", self.conversion_cycles)
        form.addRow("连续 SOC 数", self.soc_count)
        form.addRow("上一平均值权重", self.previous_weight)
        form.addRow("CLA 计算时间", self.computation_us)
        layout.addWidget(adc)

        self.run_button = QPushButton("建立完整数字电压环")
        self.run_button.clicked.connect(self._request)
        layout.addWidget(self.run_button)
        layout.addStretch(1)
        scroll.setWidget(content)
        self._update_controller_fields()
        return scroll

    def _build_result_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        top = QHBoxLayout()
        self.plot_group = QComboBox()
        self.plot_group.addItem("系统总览", "overview")
        self.plot_group.addItem("功率级与 FM", "plant")
        self.plot_group.addItem("数字控制器", "controller")
        self.plot_group.addItem("模拟与 ADC 采样", "sense")
        self.plot_group.addItem("开环延迟包络", "delay")
        self.plot_group.addItem("闭环、灵敏度与输出阻抗", "closed")
        self.plot_group.currentIndexChanged.connect(self.refresh)
        top.addWidget(QLabel("Bode 曲线组"))
        top.addWidget(self.plot_group)
        top.addStretch(1)
        layout.addLayout(top)
        self.cursor_status = QLabel("Bode 光标：在幅频图或相频图内单击并拖动竖线")
        self.cursor_status.setWordWrap(True)
        self.cursor_status.setStyleSheet(
            "QLabel { padding: 5px; border: 1px solid #b8b8b8; "
            "background: #f6f6f6; font-family: monospace; }"
        )
        layout.addWidget(self.cursor_status)

        self.figure = Figure(figsize=(11, 7))
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas, 1)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setMinimumHeight(260)
        fixed = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        fixed.setPointSize(11)
        self.text.setFont(fixed)
        layout.addWidget(self.text)
        return panel

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
                include_zero_order_hold=True,
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
            f"光标频率：{format_frequency(measurement.frequency_hz)}",
            *[
                f"{value.label}: Gain={value.gain_db:+.6g} dB, "
                f"Phase={value.phase_deg:+.7g}°"
                for value in measurement.values
            ],
        ]
        self.cursor_status.setText("\n".join(lines))

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
            "overview": [
                ("Gvf 功率级", "power_stage", "-"),
                ("Kfm×Gvf", "fm_power_stage", "--"),
                ("C(z)", "controller", ":"),
                ("采样链", "sense_total", "-."),
                ("系统开环", "open_loop_nominal", "-"),
                ("系统闭环", "closed_loop_nominal", "--"),
            ],
            "plant": [
                ("Gvf: fs→Vo", "power_stage", "-"),
                ("Gpcmd: PCMD→Vo", "fm_power_stage", "--"),
            ],
            "controller": [(result.controller.name, "controller", "-")],
            "sense": [
                ("分压/运放/ADC RC（原始）", "sense_analog_raw", "-"),
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
        for label, key, linestyle in groups[group]:
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
        if group in ("overview", "delay", "closed") and margin.critical_gain_crossover_hz:
            ax_mag.axvline(margin.critical_gain_crossover_hz, linewidth=0.9, linestyle=":")
            ax_phase.axvline(margin.critical_gain_crossover_hz, linewidth=0.9, linestyle=":")
        ax_mag.set_ylabel("Magnitude (dB)")
        ax_phase.set_ylabel("Phase (deg)")
        ax_phase.set_xlabel("Perturbation frequency (Hz)")
        for axis in (ax_mag, ax_phase):
            axis.grid(True, which="both", alpha=0.3)
            axis.legend(fontsize=8, ncol=2)
        self.figure.tight_layout()
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
        poles = ", ".join(f"{p.real:.5g}{p.imag:+.5g}j |z|={abs(p):.5g}" for p in discrete.closed_loop_poles) or "—"
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
