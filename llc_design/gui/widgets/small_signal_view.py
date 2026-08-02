"""Continuous/discrete LLC plant controls, Bode plot and equations."""

from __future__ import annotations

import math
import numpy as np

from PySide6.QtCore import Signal
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from ...control.analysis import SmallSignalAnalysis
from ...control.linearize import ControlInputKind, SISOTransferFunction
from .bode_cursor import (
    BodeCursor,
    BodeCursorMeasurement,
    BodeCursorTrace,
    format_frequency,
)


class SmallSignalView(QWidget):
    """Power-stage plant workbench for G(s), exact-ZOH G(z) and C export."""

    analysis_requested = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.result: SmallSignalAnalysis | None = None
        self._bode_cursor: BodeCursor | None = None
        self._cursor_frequency_hz: float | None = None
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        form = QFormLayout()

        self.vbus = QDoubleSpinBox()
        self.vbus.setRange(20.0, 2000.0)
        self.vbus.setDecimals(2)
        self.vbus.setValue(400.0)
        self.vbus.setSuffix(" V")
        self.load_percent = QDoubleSpinBox()
        self.load_percent.setRange(1.0, 150.0)
        self.load_percent.setDecimals(1)
        self.load_percent.setValue(100.0)
        self.load_percent.setSuffix(" %")
        self.sample_us = QDoubleSpinBox()
        self.sample_us.setRange(0.1, 10000.0)
        self.sample_us.setDecimals(3)
        self.sample_us.setValue(20.0)
        self.sample_us.setSuffix(" µs")
        self.input_kind = QComboBox()
        for kind in ControlInputKind:
            self.input_kind.addItem(kind.value, kind)
        self.timer_mhz = QDoubleSpinBox()
        self.timer_mhz.setRange(1.0, 2000.0)
        self.timer_mhz.setValue(120.0)
        self.timer_mhz.setSuffix(" MHz")
        self.delay_samples = QSpinBox()
        self.delay_samples.setRange(0, 20)
        self.transfer_kind = QComboBox()
        self.transfer_kind.addItem("Gvf：频率/周期/计数 → 输出电压", "gvf")
        self.transfer_kind.addItem("Gvg：母线电压 → 输出电压", "gvg")
        self.transfer_kind.addItem("Zout：负载电流 → 输出电压", "zout")
        self.transfer_kind.addItem("Girf：控制量 → 谐振电流 RMS", "girf")
        self.transfer_kind.addItem("Gimf：控制量 → 励磁电流 RMS", "gimf")

        form.addRow("工作点母线", self.vbus)
        form.addRow("工作点负载", self.load_percent)
        form.addRow("控制采样周期", self.sample_us)
        form.addRow("控制输入", self.input_kind)
        form.addRow("Timer 时钟", self.timer_mhz)
        form.addRow("计算/PWM 延迟", self.delay_samples)
        form.addRow("显示对象", self.transfer_kind)
        controls.addLayout(form)
        self.run_button = QPushButton("建立小信号对象")
        self.run_button.clicked.connect(self._request)
        controls.addWidget(self.run_button)
        # No stretch here: the transfer-function panel takes all remaining width.
        # Transfer functions and ZOH parameters live in the top-right area;
        # the Bode plot keeps the full width below.
        transfer_box = QGroupBox("传递函数与离散参数")
        transfer_layout = QVBoxLayout(transfer_box)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setMaximumHeight(320)
        self.text.setMinimumHeight(240)
        font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        font.setPointSize(12)
        self.text.setFont(font)
        transfer_layout.addWidget(self.text)
        controls.addWidget(transfer_box, 1)
        layout.addLayout(controls)

        self.cursor_status = QLabel("Bode 光标：在幅频图或相频图内单击并拖动竖线")
        self.cursor_status.setWordWrap(True)
        self.cursor_status.setStyleSheet(
            "QLabel { padding: 5px; border: 1px solid #b8b8b8; "
            "background: #f6f6f6; font-family: monospace; }"
        )
        layout.addWidget(self.cursor_status)
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas, 1)
        self.transfer_kind.currentIndexChanged.connect(self.refresh)

    def set_nominal_work_point(self, vbus_v: float) -> None:
        self.vbus.setValue(vbus_v)

    def _request(self) -> None:
        kind = self.input_kind.currentData()
        self.analysis_requested.emit({
            "vbus_v": self.vbus.value(),
            "load_fraction": self.load_percent.value() / 100.0,
            "sample_time_s": self.sample_us.value() * 1e-6,
            "control_input_kind": kind,
            "timer_clock_hz": self.timer_mhz.value() * 1e6,
            "input_delay_samples": self.delay_samples.value(),
        })

    def set_busy(self, busy: bool) -> None:
        self.run_button.setEnabled(not busy)

    def set_analysis(self, result: SmallSignalAnalysis) -> None:
        self.result = result
        self._cursor_frequency_hz = None
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

    def _selected_transfer(self) -> tuple[str, SISOTransferFunction, bool]:
        if self.result is None:
            raise RuntimeError("small-signal result is not available")
        key = self.transfer_kind.currentData()
        mapping = {
            "gvf": ("Gvf", self.result.continuous_transfer, True),
            "gvg": ("Gvg", self.result.line_to_output_transfer, False),
            "zout": ("Zout", self.result.output_impedance_transfer, False),
            "girf": ("Girf", self.result.resonant_current_transfer, False),
            "gimf": ("Gimf", self.result.magnetizing_current_transfer, False),
        }
        return mapping[key]

    @staticmethod
    def _poly_text(values: np.ndarray, variable: str) -> str:
        degree = len(values) - 1
        terms: list[str] = []
        for index, coefficient in enumerate(values):
            power = degree - index
            if abs(float(coefficient)) < 1e-16:
                continue
            sign = "-" if coefficient < 0 else "+"
            magnitude = abs(float(coefficient))
            factor = f"{magnitude:.8e}"
            if power == 1:
                factor += variable
            elif power > 1:
                factor += f"{variable}^{power}"
            if not terms:
                terms.append(("-" if coefficient < 0 else "") + factor)
            else:
                terms.append(f" {sign} {factor}")
        return "".join(terms) or "0"

    @staticmethod
    def _z_text(values: np.ndarray) -> str:
        """Format a G(z) polynomial in powers of z^-k (digital form)."""
        terms: list[str] = []
        for index, coefficient in enumerate(values):
            if abs(float(coefficient)) < 1e-16:
                continue
            sign = "-" if coefficient < 0 else "+"
            magnitude = abs(float(coefficient))
            factor = f"{magnitude:.8e}" if index == 0 else f"{magnitude:.8e} z^-{index}"
            if not terms:
                terms.append(("-" if coefficient < 0 else "") + factor)
            else:
                terms.append(f" {sign} {factor}")
        return "".join(terms) or "0"

    @staticmethod
    def _matrix_text(matrix: np.ndarray, label: str, shape: str) -> str:
        body = np.array2string(
            np.asarray(matrix, dtype=float),
            precision=6, suppress_small=True, max_line_width=100,
            separator=", ")
        return f"{label} ({shape}):\n{body}"

    def refresh(self) -> None:
        if self.result is None:
            return
        result = self.result
        name, transfer, _ = self._selected_transfer()
        if self._bode_cursor is not None:
            self._bode_cursor.disconnect()
            self._bode_cursor = None
        self.figure.clear()
        ax_mag = self.figure.add_subplot(211)
        ax_phase = self.figure.add_subplot(212, sharex=ax_mag)
        fsw = result.operating_point.switching_frequency_hz
        nyquist = 0.5 / result.sample_time_s
        fmax = max(100.0, min(0.45 * fsw, 0.90 * nyquist))
        frequencies = np.geomspace(max(0.5, fmax / 1e5), fmax, 1000)
        hc = transfer.frequency_response(frequencies)
        continuous_gain = 20.0 * np.log10(np.maximum(np.abs(hc), 1e-18))
        continuous_phase = np.unwrap(np.angle(hc)) * 180.0 / math.pi
        continuous_line, = ax_mag.semilogx(
            frequencies, continuous_gain, label=f"{name}(s)")
        ax_phase.semilogx(
            frequencies, continuous_phase, label=f"{name}(s)",
            color=continuous_line.get_color())
        hd = result.discrete_plant.frequency_response(frequencies)
        discrete_gain = 20.0 * np.log10(np.maximum(np.abs(hd), 1e-18))
        discrete_phase = np.unwrap(np.angle(hd)) * 180.0 / math.pi
        discrete_line, = ax_mag.semilogx(
            frequencies, discrete_gain, linestyle="--", label="exact-ZOH G(z)")
        ax_phase.semilogx(
            frequencies, discrete_phase, linestyle="--", label="exact-ZOH G(z)",
            color=discrete_line.get_color())
        cursor_traces = [
            BodeCursorTrace(
                label=f"{name}(s)", frequencies_hz=frequencies,
                gain_db=continuous_gain, phase_deg=continuous_phase,
                color=continuous_line.get_color()),
            BodeCursorTrace(
                label="exact-ZOH G(z)", frequencies_hz=frequencies,
                gain_db=discrete_gain, phase_deg=discrete_phase,
                color=discrete_line.get_color()),
        ]
        ax_mag.set_ylabel("Magnitude (dB)")
        ax_mag.grid(True, which="both", alpha=0.3)
        ax_mag.legend()
        ax_phase.set_ylabel("Phase (deg)")
        ax_phase.set_xlabel("Perturbation frequency (Hz)")
        ax_phase.grid(True, which="both", alpha=0.3)
        ax_phase.legend()
        self.figure.tight_layout()
        self._bode_cursor = BodeCursor(
            self.canvas,
            ax_mag,
            ax_phase,
            cursor_traces,
            initial_frequency_hz=self._cursor_frequency_hz,
            on_changed=self._cursor_changed,
        )
        self.canvas.draw_idle()

        discrete = result.discrete_plant
        continuous_poles = "\n".join(
            f"  {pole.real:.7g} {pole.imag:+.7g}j rad/s"
            for pole in transfer.poles)
        continuous_zeros = "\n".join(
            f"  {zero.real:.7g} {zero.imag:+.7g}j rad/s"
            for zero in transfer.zeros) or "  —"
        discrete_poles = "\n".join(
            f"  {pole.real:.7g} {pole.imag:+.7g}j"
            for pole in discrete.poles)
        discrete_zeros = "\n".join(
            f"  {zero.real:.7g} {zero.imag:+.7g}j"
            for zero in discrete.zeros) or "  —"

        text = [
            "LLC 功率级对象",
            "=" * 72,
            f"工作点: {result.operating_point.vbus_v:.1f} V, "
            f"{result.operating_point.load_fraction*100:.1f}% load, "
            f"fs={result.operating_point.switching_frequency_hz/1e3:.5f} kHz",
            f"EDF steady Vo={result.steady_state.output_voltage_v:.9f} V",
            f"等效串联阻尼={result.parameters.series_resistance_ohm:.7g} Ω",
            f"控制输入: {transfer.input_name} [{transfer.input_unit}]",
            f"输出: {transfer.output_name} [{transfer.output_unit}]",
            "",
            "一、连续域传递函数 G(s)",
            "-" * 72,
            f"显示对象: {name}",
            f"DC gain={transfer.dc_gain:.12g} {transfer.output_unit}/{transfer.input_unit}",
            f"Gvg(0)={result.line_to_output_transfer.dc_gain:.12g} V/V",
            f"Zout(0)={result.output_impedance_transfer.dc_gain:.12g} Ω",
            f"Girf(0)={result.resonant_current_transfer.dc_gain:.12g}",
            f"Gimf(0)={result.magnetizing_current_transfer.dc_gain:.12g}",
            "",
            f"G(s) = ({self._poly_text(transfer.numerator, 's')})",
            f"       / ({self._poly_text(transfer.denominator, 's')})",
            "",
            "Continuous poles:",
            continuous_poles,
            "Continuous zeros:",
            continuous_zeros,
            f"连续稳定: {result.continuous_plant.stable}",
            "",
            "二、离散域传递函数 G(z)（精确 ZOH）",
            "-" * 72,
            f"采样周期 Ts={discrete.sample_time_s*1e6:.6g} µs, "
            f"采样频率={1.0/discrete.sample_time_s/1e3:.6g} kHz, "
            f"计算/PWM 延迟={discrete.input_delay_samples} 采样",
            "",
            f"G(z) = ({self._z_text(discrete.numerator)})",
            f"       / ({self._z_text(discrete.denominator)})",
            "",
            "差分方程:",
            f"  {discrete.difference_equation.text(precision=12)}",
            "",
            "Discrete poles (z 平面):",
            discrete_poles,
            "Discrete zeros (z 平面):",
            discrete_zeros,
            f"离散稳定: {discrete.stable}",
            "",
            "三、ZOH 离散化参数",
            "-" * 72,
            self._matrix_text(discrete.ad, "Ad", f"{discrete.ad.shape[0]}×{discrete.ad.shape[1]}"),
            "",
            self._matrix_text(discrete.bd, "Bd", "×".join(map(str, discrete.bd.shape))),
            "",
            self._matrix_text(discrete.cd, "Cd", "×".join(map(str, discrete.cd.shape))),
            "",
            f"Dd = {discrete.dd[0, 0]:.12g}",
            "",
            f"状态顺序: {', '.join(result.continuous_plant.state_names)}",
            f"x[k+1] = Ad·x[k] + Bd·u[k],  y[k] = Cd·x[k] + Dd·u[k]",
        ]
        self.text.setPlainText("\n".join(text))
