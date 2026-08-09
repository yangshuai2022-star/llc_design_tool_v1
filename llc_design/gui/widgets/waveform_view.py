"""Interactive PyQtGraph LLC waveform workbench.

The widget deliberately consumes :class:`WaveformBundle` only.  It does not
know whether the data came from the fast EDF reconstruction, the piecewise
switched solver, or a future measured/PLECS import.  This keeps the GUI
independent from the waveform engine.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
import pyqtgraph as pg

from ...dynamics.waveforms import WaveformBundle

pg.setConfigOptions(antialias=True)


class WaveformView(QWidget):
    """Synchronized multi-trace waveform viewer with two time cursors."""

    fast_requested = Signal(dict)
    detailed_requested = Signal(dict)

    GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
        (
            "关键节点",
            (
                "v_bridge", "i_resonant", "v_resonant_cap",
                "v_transformer_primary", "i_transformer_secondary",
                "i_output_cap", "v_output_ripple",
            ),
        ),
        (
            "桥臂与谐振腔",
            (
                "v_leg_a", "v_leg_b", "v_bridge", "i_resonant",
                "v_resonant_inductor", "v_resonant_cap", "energy_lr",
            ),
        ),
        (
            "变压器",
            (
                "v_transformer_primary", "i_transformer_primary",
                "i_magnetizing", "i_primary_load",
                "v_transformer_secondary", "i_transformer_secondary",
            ),
        ),
        (
            "次级与输出",
            (
                "v_transformer_secondary", "i_transformer_secondary",
                "v_rectified", "i_rectified", "i_load_output",
                "i_output_cap", "v_output_cap_internal", "v_output_ripple",
            ),
        ),
        (
            "一次开关",
            (
                "gate_q1", "vds_q1", "ids_q1",
                "gate_q2", "vds_q2", "ids_q2",
                "gate_q3", "vds_q3", "ids_q3",
                "gate_q4", "vds_q4", "ids_q4",
            ),
        ),
        (
            "同步整流",
            (
                "gate_sr1", "vds_sr1", "ids_sr1",
                "gate_sr2", "vds_sr2", "ids_sr2",
                "gate_sr3", "vds_sr3", "ids_sr3",
                "gate_sr4", "vds_sr4", "ids_sr4",
            ),
        ),
        (
            "磁件",
            (
                "b_transformer", "h_transformer",
                "b_resonant_inductor", "h_resonant_inductor", "energy_lr",
            ),
        ),
    )

    COMPONENT_GROUP_INDEX = {
        "bridge": 1,
        "lr": 1,
        "cr": 1,
        "transformer": 2,
        "sr": 5,
        "output": 3,
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.bundle: WaveformBundle | None = None
        self._plots: list[pg.PlotItem] = []
        self._cursor_a_lines: list[pg.InfiniteLine] = []
        self._cursor_b_lines: list[pg.InfiniteLine] = []
        self._cursor_guard = False

        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("工作点母线"))
        self.vbus = QDoubleSpinBox()
        self.vbus.setRange(20.0, 2000.0)
        self.vbus.setDecimals(2)
        self.vbus.setValue(400.0)
        self.vbus.setSuffix(" V")
        controls.addWidget(self.vbus)
        controls.addWidget(QLabel("负载"))
        self.load_percent = QDoubleSpinBox()
        self.load_percent.setRange(1.0, 150.0)
        self.load_percent.setDecimals(1)
        self.load_percent.setValue(100.0)
        self.load_percent.setSuffix(" %")
        controls.addWidget(self.load_percent)
        controls.addWidget(QLabel("波形组"))
        self.group_combo = QComboBox()
        self.group_combo.addItems([name for name, _ in self.GROUPS])
        controls.addWidget(self.group_combo)
        self.fast_button = QPushButton("快速 EDF 波形")
        self.detail_button = QPushButton("详细分段波形")
        self.fast_button.clicked.connect(lambda: self.fast_requested.emit(self.work_point()))
        self.detail_button.clicked.connect(lambda: self.detailed_requested.emit(self.work_point()))
        controls.addWidget(self.fast_button)
        controls.addWidget(self.detail_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        cursor_controls = QHBoxLayout()
        cursor_controls.addWidget(QLabel("游标 A"))
        self.cursor_a_us = QDoubleSpinBox()
        self.cursor_a_us.setRange(-1e9, 1e9)
        self.cursor_a_us.setDecimals(5)
        self.cursor_a_us.setSuffix(" µs")
        cursor_controls.addWidget(self.cursor_a_us)
        cursor_controls.addWidget(QLabel("游标 B"))
        self.cursor_b_us = QDoubleSpinBox()
        self.cursor_b_us.setRange(-1e9, 1e9)
        self.cursor_b_us.setDecimals(5)
        self.cursor_b_us.setSuffix(" µs")
        cursor_controls.addWidget(self.cursor_b_us)
        self.cursor_measurement = QLabel("Δt = —; 1/Δt = —")
        cursor_controls.addWidget(self.cursor_measurement)
        cursor_controls.addStretch(1)
        layout.addLayout(cursor_controls)

        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setBackground("w")
        graphics_scroll = QScrollArea()
        graphics_scroll.setWidgetResizable(True)
        graphics_scroll.setWidget(self.graphics)
        layout.addWidget(graphics_scroll, 1)

        self.statistics = QTableWidget(0, 12)
        self.statistics.setHorizontalHeaderLabels(
            [
                "信号", "单位", "频率", "平均", "RMS", "最小", "最大",
                "峰峰值", "绝对峰值", "峰值因数", "基波 RMS", "THD",
            ]
        )
        self.statistics.setMaximumHeight(190)
        header = self.statistics.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        for column in range(1, self.statistics.columnCount()):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.statistics)

        self.group_combo.currentIndexChanged.connect(self.refresh)
        self.cursor_a_us.valueChanged.connect(self._cursor_spin_changed)
        self.cursor_b_us.valueChanged.connect(self._cursor_spin_changed)

    def work_point(self) -> dict[str, float]:
        return {
            "vbus_v": self.vbus.value(),
            "load_fraction": self.load_percent.value() / 100.0,
        }

    def set_nominal_work_point(self, vbus_v: float) -> None:
        self.vbus.setValue(vbus_v)

    def set_busy(self, busy: bool) -> None:
        self.fast_button.setEnabled(not busy)
        self.detail_button.setEnabled(not busy)

    def set_bundle(self, bundle: WaveformBundle) -> None:
        self.bundle = bundle
        if len(bundle.time_s):
            start = float(bundle.time_s[0] * 1e6)
            stop = float(bundle.time_s[-1] * 1e6)
            self.cursor_a_us.setValue(start + 0.25 * (stop - start))
            self.cursor_b_us.setValue(start + 0.75 * (stop - start))
        self.refresh()

    def select_component(self, key: str) -> None:
        index = self.COMPONENT_GROUP_INDEX.get(key)
        if index is not None:
            self.group_combo.setCurrentIndex(index)

    def _selected_keys(self) -> tuple[str, ...]:
        if self.bundle is None:
            return ()
        _, keys = self.GROUPS[self.group_combo.currentIndex()]
        return tuple(key for key in keys if key in self.bundle.signals)

    def _cursor_spin_changed(self) -> None:
        if self._cursor_guard:
            return
        self._set_cursor_lines(self.cursor_a_us.value(), self.cursor_b_us.value())
        self._update_cursor_measurement()

    def _line_moved(self, source: str, position_us: float) -> None:
        if self._cursor_guard:
            return
        self._cursor_guard = True
        try:
            target = self.cursor_a_us if source == "a" else self.cursor_b_us
            target.setValue(float(position_us))
            lines = self._cursor_a_lines if source == "a" else self._cursor_b_lines
            for line in lines:
                if abs(line.value() - position_us) > 1e-12:
                    line.setValue(position_us)
        finally:
            self._cursor_guard = False
        self._update_cursor_measurement()

    def _set_cursor_lines(self, a_us: float, b_us: float) -> None:
        self._cursor_guard = True
        try:
            for line in self._cursor_a_lines:
                line.setValue(a_us)
            for line in self._cursor_b_lines:
                line.setValue(b_us)
        finally:
            self._cursor_guard = False

    def _update_cursor_measurement(self) -> None:
        delta_us = self.cursor_b_us.value() - self.cursor_a_us.value()
        if abs(delta_us) < 1e-15:
            self.cursor_measurement.setText("Δt = 0 µs; 1/Δt = —")
            return
        reciprocal_khz = 1e3 / abs(delta_us)
        self.cursor_measurement.setText(
            f"Δt = {delta_us:.6g} µs; 1/|Δt| = {reciprocal_khz:.6g} kHz"
        )

    def refresh(self) -> None:
        self.graphics.clear()
        self.statistics.setRowCount(0)
        self._plots.clear()
        self._cursor_a_lines.clear()
        self._cursor_b_lines.clear()
        if self.bundle is None:
            return
        keys = self._selected_keys()
        time_us = self.bundle.time_s * 1e6
        previous: pg.PlotItem | None = None
        for row, key in enumerate(keys):
            signal = self.bundle.signals[key]
            plot = self.graphics.addPlot(row=row, col=0)
            plot.setMinimumHeight(180)
            plot.showGrid(x=True, y=True, alpha=0.25)
            plot.setLabel("left", signal.label, units=signal.unit)
            if row == len(keys) - 1:
                plot.setLabel("bottom", "Time", units="µs")
            if previous is not None:
                plot.setXLink(previous)
            plot.plot(time_us, signal.values, pen=pg.mkPen(width=3.0))
            line_a = pg.InfiniteLine(
                pos=self.cursor_a_us.value(), angle=90, movable=(row == 0),
                pen=pg.mkPen((30, 120, 220), width=1.4), label="A" if row == 0 else None,
            )
            line_b = pg.InfiniteLine(
                pos=self.cursor_b_us.value(), angle=90, movable=(row == 0),
                pen=pg.mkPen((220, 100, 30), width=1.4), label="B" if row == 0 else None,
            )
            plot.addItem(line_a)
            plot.addItem(line_b)
            if row == 0:
                line_a.sigPositionChanged.connect(
                    lambda line: self._line_moved("a", float(line.value())))
                line_b.sigPositionChanged.connect(
                    lambda line: self._line_moved("b", float(line.value())))
            self._cursor_a_lines.append(line_a)
            self._cursor_b_lines.append(line_b)
            self._plots.append(plot)
            previous = plot

            stats = signal.statistics
            table_row = self.statistics.rowCount()
            self.statistics.insertRow(table_row)
            values = (
                signal.label,
                signal.unit,
                f"{stats.frequency_hz/1e3:.7g} kHz",
                f"{stats.average:.8g}",
                f"{stats.rms:.8g}",
                f"{stats.minimum:.8g}",
                f"{stats.maximum:.8g}",
                f"{stats.peak_to_peak:.8g}",
                f"{stats.absolute_peak:.8g}",
                f"{stats.crest_factor:.6g}",
                f"{stats.fundamental_rms:.8g}",
                f"{stats.thd_percent:.5g}%",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.statistics.setItem(table_row, column, item)
        self._update_cursor_measurement()
