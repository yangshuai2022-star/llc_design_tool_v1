"""GUI page for the V8 FHA / multi-harmonic / switched-model comparison."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...analysis.types import MultiFidelityAnalysis


class ModelComparisonView(QWidget):
    """Work-point controls and tabular FHA/HB/TD error presentation."""

    analysis_requested = Signal(dict)

    HEADERS = (
        "模型",
        "参考",
        "收敛",
        "残差",
        "Fsw/kHz",
        "Vo/V",
        "Gain",
        "Ir RMS/A",
        "Ir PK/A",
        "Im RMS/A",
        "Is RMS/A",
        "VCr PK/V",
        "ΔIr RMS/%",
        "ΔVCr PK/%",
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.analysis: MultiFidelityAnalysis | None = None
        layout = QVBoxLayout(self)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("母线"))
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

        controls.addWidget(QLabel("最高谐波"))
        self.max_harmonic = QSpinBox()
        self.max_harmonic.setRange(1, 21)
        self.max_harmonic.setSingleStep(2)
        self.max_harmonic.setValue(7)
        controls.addWidget(self.max_harmonic)

        controls.addWidget(QLabel("HB 每周期采样"))
        self.hb_samples = QSpinBox()
        self.hb_samples.setRange(256, 8192)
        self.hb_samples.setSingleStep(256)
        self.hb_samples.setValue(1024)
        controls.addWidget(self.hb_samples)

        self.include_td = QCheckBox("运行分段时域参考")
        self.include_td.setChecked(True)
        controls.addWidget(self.include_td)
        self.run_button = QPushButton("运行 FHA / HB / TD 对比")
        self.run_button.clicked.connect(
            lambda: self.analysis_requested.emit(self.options()))
        controls.addWidget(self.run_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        note = QLabel(
            "FHA 用于快速设计；HB 对整流换相与高次谐波自洽求解；"
            "分段时域为当前 V8.1 理想拓扑参考。误差均相对最高已收敛模型。"
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        self.table = QTableWidget(0, len(self.HEADERS))
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        layout.addWidget(self.table, 1)

        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(210)
        layout.addWidget(self.details)

    def options(self) -> dict:
        maximum = self.max_harmonic.value()
        if maximum % 2 == 0:
            maximum += 1
        return {
            "vbus_v": self.vbus.value(),
            "load_fraction": self.load_percent.value() / 100.0,
            "max_harmonic": maximum,
            "hb_samples": self.hb_samples.value(),
            "include_time_domain": self.include_td.isChecked(),
        }

    def set_nominal_work_point(self, vbus_v: float) -> None:
        self.vbus.setValue(vbus_v)

    def set_busy(self, busy: bool) -> None:
        self.run_button.setEnabled(not busy)

    @staticmethod
    def _item(value: str) -> QTableWidgetItem:
        item = QTableWidgetItem(value)
        return item

    def set_analysis(self, analysis: MultiFidelityAnalysis) -> None:
        self.analysis = analysis
        self.table.setRowCount(len(analysis.comparison_rows))
        for row_index, row in enumerate(analysis.comparison_rows):
            metrics = row.metrics
            values = (
                row.fidelity.value,
                "REF" if row.fidelity is analysis.reference_level else "",
                "PASS" if row.converged else "FAIL",
                f"{row.residual_norm:.3e}",
                f"{metrics.switching_frequency_hz/1e3:.6f}",
                f"{metrics.output_voltage_v:.6f}",
                f"{metrics.normalized_gain:.8f}",
                f"{metrics.resonant_current_rms_a:.6f}",
                f"{metrics.resonant_current_peak_a:.6f}",
                f"{metrics.magnetizing_current_rms_a:.6f}",
                f"{metrics.secondary_current_rms_a:.6f}",
                f"{metrics.resonant_capacitor_peak_v:.6f}",
                f"{row.resonant_rms_error_percent:.4f}",
                f"{row.resonant_cap_peak_error_percent:.4f}",
            )
            for column, value in enumerate(values):
                self.table.setItem(row_index, column, self._item(value))

        lines = [
            f"Reference: {analysis.reference_level.value}",
            f"Work point: {analysis.request.bus_voltage_v:.3f} V / "
            f"{analysis.request.load_fraction*100:.3f}% / "
            f"{analysis.request.requested_output_power_w:.3f} W",
            "",
        ]
        for level, result in analysis.results.items():
            lines.append(
                f"[{level.value}] {result.convergence.method}; "
                f"converged={result.convergence.converged}; "
                f"residual={result.convergence.residual_norm:.6e}"
            )
            if result.harmonic_orders:
                lines.append(
                    "  Harmonics: " + ", ".join(str(value) for value in result.harmonic_orders)
                )
            for warning in result.warnings:
                lines.append(f"  - {warning}")
        for warning in analysis.warnings:
            lines.append(f"[analysis] {warning}")
        self.details.setPlainText("\n".join(lines))
