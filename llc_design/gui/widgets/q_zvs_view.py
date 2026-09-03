"""LLC Q / gain / ZVS region visualizer."""
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Signal
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...core.q_zvs import LLCQZVSAnalysis


class LLCQZVSView(QWidget):
    analysis_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.result: LLCQZVSAnalysis | None = None
        root = QVBoxLayout(self)
        top = QHBoxLayout()
        text = QLabel(
            "LLC Q / ZVS 工作区域：不同负载 Q、增益、理论感性区与实际换流裕量"
        )
        text.setStyleSheet("font-size:16px;font-weight:600;padding:4px;")
        top.addWidget(text)
        top.addStretch(1)
        self.normalized_x = QCheckBox("横轴使用 Fn=Fsw/Fr")
        self.normalized_x.setChecked(True)
        self.normalized_x.toggled.connect(self.refresh)
        top.addWidget(self.normalized_x)
        run = QPushButton("重新计算 Q / ZVS")
        run.clicked.connect(self.analysis_requested)
        top.addWidget(run)
        root.addLayout(top)

        self.tabs = QTabWidget()
        self.gain_figure = Figure(figsize=(10, 7))
        self.gain_canvas = FigureCanvasQTAgg(self.gain_figure)
        gain_page = QWidget(); gl = QVBoxLayout(gain_page); gl.addWidget(self.gain_canvas)
        self.tabs.addTab(gain_page, "多负载 Gain / Q")

        self.zvs_figure = Figure(figsize=(10, 7))
        self.zvs_canvas = FigureCanvasQTAgg(self.zvs_figure)
        zvs_page = QWidget(); zl = QVBoxLayout(zvs_page); zl.addWidget(self.zvs_canvas)
        self.tabs.addTab(zvs_page, "ZVS Region Map")

        self.margin_figure = Figure(figsize=(10, 7))
        self.margin_canvas = FigureCanvasQTAgg(self.margin_figure)
        margin_page = QWidget(); ml = QVBoxLayout(margin_page); ml.addWidget(self.margin_canvas)
        self.tabs.addTab(margin_page, "ZVS Margin")

        self.table = QPlainTextEdit(); self.table.setReadOnly(True)
        fixed = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        fixed.setPointSize(10); self.table.setFont(fixed)
        self.tabs.addTab(self.table, "工作点表")
        root.addWidget(self.tabs, 1)

    def set_analysis(self, result: LLCQZVSAnalysis) -> None:
        self.result = result
        self.refresh()

    def refresh(self, *_args) -> None:
        if self.result is None:
            return
        r = self.result
        m = r.map
        x = m.normalized_frequency if self.normalized_x.isChecked() else m.frequencies_hz / 1e3
        xlabel = "Normalized frequency Fn = Fsw / Fr" if self.normalized_x.isChecked() else "Switching frequency (kHz)"

        self.gain_figure.clear()
        ax = self.gain_figure.add_subplot(211)
        axq = self.gain_figure.add_subplot(212)
        for idx, load in enumerate(m.load_fractions):
            ax.semilogx(x, m.gain[idx], label=f"{load*100:.0f}% load, Q={m.q_effective[idx]:.3f}")
        ax.axvline(1.0 if self.normalized_x.isChecked() else r.tank.fr_hz/1e3, linestyle="--", linewidth=0.9, label="Fr")
        ax.set_ylabel("FHA gain |M|")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        axq.plot(m.load_fractions * 100.0, m.q_effective, marker="o")
        axq.set_xlabel("Load (%)")
        axq.set_ylabel("Effective Q")
        axq.grid(True, alpha=0.3)
        self.gain_figure.suptitle("LLC 多负载增益与有效 Q")
        self.gain_figure.tight_layout()
        self.gain_canvas.draw_idle()

        self.zvs_figure.clear()
        ax = self.zvs_figure.add_subplot(111)
        # 0=capacitive, 1=inductive but margin<1, 2=warning, 3=safe
        region = np.zeros_like(m.zvs_margin)
        region[m.theoretical_inductive] = 1.0
        region[m.zvs_warning] = 2.0
        region[m.zvs_safe] = 3.0
        xx, yy = np.meshgrid(x, m.load_fractions * 100.0)
        contour = ax.contourf(xx, yy, region, levels=[-0.5,0.5,1.5,2.5,3.5])
        cb = self.zvs_figure.colorbar(contour, ax=ax, ticks=[0,1,2,3])
        cb.ax.set_yticklabels(["Capacitive", "ZVS fail", "Warning", "ZVS safe"])
        for wp in r.workpoints:
            wx = wp.normalized_frequency if self.normalized_x.isChecked() else wp.frequency_hz/1e3
            ax.scatter([wx], [wp.load_fraction*100.0], s=28)
            ax.annotate(f"{wp.vbus_v:.0f}V", (wx, wp.load_fraction*100.0), xytext=(4,4), textcoords="offset points", fontsize=7)
        ax.set_xscale("log")
        ax.set_xlabel(xlabel); ax.set_ylabel("Load (%)")
        ax.grid(True, which="both", alpha=0.25)
        ax.set_title("理论感性区 + 实际 Qoss/Coss/Deadtime ZVS 区域")
        self.zvs_figure.tight_layout(); self.zvs_canvas.draw_idle()

        self.margin_figure.clear()
        ax = self.margin_figure.add_subplot(111)
        for idx, load in enumerate(m.load_fractions):
            ax.semilogx(x, m.zvs_margin[idx], label=f"{load*100:.0f}% load")
        ax.axhline(1.0, linestyle="--", linewidth=0.9, label="Physical threshold")
        ax.axhline(r.spec.primary_zvs_margin_required, linestyle=":", linewidth=1.0, label="Preferred margin")
        ax.set_xlabel(xlabel); ax.set_ylabel("min(Q-margin, E-margin)")
        ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8, ncol=2)
        ax.set_title("Primary MOSFET actual ZVS commutation margin")
        self.margin_figure.tight_layout(); self.margin_canvas.draw_idle()

        lines = [
            "LLC Q / ZVS WORK POINTS",
            "="*118,
            f"{'Vbus':>8} {'Load':>8} {'Q':>9} {'Fsw/kHz':>11} {'Fn':>8} {'Gain':>9} {'Phase':>10} {'Icomm/A':>10} {'Qm':>9} {'Em':>9} {'Margin':>9} {'Status':>13}",
            "-"*118,
        ]
        for wp in r.workpoints:
            lines.append(
                f"{wp.vbus_v:8.1f} {wp.load_fraction*100:7.1f}% {wp.q_effective:9.4f} "
                f"{wp.frequency_hz/1e3:11.3f} {wp.normalized_frequency:8.4f} {wp.gain:9.4f} "
                f"{wp.phase_deg:9.3f}° {wp.commutation_current_a:10.4f} {wp.zvs_charge_margin:9.3f} "
                f"{wp.zvs_energy_margin:9.3f} {wp.zvs_margin:9.3f} {wp.zvs_status:>13}"
            )
        if r.warnings:
            lines.extend(["", "Warnings:", *[f"- {w}" for w in r.warnings]])
        self.table.setPlainText("\n".join(lines))


__all__ = ["LLCQZVSView"]
