"""GUI page for fixed 90deg two-phase / 120deg three-phase LLC analysis."""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox,QDoubleSpinBox,QFormLayout,QGroupBox,QHBoxLayout,QLabel,QPlainTextEdit,QPushButton,QVBoxLayout,QWidget
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class InterleavedLLCView(QWidget):
    analysis_requested = Signal(int, dict)
    def __init__(self):
        super().__init__()
        layout=QVBoxLayout(self)
        box=QGroupBox("固定相位 Interleaved LLC")
        form=QFormLayout(box)
        self.phase=QComboBox(); self.phase.addItem("2 相交错 — 固定 90°",2); self.phase.addItem("3 相交错 — 固定 120°",3)
        self.vbus=QDoubleSpinBox(); self.vbus.setRange(20,2000); self.vbus.setDecimals(1); self.vbus.setSuffix(" V"); self.vbus.setValue(400)
        self.load=QDoubleSpinBox(); self.load.setRange(1,150); self.load.setDecimals(1); self.load.setSuffix(" %"); self.load.setValue(100)
        form.addRow("系统",self.phase); form.addRow("Vbus",self.vbus); form.addRow("Load",self.load)
        self.run=QPushButton("计算 Interleaved LLC")
        self.run.clicked.connect(lambda:self.analysis_requested.emit(int(self.phase.currentData()),{"vbus_v":self.vbus.value(),"load_fraction":self.load.value()/100.0}))
        form.addRow(self.run); layout.addWidget(box)
        note=QLabel("V8 固定定义：2 相 = 0°/90°；3 相 = 0°/120°/240°。不提供用户相位配置。")
        note.setWordWrap(True); layout.addWidget(note)
        self.figure=Figure(figsize=(8,4)); self.canvas=FigureCanvasQTAgg(self.figure); layout.addWidget(self.canvas,2)
        self.text=QPlainTextEdit(); self.text.setReadOnly(True); layout.addWidget(self.text,1)
    def set_nominal_spec(self,spec): self.vbus.setValue(float(spec.vbus_nom_v))
    def set_busy(self,busy:bool): self.run.setEnabled(not busy)
    def set_result(self,r):
        self.figure.clear(); ax=self.figure.add_subplot(111)
        t=r.waveform.time_s*1e6
        for p in range(r.phase_count): ax.plot(t,r.waveform.signal(f'i_rectified_p{p+1}').values,label=f'P{p+1} {r.phase_offsets_deg[p]:g}°')
        ax.plot(t,r.waveform.signal('i_rectified_total').values,label='Total',linewidth=2)
        ax.set_xlabel('Time (us)'); ax.set_ylabel('Rectified current (A)'); ax.grid(True,alpha=.3); ax.legend(); self.figure.tight_layout(); self.canvas.draw_idle()
        lines=[f"{r.phase_count}-phase fixed offsets: {r.phase_offsets_deg}",f"Cout RMS current: {r.output_capacitor_rms_a:.4f} A",f"Output ripple: {r.output_ripple_vpp*1e3:.3f} mVpp",f"Input bus ripple RMS: {r.input_bus_ripple_rms_a:.4f} A",f"Current-share imbalance: {r.current_share_imbalance_percent:.5f}%",""]
        for p in r.phases: lines.append(f"Phase {p.index}: offset={p.phase_offset_deg:g}°, P={p.output_power_w:.2f} W, share={p.share_percent:.3f}%")
        if r.warnings: lines += ["",*r.warnings]
        self.text.setPlainText("\n".join(lines))
