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
        self._spec=None
    def set_nominal_spec(self,spec):
        self._spec=spec
        self.vbus.setValue(float(spec.vbus_nom_v))
    def set_busy(self,busy:bool): self.run.setEnabled(not busy)
    def set_result(self,r):
        self.figure.clear(); axg=self.figure.add_subplot(211); ax=self.figure.add_subplot(212)
        if r.gain_curve is not None:
            g=r.gain_curve
            topo=g.topology.value if getattr(g, "topology", None) is not None else f"{r.phase_count}P"
            axg.semilogx(g.frequency_hz/1e3,g.normalized_gain,label=f'{topo} system gain')
            # SystemGainCurve is the physical DC ratio Vo/Vin.  Do not recover
            # a single-phase FHA gain from a per-phase result here: 3P has a
            # different equivalent circuit and may not own a 1P model result.
            if self._spec is not None:
                req=float(self._spec.vout_v)/max(float(self.vbus.value()),1e-12)
                axg.axhline(req,linestyle='--',label='Required Vo/Vin')
            axg.axvline(r.common_switching_frequency_hz/1e3,linestyle=':',label='Operating Fsw')
            axg.set_ylabel('DC gain  Vo/Vin');axg.set_xlabel('Frequency (kHz)');axg.grid(True,which='both',alpha=.3);axg.legend()
        t=r.waveform.time_s*1e6
        for p in range(r.phase_count): ax.plot(t,r.waveform.signal(f'i_rectified_p{p+1}').values,label=f'P{p+1} {r.phase_offsets_deg[p]:g}°')
        ax.plot(t,r.waveform.signal('i_rectified_total').values,label='Total',linewidth=2)
        ax.set_xlabel('Time (us)'); ax.set_ylabel('Rectified current (A)'); ax.grid(True,alpha=.3); ax.legend(); self.figure.tight_layout(); self.canvas.draw_idle()
        exact = r.electrical_point.exact_common_solution if r.electrical_point is not None else True
        residual = r.electrical_point.solver_residual if r.electrical_point is not None else 0.0
        topology = r.electrical_point.topology.value if r.electrical_point is not None else f"{r.phase_count}P"
        lines=[f"Topology: {topology}",f"{r.phase_count}-phase fixed offsets: {r.phase_offsets_deg}",f"Common Fsw: {r.common_switching_frequency_hz/1e3:.5f} kHz",f"Electrical load-share solve: {'EXACT' if exact else 'BEST-FIT / CURRENT-SHARING RISK'} (residual={residual:.3e})",f"Cout RMS current: {r.output_capacitor_rms_a:.4f} A",f"Output ripple: {r.output_ripple_vpp*1e3:.3f} mVpp",f"Input bus ripple RMS: {r.input_bus_ripple_rms_a:.4f} A",f"Current-share imbalance: {r.current_share_imbalance_percent:.5f}%",""]
        for p in r.phases:
            lines.append(
                f"Phase {p.index}: offset={p.phase_offset_deg:g}°, P={p.output_power_w:.2f} W, "
                f"share={p.share_percent:.3f}%, Q={p.q_effective:.5f}, phase={p.input_phase_deg:.3f}°, "
                f"Lr={(r.electrical_point.phases[p.index-1].tank.lr_h*1e6 if r.electrical_point else 0):.3f} µH"
            )
        if r.electrical_point is not None and r.electrical_point.phases:
            e0=r.electrical_point.phases[0]
            lines += [
                "",
                f"Topology tank: Lr={e0.tank.lr_h*1e6:.4f} µH, Cr={e0.tank.cr_f*1e9:.4f} nF, Lm={e0.tank.lm_h*1e6:.4f} µH",
                f"Turns: {e0.primary_turns}:{e0.secondary_turns}  n={e0.turns_ratio:.5f}",
            ]
        if r.warnings: lines += ["",*r.warnings]
        self.text.setPlainText("\n".join(lines))
