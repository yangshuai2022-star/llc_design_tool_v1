"""GUI page for V8 digital SR timing/LUT/loss."""
from __future__ import annotations
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QDoubleSpinBox,QFormLayout,QGroupBox,QLabel,QPlainTextEdit,QPushButton,QVBoxLayout,QWidget

class SRDesignView(QWidget):
    analysis_requested=Signal(dict)
    def __init__(self):
        super().__init__(); layout=QVBoxLayout(self); box=QGroupBox("Digital SR Timing / Loss") ; form=QFormLayout(box)
        self.vbus=QDoubleSpinBox(); self.vbus.setRange(20,2000); self.vbus.setValue(400); self.vbus.setSuffix(' V')
        self.load=QDoubleSpinBox(); self.load.setRange(1,150); self.load.setValue(100); self.load.setSuffix(' %')
        form.addRow('Vbus',self.vbus); form.addRow('Load',self.load)
        self.run=QPushButton('计算 SR Timing / Loss'); self.run.clicked.connect(lambda:self.analysis_requested.emit({'vbus_v':self.vbus.value(),'load_fraction':self.load.value()/100.0})); form.addRow(self.run); layout.addWidget(box)
        note=QLabel('基于 Golden Isec(t) 计算 ON delay、OFF advance、第三象限、body-diode、Qrr/trr recovery 与 C99 LUT 参数。'); note.setWordWrap(True); layout.addWidget(note)
        self.text=QPlainTextEdit(); self.text.setReadOnly(True); layout.addWidget(self.text,1)
    def set_nominal_spec(self,spec): self.vbus.setValue(float(spec.vbus_nom_v))
    def set_busy(self,busy): self.run.setEnabled(not busy)
    def set_result(self,sr):
        t=sr.timing; p=sr.loss
        self.text.setPlainText('\n'.join([
            f'Fsw = {t.switching_frequency_hz/1e3:.5f} kHz',f'ON delay = {t.on_delay_s*1e9:.2f} ns',f'OFF advance = {t.off_advance_s*1e9:.2f} ns',f'Conduction = {t.conduction_s*1e9:.2f} ns',f'Zero+ = {t.positive_zero_crossing_s*1e9:.2f} ns',f'Zero- = {t.negative_zero_crossing_s*1e9:.2f} ns','',f'Channel = {p.channel_w:.4f} W',f'Third quadrant = {p.third_quadrant_w:.4f} W',f'Body diode = {p.body_diode_w:.4f} W',f'Reverse current = {p.reverse_current_w:.4f} W',f'Reverse recovery = {p.reverse_recovery_w:.4f} W',f'Coss = {p.coss_w:.4f} W',f'Gate = {p.gate_drive_w:.4f} W',f'Total SR = {p.total_w:.4f} W',f'Qrr/cycle = {p.recovery_charge_c_per_cycle*1e9:.3f} nC',f'Reverse charge = {p.reverse_charge_c*1e9:.3f} nC','',*sr.warnings]))
