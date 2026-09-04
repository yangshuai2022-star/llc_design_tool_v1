"""Digital Control Tools GUI for power-electronics controller/filter design.

V8.3-alpha2 interaction rules:
* Controller and filter design are separate pages, not one crowded form.
* Selecting a controller exposes only the parameters meaningful to that type.
* H(s) and H(z) are always visible directly below the parameter selector and
  update while a text box or slider is changed.
* C99 export is one header-only float32_t file using DF2T/SOS.
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton,
    QScrollArea, QSlider, QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from llc_design.gui import theme
from llc_design.gui.updater import add_toolbar_right_side
from power_control_tools.analysis import analyze_digital_filter
from power_control_tools.codegen import export_c99_filter, render_c99_single_file, verify_c99_filter
from power_control_tools.controllers import design_controller
from power_control_tools.discretize import discretize_transfer_function
from power_control_tools.filters import design_iir_filter, design_fir_filter, design_moving_average, design_dc_blocker
from power_control_tools.models import ControllerKind, DiscretizationMethod, FilterResponse, IIRFamily, StabilityClass


class SliderSpin(QWidget):
    valueChanged = Signal(float)

    def __init__(self, minimum: float, maximum: float, value: float, *, decimals: int = 4,
                 logarithmic: bool = False, suffix: str = ""):
        super().__init__()
        self.minimum = float(minimum); self.maximum = float(maximum); self.logarithmic = logarithmic
        row = QHBoxLayout(self); row.setContentsMargins(0, 0, 0, 0)
        self.spin = QDoubleSpinBox(); self.spin.setRange(minimum, maximum); self.spin.setDecimals(decimals)
        self.spin.setSuffix(suffix); self.spin.setKeyboardTracking(False); self.spin.setMinimumWidth(132)
        self.slider = QSlider(Qt.Orientation.Horizontal); self.slider.setRange(0, 1000)
        row.addWidget(self.spin); row.addWidget(self.slider, 1)
        self.spin.valueChanged.connect(self._from_spin); self.slider.valueChanged.connect(self._from_slider)
        self.setValue(value)

    def _to_slider(self, v: float) -> int:
        if self.logarithmic:
            lo, hi = math.log10(self.minimum), math.log10(self.maximum)
            x = (math.log10(max(v, self.minimum)) - lo) / (hi - lo)
        else:
            x = (v - self.minimum) / (self.maximum - self.minimum)
        return int(round(np.clip(x, 0, 1) * 1000))

    def _from_slider_value(self, n: int) -> float:
        x = n / 1000.0
        if self.logarithmic:
            return 10 ** (math.log10(self.minimum) + x * (math.log10(self.maximum) - math.log10(self.minimum)))
        return self.minimum + x * (self.maximum - self.minimum)

    def _from_spin(self, v: float):
        self.slider.blockSignals(True); self.slider.setValue(self._to_slider(v)); self.slider.blockSignals(False)
        self.valueChanged.emit(float(v))

    def _from_slider(self, n: int):
        v = self._from_slider_value(n)
        self.spin.blockSignals(True); self.spin.setValue(v); self.spin.blockSignals(False)
        self.valueChanged.emit(float(v))

    def value(self) -> float: return float(self.spin.value())
    def setValue(self, v: float): self.spin.setValue(float(v)); self.slider.setValue(self._to_slider(float(v)))


_CONTROLLER_LABELS = {
    ControllerKind.INTEGRATOR: "Integrator",
    ControllerKind.PI: "PI — Kp · (1 + 1/(Ti·s))",
    ControllerKind.PIF: "PIF — PI + 1st-order LPF",
    ControllerKind.PID: "PID — Kp · (1 + 1/(Ti·s) + Td·s)",
    ControllerKind.PIDF: "PIDF — PID + 1st-order LPF",
    ControllerKind.TYPE_II: "Analog Type-II",
    ControllerKind.TYPE_III: "Analog Type-III",
    ControllerKind.MODIFIED_PI: "Modified PI",
    ControllerKind.LEAD: "Lead",
    ControllerKind.LAG: "Lag",
    ControllerKind.ONE_P_ONE_Z: "1P1Z",
    ControllerKind.TWO_P_TWO_Z: "2P2Z",
    ControllerKind.THREE_P_THREE_Z: "3P3Z",
    ControllerKind.GENERAL: "General H(s)",
}


class ControlToolsMainWindow(QMainWindow):
    workspace_switch_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("电源设计工具箱 — Digital Control Tools")
        self.resize(1800, 1050)
        self.current_analog = None; self.current_digital = None; self.current_analysis = None
        # QTabWidget.addTab() fires currentChanged -> _changed() -> schedule() during
        # construction, so the timer must exist before the parameter widgets are built.
        self.timer = QTimer(self); self.timer.setSingleShot(True); self.timer.setInterval(45)
        self.timer.timeout.connect(self.recalculate)
        root = QWidget(); row = QHBoxLayout(root)
        self.param_widget = self._build_parameters(); self.tabs = self._build_tabs()
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(self.param_widget); scroll.setMinimumWidth(500); scroll.setMaximumWidth(650)
        row.addWidget(scroll, 0); row.addWidget(self.tabs, 1); self.setCentralWidget(root)
        self._build_toolbar(); self.setStyleSheet(theme.workspace_stylesheet(theme.active_theme()))
        self._update_dynamic_visibility(); self.recalculate()

    def _build_toolbar(self):
        tb = self.addToolBar("Control Tools"); tb.setMovable(False)
        for name, target in [("LLC", "llc"), ("PFC", "pfc"), ("功能选择", "home")]:
            a = QAction(name, self); a.triggered.connect(lambda checked=False, t=target: self.workspace_switch_requested.emit(t)); tb.addAction(a)
        tb.addSeparator(); a = QAction("重新计算", self); a.triggered.connect(self.recalculate); tb.addAction(a)
        a = QAction("导出单文件 C99", self); a.triggered.connect(self.export_c99); tb.addAction(a); add_toolbar_right_side(tb, self)

    def _hook(self, w):
        if hasattr(w, "valueChanged"): w.valueChanged.connect(lambda *_: self.schedule())
        if hasattr(w, "currentIndexChanged"): w.currentIndexChanged.connect(lambda *_: self._changed())

    def _changed(self):
        self._update_dynamic_visibility(); self.schedule()

    def schedule(self): self.timer.start()

    @staticmethod
    def _field_visible(form: QFormLayout, widget: QWidget, visible: bool):
        widget.setVisible(visible)
        label = form.labelForField(widget)
        if label is not None: label.setVisible(visible)

    def _build_parameters(self):
        w = QWidget(); v = QVBoxLayout(w); w.setMinimumWidth(470); w.setMaximumWidth(620)

        g = QGroupBox("Sampling / Discretization"); f = QFormLayout(g)
        self.fs = SliderSpin(1000, 1_000_000, 40_000, decimals=1, logarithmic=True, suffix=" Hz")
        self.method = QComboBox(); [(self.method.addItem(x.value, x)) for x in DiscretizationMethod]
        self.prewarp = SliderSpin(1, 200_000, 1000, decimals=1, logarithmic=True, suffix=" Hz")
        for x in (self.fs, self.method, self.prewarp): self._hook(x)
        f.addRow("Fs", self.fs); f.addRow("S→Z", self.method); f.addRow("Prewarp", self.prewarp)
        self.sampling_form = f; v.addWidget(g)

        self.design_pages = QTabWidget(); self.design_pages.currentChanged.connect(lambda *_: self._changed())
        self.design_pages.addTab(self._build_controller_page(), "Controller")
        self.design_pages.addTab(self._build_filter_page(), "Filter Designer")
        v.addWidget(self.design_pages)

        # Always-visible immediate feedback, intentionally placed directly under
        # parameter selection as requested by power-control users.
        g = QGroupBox("实时传递函数 — 参数/Slider 改变立即更新")
        gv = QVBoxLayout(g)
        self.live_hs = QPlainTextEdit(); self.live_hs.setReadOnly(True); self.live_hs.setMaximumHeight(100)
        self.live_hz = QPlainTextEdit(); self.live_hz.setReadOnly(True); self.live_hz.setMaximumHeight(115)
        gv.addWidget(QLabel("S-domain  H(s)")); gv.addWidget(self.live_hs)
        gv.addWidget(QLabel("Z-domain  H(z)")); gv.addWidget(self.live_hz)
        v.addWidget(g)

        g = QGroupBox("C99 Export"); f = QFormLayout(g); self.export_prefix = QLineEdit("POWER_CTRL")
        f.addRow("Symbol Prefix", self.export_prefix); v.addWidget(g)
        self.status = QLabel(); self.status.setWordWrap(True); v.addWidget(self.status)
        self.export = QPushButton("一键导出单文件 C99 float32_t / DF2T"); self.export.clicked.connect(self.export_c99); v.addWidget(self.export)
        v.addStretch(1); return w

    def _build_controller_page(self):
        page = QWidget(); f = QFormLayout(page); self.ctrl_form = f
        self.ctrl = QComboBox()
        for k in ControllerKind: self.ctrl.addItem(_CONTROLLER_LABELS[k], k)
        self._hook(self.ctrl); f.addRow("Controller", self.ctrl)

        self.type_input_mode = QComboBox(); self.type_input_mode.addItem("Pole / Zero", "pz"); self.type_input_mode.addItem("R / C Components", "rc"); self._hook(self.type_input_mode)
        self.gain = SliderSpin(0.0001, 10000, 1, decimals=6, logarithmic=True)
        self.kp = SliderSpin(0.0001, 10000, 1, decimals=6, logarithmic=True)
        self.ti = SliderSpin(1e-6, 10.0, 0.01, decimals=7, logarithmic=True, suffix=" s")
        self.td = SliderSpin(1e-8, 1.0, 1e-4, decimals=8, logarithmic=True, suffix=" s")
        self.lpf_pole = SliderSpin(0.1, 500_000, 10_000, decimals=2, logarithmic=True, suffix=" Hz")
        self.fp0 = SliderSpin(0.01, 200_000, 100, decimals=2, logarithmic=True, suffix=" Hz")
        self.fz = SliderSpin(0.1, 200_000, 500, decimals=2, logarithmic=True, suffix=" Hz")
        self.fp = SliderSpin(0.1, 500_000, 10_000, decimals=2, logarithmic=True, suffix=" Hz")
        self.fz2 = SliderSpin(0.1, 200_000, 1000, decimals=2, logarithmic=True, suffix=" Hz")
        self.fp2 = SliderSpin(0.1, 500_000, 20_000, decimals=2, logarithmic=True, suffix=" Hz")
        self.fz3 = SliderSpin(0.1, 200_000, 3000, decimals=2, logarithmic=True, suffix=" Hz")
        self.fp3 = SliderSpin(0.1, 500_000, 30_000, decimals=2, logarithmic=True, suffix=" Hz")
        self.r1 = SliderSpin(10, 10_000_000, 10_000, decimals=1, logarithmic=True, suffix=" Ω")
        self.r2 = SliderSpin(10, 10_000_000, 47_000, decimals=1, logarithmic=True, suffix=" Ω")
        self.r3 = SliderSpin(10, 10_000_000, 10_000, decimals=1, logarithmic=True, suffix=" Ω")
        self.c1_nf = SliderSpin(0.001, 100_000, 10, decimals=4, logarithmic=True, suffix=" nF")
        self.c2_nf = SliderSpin(0.001, 100_000, 0.47, decimals=4, logarithmic=True, suffix=" nF")
        self.c3_nf = SliderSpin(0.001, 100_000, 1.0, decimals=4, logarithmic=True, suffix=" nF")
        self.general_num = QLineEdit("1"); self.general_den = QLineEdit("1, 1")
        self.general_num.textChanged.connect(lambda *_: self.schedule()); self.general_den.textChanged.connect(lambda *_: self.schedule())
        fields = [
            ("Type-II/III input", self.type_input_mode), ("Gain K", self.gain), ("Kp", self.kp), ("Ti", self.ti), ("Td", self.td),
            ("LPF pole", self.lpf_pole), ("Integrator fp0", self.fp0), ("Zero fz1", self.fz), ("Pole fp1", self.fp),
            ("Zero fz2", self.fz2), ("Pole fp2", self.fp2), ("Zero fz3", self.fz3), ("Pole fp3", self.fp3),
            ("R1", self.r1), ("R2", self.r2), ("R3", self.r3), ("C1", self.c1_nf), ("C2", self.c2_nf), ("C3", self.c3_nf),
            ("General numerator", self.general_num), ("General denominator", self.general_den),
        ]
        self.ctrl_fields = {widget: label for label, widget in fields}
        for label, widget in fields:
            if isinstance(widget, SliderSpin) or isinstance(widget, QComboBox): self._hook(widget)
            f.addRow(label, widget)
        return page

    def _build_filter_page(self):
        page = QWidget(); f = QFormLayout(page); self.filter_form = f
        self.filter_impl = QComboBox(); self.filter_impl.addItems(["IIR", "FIR Window", "Moving Average", "DC Blocker"]); self._hook(self.filter_impl)
        self.response = QComboBox(); [(self.response.addItem(x.value, x)) for x in FilterResponse]; self._hook(self.response)
        self.family = QComboBox(); [(self.family.addItem(x.value, x)) for x in IIRFamily]; self._hook(self.family)
        self.order = QSpinBox(); self.order.setRange(1, 12); self.order.setValue(2); self._hook(self.order)
        self.fir_taps = QSpinBox(); self.fir_taps.setRange(3, 255); self.fir_taps.setSingleStep(2); self.fir_taps.setValue(31); self._hook(self.fir_taps)
        self.fir_window = QComboBox(); self.fir_window.addItems(["hamming", "hann", "blackman"]); self._hook(self.fir_window)
        self.fc = SliderSpin(0.1, 200_000, 1080, decimals=2, logarithmic=True, suffix=" Hz")
        self.fc2 = SliderSpin(0.1, 200_000, 5000, decimals=2, logarithmic=True, suffix=" Hz")
        self.q = SliderSpin(0.1, 100, 10, decimals=4, logarithmic=True)
        self.rp = SliderSpin(0.01, 6.0, 1.0, decimals=3, logarithmic=True, suffix=" dB")
        self.rs = SliderSpin(10, 120, 40, decimals=1, logarithmic=False, suffix=" dB")
        self.dc_radius = SliderSpin(0.80, 0.999999, 0.995, decimals=6, logarithmic=False)
        for x in (self.fc, self.fc2, self.q, self.rp, self.rs, self.dc_radius): self._hook(x)
        fields = [
            ("Implementation", self.filter_impl), ("Response", self.response), ("IIR Family", self.family), ("IIR Order", self.order),
            ("FIR taps / MA length", self.fir_taps), ("FIR Window", self.fir_window), ("Fc / F1", self.fc), ("F2", self.fc2),
            ("Q (Notch)", self.q), ("Passband ripple Rp", self.rp), ("Stop attenuation Rs", self.rs), ("DC Blocker pole r", self.dc_radius),
        ]
        self.filter_fields = {widget: label for label, widget in fields}
        for label, widget in fields: f.addRow(label, widget)
        return page

    def _update_dynamic_visibility(self):
        if not hasattr(self, "ctrl_form"): return
        kind = self.ctrl.currentData(); rc = self.type_input_mode.currentData() == "rc"
        visible = {self.ctrl}
        if kind == ControllerKind.INTEGRATOR: visible |= {self.gain}
        elif kind == ControllerKind.PI: visible |= {self.kp, self.ti}
        elif kind == ControllerKind.PIF: visible |= {self.kp, self.ti, self.lpf_pole}
        elif kind == ControllerKind.PID: visible |= {self.kp, self.ti, self.td}
        elif kind == ControllerKind.PIDF: visible |= {self.kp, self.ti, self.td, self.lpf_pole}
        elif kind == ControllerKind.TYPE_II:
            visible |= {self.type_input_mode}
            visible |= ({self.r1, self.r2, self.c1_nf, self.c2_nf} if rc else {self.fp0, self.fz, self.fp})
        elif kind == ControllerKind.TYPE_III:
            visible |= {self.type_input_mode}
            visible |= ({self.r1, self.r2, self.r3, self.c1_nf, self.c2_nf, self.c3_nf} if rc else {self.fp0, self.fz, self.fz2, self.fp, self.fp2})
        elif kind == ControllerKind.MODIFIED_PI: visible |= {self.gain, self.fz, self.fp}
        elif kind in (ControllerKind.LEAD, ControllerKind.LAG, ControllerKind.ONE_P_ONE_Z): visible |= {self.gain, self.fz, self.fp}
        elif kind == ControllerKind.TWO_P_TWO_Z: visible |= {self.gain, self.fz, self.fz2, self.fp, self.fp2}
        elif kind == ControllerKind.THREE_P_THREE_Z: visible |= {self.gain, self.fz, self.fz2, self.fz3, self.fp, self.fp2, self.fp3}
        elif kind == ControllerKind.GENERAL: visible |= {self.general_num, self.general_den}
        for widget in self.ctrl_fields: self._field_visible(self.ctrl_form, widget, widget in visible)

        if hasattr(self, "filter_form"):
            impl = self.filter_impl.currentText(); response = self.response.currentData()
            fv = {self.filter_impl}
            if impl == "IIR":
                fv |= {self.response, self.family, self.order, self.fc}
                if response in (FilterResponse.BANDPASS, FilterResponse.BANDSTOP): fv.add(self.fc2)
                if response == FilterResponse.NOTCH: fv.add(self.q)
                if self.family.currentData() in (IIRFamily.CHEBYSHEV1, IIRFamily.ELLIPTIC): fv.add(self.rp)
                if self.family.currentData() in (IIRFamily.CHEBYSHEV2, IIRFamily.ELLIPTIC): fv.add(self.rs)
            elif impl == "FIR Window":
                fv |= {self.response, self.fir_taps, self.fir_window, self.fc}
                if response in (FilterResponse.BANDPASS, FilterResponse.BANDSTOP, FilterResponse.NOTCH): fv.add(self.fc2)
            elif impl == "Moving Average": fv |= {self.fir_taps}
            else: fv |= {self.dc_radius}
            for widget in self.filter_fields: self._field_visible(self.filter_form, widget, widget in fv)
        self._field_visible(self.sampling_form, self.prewarp, self.method.currentData() == DiscretizationMethod.PREWARP_TUSTIN)

    def _figure_tab(self):
        fig = Figure(figsize=(8, 6)); canvas = FigureCanvasQTAgg(fig); return fig, canvas

    def _build_tabs(self):
        t = QTabWidget()
        self.bode_fig, self.bode_canvas = self._figure_tab(); t.addTab(self.bode_canvas, "Bode")
        self.step_fig, self.step_canvas = self._figure_tab(); t.addTab(self.step_canvas, "Step")
        self.imp_fig, self.imp_canvas = self._figure_tab(); t.addTab(self.imp_canvas, "Impulse")
        self.pz_fig, self.pz_canvas = self._figure_tab(); t.addTab(self.pz_canvas, "Pole-Zero")
        self.gd_fig, self.gd_canvas = self._figure_tab(); t.addTab(self.gd_canvas, "Group Delay")
        self.coeff = QPlainTextEdit(); self.coeff.setReadOnly(True); t.addTab(self.coeff, "Coefficients / SOS")
        self.tf = QPlainTextEdit(); self.tf.setReadOnly(True); t.addTab(self.tf, "Transfer Function")
        self.c99 = QPlainTextEdit(); self.c99.setReadOnly(True); t.addTab(self.c99, "C99 Single File")
        return t

    @staticmethod
    def _parse_coeff(text: str):
        vals = [float(v.strip()) for v in text.replace(';', ',').split(',') if v.strip()]
        if not vals: raise ValueError("General coefficients cannot be empty")
        return vals

    def _controller_kwargs(self):
        return dict(
            gain=self.gain.value(), kp=self.kp.value(), ti_s=self.ti.value(), td_s=self.td.value(), lpf_pole_hz=self.lpf_pole.value(),
            fp0_hz=self.fp0.value(), fz_hz=self.fz.value(), fp_hz=self.fp.value(), fz1_hz=self.fz.value(), fp1_hz=self.fp.value(),
            fz2_hz=self.fz2.value(), fp2_hz=self.fp2.value(), fz3_hz=self.fz3.value(), fp3_hz=self.fp3.value(),
            type_input_mode=self.type_input_mode.currentData(), r1_ohm=self.r1.value(), r2_ohm=self.r2.value(), r3_ohm=self.r3.value(),
            c1_f=self.c1_nf.value()*1e-9, c2_f=self.c2_nf.value()*1e-9, c3_f=self.c3_nf.value()*1e-9,
        )

    def _design(self):
        fs = self.fs.value()
        if self.design_pages.currentIndex() == 0:
            k = self.ctrl.currentData(); kwargs = self._controller_kwargs()
            if k == ControllerKind.GENERAL:
                kwargs['numerator'] = self._parse_coeff(self.general_num.text()); kwargs['denominator'] = self._parse_coeff(self.general_den.text())
            a = design_controller(k, **kwargs)
            method = self.method.currentData(); pw = self.prewarp.value() if method == DiscretizationMethod.PREWARP_TUSTIN else None
            d = discretize_transfer_function(a, fs, method, prewarp_frequency_hz=pw); return a, d
        response = self.response.currentData(); f2 = self.fc2.value() if response in (FilterResponse.BANDPASS, FilterResponse.BANDSTOP) else None
        impl = self.filter_impl.currentText()
        if impl == "IIR":
            result = design_iir_filter(response, self.family.currentData(), sample_rate_hz=fs, order=int(self.order.value()), f1_hz=self.fc.value(), f2_hz=f2, q=self.q.value(), passband_ripple_db=self.rp.value(), stopband_atten_db=self.rs.value())
        elif impl == "FIR Window":
            if response == FilterResponse.NOTCH:
                response = FilterResponse.BANDSTOP; f2 = max(self.fc2.value(), self.fc.value()*1.05)
            result = design_fir_filter(response, sample_rate_hz=fs, num_taps=int(self.fir_taps.value()), f1_hz=self.fc.value(), f2_hz=f2, window=self.fir_window.currentText())
        elif impl == "Moving Average": result = design_moving_average(sample_rate_hz=fs, length=int(self.fir_taps.value()))
        else: result = design_dc_blocker(sample_rate_hz=fs, pole_radius=self.dc_radius.value())
        return None, result.digital

    @staticmethod
    def _poly_text(coeffs, variable: str) -> str:
        vals = list(coeffs); degree = len(vals)-1; terms = []
        for i, c in enumerate(vals):
            power = degree-i
            if abs(float(c)) < 1e-18: continue
            if power == 0: term = f"{float(c):+.8g}"
            elif power == 1: term = f"{float(c):+.8g}{variable}"
            else: term = f"{float(c):+.8g}{variable}^{power}"
            terms.append(term)
        text = " ".join(terms) if terms else "0"; return text[1:].lstrip() if text.startswith('+') else text

    @staticmethod
    def _zinv_poly_text(coeffs) -> str:
        terms = []
        for i, c in enumerate(coeffs):
            if abs(float(c)) < 1e-18: continue
            if i == 0: term = f"{float(c):+.8g}"
            elif i == 1: term = f"{float(c):+.8g} z^-1"
            else: term = f"{float(c):+.8g} z^-{i}"
            terms.append(term)
        text = " ".join(terms) if terms else "0"; return text[1:].lstrip() if text.startswith('+') else text

    def _update_live_transfer(self, a, d):
        if a is None:
            self.live_hs.setPlainText("Direct digital filter — no continuous H(s) source")
        else:
            self.live_hs.setPlainText(f"H(s) = ({self._poly_text(a.numerator, 's')}) / ({self._poly_text(a.denominator, 's')})")
        self.live_hz.setPlainText(
            f"H(z) = ({self._zinv_poly_text(d.b)}) / ({self._zinv_poly_text(d.a)})\n"
            "y[n] = Σ b[k]x[n-k] − Σ a[k]y[n-k]"
        )

    def recalculate(self):
        try:
            self._update_dynamic_visibility(); a, d = self._design(); self._update_live_transfer(a, d)
            r = analyze_digital_filter(d, analog=a, response_samples=400)
            self.current_analog = a; self.current_digital = d; self.current_analysis = r; self._render()
            warn = []
            if d.max_pole_radius > 0.995: warn.append("Pole very close to unit circle: float32_t / transient robustness requires review.")
            critical = self.lpf_pole.value() if self.design_pages.currentIndex() == 0 and self.ctrl.currentData() in (ControllerKind.PIF, ControllerKind.PIDF) else self.fc.value()
            if critical > 0.2*d.sample_rate_hz: warn.append("Critical frequency > 0.2 Fs: frequency warping deserves review; consider Prewarped Tustin.")
            label = {StabilityClass.STABLE:"STABLE", StabilityClass.MARGINAL:"MARGINAL / INTEGRATOR", StabilityClass.UNSTABLE:"UNSTABLE"}[d.stability_class]
            if d.stability_class == StabilityClass.MARGINAL: warn.append("Pole on the unit circle is expected for ideal integral/derivative controllers; check closed-loop stability with the plant.")
            self.status.setText(label + f" | max |p|={d.max_pole_radius:.8f} | Nyquist={d.sample_rate_hz/2:.1f} Hz" + (("\n"+"\n".join(warn)) if warn else ""))
            self.export.setEnabled(d.implementable)
        except Exception as exc:
            self.live_hz.setPlainText("ERROR: " + str(exc)); self.status.setText("ERROR: " + str(exc)); self.export.setEnabled(False)

    def _render(self):
        r = self.current_analysis; d = self.current_digital; a = self.current_analog
        self.bode_fig.clear(); ax = self.bode_fig.add_subplot(211); ax2 = self.bode_fig.add_subplot(212, sharex=ax)
        ax.semilogx(r.frequency_hz, r.magnitude_db, label="H(z)"); ax2.semilogx(r.frequency_hz, r.phase_deg, label="H(z)")
        if r.analog_magnitude_db is not None:
            ax.semilogx(r.frequency_hz, r.analog_magnitude_db, label="H(s)"); ax2.semilogx(r.frequency_hz, r.analog_phase_deg, label="H(s)")
        ax.set_ylabel("Magnitude (dB)"); ax2.set_ylabel("Phase (deg)"); ax2.set_xlabel("Frequency (Hz)")
        ax.grid(True, which="both"); ax2.grid(True, which="both"); ax.legend(); ax2.legend(); self.bode_fig.tight_layout(); self.bode_canvas.draw_idle()

        self.step_fig.clear(); ax = self.step_fig.add_subplot(111); ax.plot(r.step_time_s*1e3, r.step); ax.set_xlabel("Time (ms)"); ax.set_ylabel("Amplitude"); ax.grid(True)
        ax.set_title(f"Step: overshoot={r.step_overshoot_percent:.2f}%, settling={r.settling_time_s*1e3:.3f} ms"); self.step_fig.tight_layout(); self.step_canvas.draw_idle()
        self.imp_fig.clear(); ax = self.imp_fig.add_subplot(111); ax.stem(r.impulse_time_s*1e3, r.impulse); ax.set_xlabel("Time (ms)"); ax.set_ylabel("Amplitude"); ax.grid(True); self.imp_fig.tight_layout(); self.imp_canvas.draw_idle()
        self.pz_fig.clear(); ax = self.pz_fig.add_subplot(111); th = np.linspace(0, 2*np.pi, 400); ax.plot(np.cos(th), np.sin(th), '--')
        if len(r.zeros): ax.plot(r.zeros.real, r.zeros.imag, 'o', label="Zero")
        if len(r.poles): ax.plot(r.poles.real, r.poles.imag, 'x', label="Pole")
        ax.axhline(0, linewidth=.8); ax.axvline(0, linewidth=.8); ax.set_aspect('equal', adjustable='box'); ax.grid(True); ax.legend(); ax.set_xlabel("Real"); ax.set_ylabel("Imag"); self.pz_fig.tight_layout(); self.pz_canvas.draw_idle()
        self.gd_fig.clear(); ax = self.gd_fig.add_subplot(111); ax.semilogx(r.frequency_hz, r.group_delay_s*1e6); ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Group delay (us)"); ax.grid(True, which="both"); self.gd_fig.tight_layout(); self.gd_canvas.draw_idle()

        lines = ["H(z) coefficient convention:", "H(z)=(b0+b1 z^-1+...)/(1+a1 z^-1+...)", "y[n]=Σ b[k]x[n-k] - Σ a[k]y[n-k]", ""]
        lines += [f"b{i} = {v:+.12e}" for i, v in enumerate(d.b)] + [f"a{i} = {v:+.12e}" for i, v in enumerate(d.a)]
        lines += ["", f"SOS =\n{d.sos}", "", f"Stability = {d.stability_class.value}", f"Max pole radius = {d.max_pole_radius:.12g}", f"DC gain = {r.dc_gain:.12g}"]
        self.coeff.setPlainText('\n'.join(lines))
        tf = ["DISCRETE:", f"H(z) = ({self._zinv_poly_text(d.b)}) / ({self._zinv_poly_text(d.a)})", f"Poles = {list(d.poles)}", f"Zeros = {list(d.zeros)}", "", f"SOS =\n{d.sos}"]
        if a is not None: tf = ["CONTINUOUS:", f"H(s) = ({self._poly_text(a.numerator, 's')}) / ({self._poly_text(a.denominator, 's')})", f"Numerator = {a.numerator}", f"Denominator = {a.denominator}", ""] + tf
        self.tf.setPlainText('\n'.join(tf))
        self.c99.setPlainText(render_c99_single_file(d, prefix=self.export_prefix.text().strip() or "POWER_CTRL"))

    def export_c99(self):
        if self.current_digital is None: return
        prefix = self.export_prefix.text().strip() or "POWER_CTRL"
        default = str(Path.cwd() / f"{prefix.lower()}.h")
        path, _ = QFileDialog.getSaveFileName(self, "导出单文件 C99", default, "C99 Header (*.h)")
        if not path: return
        if not path.lower().endswith('.h'): path += '.h'
        try:
            out = export_c99_filter(self.current_digital, path, prefix=prefix); verify = verify_c99_filter(self.current_digital, out)
            self.c99.setPlainText(out.file_path.read_text(encoding='utf-8') + f"\n/* Verification: {verify.message}; impulse={verify.impulse_max_abs_error:.3e}; step={verify.step_max_abs_error:.3e} */\n")
            QMessageBox.information(self, "C99 导出完成", f"单文件已生成:\n{out.file_path}\n\nVerification: {verify.message}")
        except Exception as exc:
            QMessageBox.critical(self, "C99 导出失败", str(exc))


__all__ = ["ControlToolsMainWindow"]
