"""Reusable TTPL/Vienna PFC boost-inductor design widgets."""
from __future__ import annotations

from dataclasses import replace
from typing import Callable

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
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from pfc_design.magnetics import (
    HIGH_FLUX_254,
    HIGH_FLUX_254_MATERIALS,
    HighFluxCoreGeometry,
    PFCInductorDesignRequest,
    PFCInductorDesignResult,
    design_pfc_inductor,
)
from llc_design.gui import theme


class PFCInductorDesignEditor(QWidget):
    """Narrow parameter editor suitable for the PFC left inspector."""

    design_completed = Signal(object)

    def __init__(
        self,
        topology: str,
        context_provider: Callable[[], dict[str, float]],
        apply_callback: Callable[[PFCInductorDesignResult], None],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.topology = topology
        self.context_provider = context_provider
        self.apply_callback = apply_callback
        self.last_result: PFCInductorDesignResult | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(7)

        intro = QLabel(
            "High Flux 粉芯电感自动设计。默认磁芯为 Magnetics Core Data 254；"
            "漆包圆铜线，铜损默认仅计算 DC I²R，不计趋肤/邻近效应。"
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color:{theme.active_theme().text_muted};padding:4px 2px;")
        root.addWidget(intro)

        op_group = QGroupBox("工作点 / 目标")
        op = QFormLayout(op_group)
        self.context_label = QLabel("尚未同步")
        self.context_label.setWordWrap(True)
        self.target_l = self._spin(1.0, 20000.0, 2, 220.0, " µH")
        self.eff = self._spin(0.80, 1.0, 4, 0.97)
        self.n_cores = QSpinBox(); self.n_cores.setRange(1, 16)
        self.n_cores.setValue(2 if topology == "ttpl" else 5)
        op.addRow("功率级", self.context_label)
        op.addRow("满载目标 L", self.target_l)
        op.addRow("估算效率 η", self.eff)
        op.addRow("叠加磁芯数", self.n_cores)
        root.addWidget(op_group)

        core_group = QGroupBox("磁芯 — Magnetics High Flux 254")
        core = QFormLayout(core_group)
        self.perm = QComboBox()
        for mu in sorted(HIGH_FLUX_254_MATERIALS):
            self.perm.addItem(f"{mu} µ", mu)
        self.perm.setCurrentIndex(self.perm.findData(60))
        self.al = self._spin(0.01, 5000.0, 3, 81.0, " nH/T²")
        self.le = self._spin(1.0, 1000.0, 2, HIGH_FLUX_254.le_mm, " mm")
        self.ae = self._spin(0.1, 10000.0, 2, HIGH_FLUX_254.ae_mm2, " mm²")
        self.ve = self._spin(1.0, 1e7, 1, HIGH_FLUX_254.ve_mm3, " mm³")
        self.od = self._spin(1.0, 500.0, 2, HIGH_FLUX_254.od_mm, " mm")
        self.id_ = self._spin(0.1, 499.0, 2, HIGH_FLUX_254.id_mm, " mm")
        self.ht = self._spin(0.1, 200.0, 2, HIGH_FLUX_254.ht_mm, " mm")
        self.bsat = self._spin(0.1, 3.0, 3, HIGH_FLUX_254.bs_t, " T")
        for label, widget in (
            ("材料磁导率", self.perm), ("AL", self.al), ("Le", self.le),
            ("Ae", self.ae), ("Ve", self.ve), ("OD", self.od),
            ("ID", self.id_), ("HT", self.ht), ("Bsat", self.bsat),
        ):
            core.addRow(label, widget)
        root.addWidget(core_group)
        self.perm.currentIndexChanged.connect(self._material_changed)

        wire_group = QGroupBox("绕组 — 漆包圆铜线")
        wire = QFormLayout(wire_group)
        self.wire_d = self._spin(0.10, 5.0, 3, 1.00, " mm")
        self.enamel = self._spin(0.0, 0.5, 3, 0.05, " mm/side")
        self.j_target = self._spin(0.5, 20.0, 2, 5.0, " A/mm²")
        self.cu_temp = self._spin(20.0, 180.0, 1, 100.0, " °C")
        self.fill = self._spin(0.10, 0.80, 3, 0.45)
        for label, widget in (
            ("单根铜径", self.wire_d), ("漆膜单边厚度", self.enamel),
            ("目标电流密度", self.j_target), ("铜温", self.cu_temp),
            ("窗口填充上限", self.fill),
        ):
            wire.addRow(label, widget)
        root.addWidget(wire_group)

        buttons = QHBoxLayout()
        self.sync_button = QPushButton("从功率级同步")
        self.run_button = QPushButton("计算电感")
        self.run_button.setDefault(True)
        buttons.addWidget(self.sync_button)
        buttons.addWidget(self.run_button)
        root.addLayout(buttons)
        self.apply_button = QPushButton("应用 L / DCR 到功率级")
        self.apply_button.setEnabled(False)
        root.addWidget(self.apply_button)

        self.status = QLabel("等待计算")
        self.status.setWordWrap(True)
        _t = theme.active_theme()
        self.status.setStyleSheet(
            f"padding:6px;border:1px solid {_t.border_card};border-radius:5px;background:{_t.card_bg};color:{_t.text};"
        )
        root.addWidget(self.status)
        root.addStretch(1)

        self.sync_button.clicked.connect(self.sync_from_stage)
        self.run_button.clicked.connect(self.calculate)
        self.apply_button.clicked.connect(self.apply_to_stage)
        self.sync_from_stage()

    @staticmethod
    def _spin(lo: float, hi: float, dec: int, value: float, suffix: str = "") -> QDoubleSpinBox:
        w = QDoubleSpinBox()
        w.setRange(lo, hi); w.setDecimals(dec); w.setValue(value); w.setSuffix(suffix)
        w.setKeyboardTracking(False)
        return w

    def _material_changed(self) -> None:
        mu = int(self.perm.currentData())
        self.al.setValue(HIGH_FLUX_254_MATERIALS[mu].al_nh_per_t2)

    def sync_from_stage(self) -> None:
        context = self.context_provider()
        self.target_l.setValue(float(context["target_inductance_uh"]))
        if "recommended_core_count" in context:
            self.n_cores.setValue(int(context["recommended_core_count"]))
        label = (
            f"Vin={context['input_rms_v']:.4g} V · Vbus={context['bus_voltage_v']:.4g} V\n"
            f"P={context['output_power_w']:.5g} W · fs={context['switching_frequency_hz']/1e3:.4g} kHz"
        )
        self.context_label.setText(label)

    def _request(self) -> PFCInductorDesignRequest:
        context = self.context_provider()
        mu = int(self.perm.currentData())
        material = replace(HIGH_FLUX_254_MATERIALS[mu], al_nh_per_t2=self.al.value())
        core = HighFluxCoreGeometry(
            core_data="254/custom",
            le_mm=self.le.value(), ae_mm2=self.ae.value(), ve_mm3=self.ve.value(),
            od_mm=self.od.value(), id_mm=self.id_.value(), ht_mm=self.ht.value(),
            bs_t=self.bsat.value(),
        )
        return PFCInductorDesignRequest(
            topology=self.topology,
            input_rms_v=float(context["input_rms_v"]),
            bus_voltage_v=float(context["bus_voltage_v"]),
            output_power_w=float(context["output_power_w"]),
            switching_frequency_hz=float(context["switching_frequency_hz"]),
            target_inductance_uh=self.target_l.value(),
            efficiency=self.eff.value(),
            core=core,
            material=material,
            n_cores=self.n_cores.value(),
            wire_copper_diameter_mm=self.wire_d.value(),
            enamel_build_mm=self.enamel.value(),
            target_current_density_a_mm2=self.j_target.value(),
            copper_temperature_c=self.cu_temp.value(),
            max_fill_factor=self.fill.value(),
        )

    def calculate(self) -> None:
        try:
            result = design_pfc_inductor(self._request())
        except Exception as exc:
            QMessageBox.warning(self, "PFC 电感设计失败", str(exc))
            return
        self.last_result = result
        self.apply_button.setEnabled(True)
        state = "PASS" if result.inductance_target_met and result.window_ok else "CHECK"
        self.status.setText(
            f"{state} · N={result.turns}T · {result.parallel_wires}×{result.request.wire_copper_diameter_mm:.3g}mm · "
            f"L(full)={result.l_full_load_peak_uh:.3f}µH · PΣ={result.total_loss_w:.3f}W"
        )
        self.status.setStyleSheet(
            "padding:6px;border:1px solid #75e0a7;border-radius:5px;background:#ecfdf3;color:#067647;"
            if state == "PASS" else
            "padding:6px;border:1px solid #fec84b;border-radius:5px;background:#fffaeb;color:#b54708;"
        )
        self.design_completed.emit(result)

    def apply_to_stage(self) -> None:
        if self.last_result is None:
            return
        self.apply_callback(self.last_result)


class PFCInductorDesignResultView(QWidget):
    """Summary + L(I), line ripple/core-loss and loss-breakdown plots."""

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self.title = title
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(5)
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(190)
        self.summary.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.summary.setPlainText("尚未计算电感。左侧进入“电感设计”，然后点击“计算电感”。")
        root.addWidget(self.summary)
        self.figure = Figure(figsize=(11, 7))
        self.canvas = FigureCanvasQTAgg(self.figure)
        root.addWidget(self.canvas, 1)

    def set_result(self, result: PFCInductorDesignResult) -> None:
        req = result.request
        warn = "\n".join(f"  - {w}" for w in result.warnings)
        self.summary.setPlainText(
            f"{self.title}\n"
            f"Core      : Magnetics High Flux 254/custom, μ={req.material.permeability}, AL={req.material.al_nh_per_t2:.3f} nH/T², cores={req.n_cores}\n"
            f"Geometry  : Le={req.core.le_mm:.3f} mm, Ae={req.core.ae_mm2:.3f} mm², Ve={req.core.ve_mm3:.1f} mm³, OD/ID/HT={req.core.od_mm:.2f}/{req.core.id_mm:.2f}/{req.core.ht_mm:.2f} mm\n"
            f"Winding   : N={result.turns} T, {result.parallel_wires} × Ø{req.wire_copper_diameter_mm:.3f} mm enamel Cu, Cu area={result.copper_area_mm2:.3f} mm², fill={100*result.fill_factor:.2f}%\n"
            f"Inductance: L0={result.l_no_bias_uh:.3f} µH, L@full-peak={result.l_full_load_peak_uh:.3f} µH, drop={result.l_drop_percent:.2f}%\n"
            f"Current   : Iphase,rms={result.phase_current_rms_a:.3f} A, Irms+ripple={result.total_current_rms_with_ripple_a:.3f} A, Ipk={result.full_load_peak_with_ripple_a:.3f} A\n"
            f"Bias      : Hpk={result.h_peak_oe:.3f} Oe, μ/μi={result.permeability_at_peak_percent:.2f}%, Bdc≈{result.b_dc_approx_peak_t:.4f} T, Bac,max={result.b_ac_line_max_t:.4f} T\n"
            f"Copper    : R20={1e3*result.rdc_20_ohm:.3f} mΩ, Rhot={1e3*result.rdc_hot_ohm:.3f} mΩ, J={result.current_density_a_mm2:.3f} A/mm², Pcu={result.copper_loss_w:.3f} W\n"
            f"Core loss : {result.core_loss_w:.3f} W (Magnetics High Flux fit, line-cycle average)\n"
            f"Total loss: {result.total_loss_w:.3f} W\n"
            f"Warnings:\n{warn}"
        )

        self.figure.clear()
        ax_l = self.figure.add_subplot(221)
        ax_ripple = self.figure.add_subplot(222)
        ax_core = self.figure.add_subplot(223)
        ax_loss = self.figure.add_subplot(224)

        ax_l.plot(result.current_a, result.inductance_uh, linewidth=2.0)
        ax_l.axhline(req.target_inductance_uh, linestyle="--", linewidth=1.0, label="Target L")
        ax_l.scatter([result.full_load_peak_with_ripple_a], [result.l_full_load_peak_uh], s=35, zorder=5, label="Full-load peak")
        ax_l.set_title("Inductance droop vs current")
        ax_l.set_xlabel("Current (A)"); ax_l.set_ylabel("L (µH)"); ax_l.grid(True, alpha=.3); ax_l.legend(fontsize=8)

        ax_ripple.plot(result.line_angle_deg, result.line_ripple_pp_a)
        ax_ripple.set_title("Switching ripple over line cycle")
        ax_ripple.set_xlabel("Line angle (deg)"); ax_ripple.set_ylabel("ΔIpp (A)"); ax_ripple.grid(True, alpha=.3)

        ax_core.plot(result.line_angle_deg, result.line_core_loss_w, label="Core loss")
        ax_core2 = ax_core.twinx()
        ax_core2.plot(result.line_angle_deg, result.line_b_ac_t, linestyle="--", label="Bac")
        ax_core.set_title("Core loss / AC flux over line cycle")
        ax_core.set_xlabel("Line angle (deg)"); ax_core.set_ylabel("Core loss (W)"); ax_core2.set_ylabel("Bac (T)")
        ax_core.grid(True, alpha=.3)

        labels = ["Core", "Copper", "Total"]
        values = [result.core_loss_w, result.copper_loss_w, result.total_loss_w]
        ax_loss.bar(labels, values)
        ax_loss.set_title("Loss breakdown")
        ax_loss.set_ylabel("Loss (W)"); ax_loss.grid(True, axis="y", alpha=.3)
        for idx, value in enumerate(values):
            ax_loss.text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

        self.figure.suptitle(self.title)
        self.figure.tight_layout()
        self.canvas.draw_idle()


__all__ = ["PFCInductorDesignEditor", "PFCInductorDesignResultView"]
