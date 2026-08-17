"""Interactive transformer-design page for the LLC workspace."""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from ...core.spec import LLCDesignSpec
from ...magnetics.transformer_designer import (
    FerriteCoreInput,
    TransformerSynthesisResult,
    TransformerSynthesisSettings,
    load_transformer_core_presets,
)
from .. import theme


class TransformerWindingSketch(QWidget):
    """Compact visual explanation of the selected P/2-S-P/2 winding stack."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.result: TransformerSynthesisResult | None = None
        self.setMinimumHeight(220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_result(self, result: TransformerSynthesisResult | None) -> None:
        self.result = result
        self.update()

    def paintEvent(self, event):  # noqa: N802 - Qt API
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), QColor("#ffffff"))
        p.setPen(QPen(QColor("#d0d5dd"), 1))
        p.drawRoundedRect(self.rect().adjusted(1, 1, -2, -2), 10, 10)
        if self.result is None:
            p.setPen(QColor("#667085"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "运行变压器自动设计后显示绕组结构")
            return

        r = self.result
        p.setPen(QColor("#101828"))
        title_font = QFont(p.font()); title_font.setBold(True); title_font.setPointSize(title_font.pointSize() + 2)
        p.setFont(title_font)
        p.drawText(20, 30, f"{r.core.shape}  ·  {r.primary_turns}:{r.secondary_turns}  ·  {r.settings.winding_layout}")

        x0, y0 = 42, 62
        w = max(self.width() - 84, 260)
        h = max(self.height() - 116, 100)
        p.setPen(QPen(QColor("#475467"), 2))
        p.setBrush(QColor("#f2f4f7"))
        p.drawRoundedRect(x0, y0, w, h, 12, 12)

        bands = [
            ("P/2", max(r.primary_layers_per_half, 1), QColor("#dbeafe"), QColor("#175cd3")),
            ("S", max(r.secondary_layers, 1), QColor("#dcfae6"), QColor("#067647")),
            ("P/2", max(r.primary_layers_per_half, 1), QColor("#dbeafe"), QColor("#175cd3")),
        ]
        total = sum(v for _, v, _, _ in bands)
        inner_x = x0 + 18
        inner_w = w - 36
        cursor = inner_x
        normal_font = QFont(p.font()); normal_font.setBold(False); normal_font.setPointSize(max(8, normal_font.pointSize() - 1))
        for label, count, fill, edge in bands:
            bw = max(int(inner_w * count / total), 50)
            p.setPen(QPen(edge, 2)); p.setBrush(fill)
            p.drawRoundedRect(cursor, y0 + 18, bw - 6, h - 36, 7, 7)
            p.setPen(edge); p.setFont(title_font)
            p.drawText(cursor, y0 + 45, bw - 6, 22, Qt.AlignmentFlag.AlignCenter, label)
            p.setFont(normal_font)
            extra = (
                f"{r.primary_turns_per_layer} T/layer\n{r.primary_litz.strand_count}×{r.primary_litz.strand_diameter_mm:.2f} mm"
                if label.startswith("P") else
                f"{r.secondary_turns_per_layer} T/layer\n{r.secondary_litz.strand_count}×{r.secondary_litz.strand_diameter_mm:.2f} mm"
            )
            p.drawText(cursor + 4, y0 + 72, bw - 14, h - 88,
                       Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap,
                       extra)
            cursor += bw

        p.setFont(normal_font); p.setPen(QColor("#475467"))
        footer = (
            f"窗口填充 {r.fill_factor*100:.1f}%    |    等效径向构建 {r.radial_build_mm:.2f} mm    |    "
            f"Bpk(max) {r.worst_b_peak_t*1e3:.1f} mT    |    gap≈{r.estimated_gap_mm:.3f} mm"
        )
        p.drawText(24, self.height() - 22, footer)


class TransformerDesignView(QWidget):
    analysis_requested = Signal(object, object)  # FerriteCoreInput, TransformerSynthesisSettings
    apply_turns_requested = Signal(int, int)
    export_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._presets = load_transformer_core_presets()
        self.result: TransformerSynthesisResult | None = None
        self._build_ui()
        self._load_preset(next(iter(self._presets)))

    @staticmethod
    def _dspin(lo, hi, dec, suffix="") -> QDoubleSpinBox:
        w = QDoubleSpinBox(); w.setRange(lo, hi); w.setDecimals(dec); w.setSuffix(suffix)
        w.setKeyboardTracking(False); w.setMinimumWidth(120)
        return w

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel("LLC 变压器自动设计")
        title.setStyleSheet(f"font-size:18px;font-weight:700;color:{theme.active_theme().text_strong};")
        subtitle = QLabel("规格书磁芯参数 → 自动匝数 / 0.1 mm Litz → 绕组、磁芯损耗与工况校核")
        subtitle.setStyleSheet(f"color:{theme.active_theme().text_muted};")
        header.addWidget(title); header.addWidget(subtitle); header.addStretch(1)
        self.run_btn = QPushButton("自动设计变压器")
        self.run_btn.setStyleSheet("font-weight:600;padding:6px 16px;")
        self.run_btn.clicked.connect(self._request_analysis)
        self.apply_btn = QPushButton("应用匝数到 LLC 主设计")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self._apply_turns)
        self.export_btn = QPushButton("导出设计结果")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_requested.emit)
        header.addWidget(self.run_btn); header.addWidget(self.apply_btn); header.addWidget(self.export_btn)
        root.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_inputs())
        splitter.addWidget(self._build_results())
        splitter.setStretchFactor(0, 0); splitter.setStretchFactor(1, 1)
        splitter.setSizes([390, 1200])
        root.addWidget(splitter, 1)

    def _build_inputs(self) -> QWidget:
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setMinimumWidth(340); scroll.setMaximumWidth(520)
        body = QWidget(); lay = QVBoxLayout(body); lay.setContentsMargins(8, 6, 8, 10); lay.setSpacing(8)

        preset_box = QGroupBox("规格书 / 磁芯")
        f = QFormLayout(preset_box)
        self.preset = QComboBox()
        for key, item in self._presets.items():
            self.preset.addItem(f"{item.shape} {item.material_grade} · {item.part_number}", key)
        self.preset.addItem("自定义规格书参数", "CUSTOM")
        self.preset.currentIndexChanged.connect(self._preset_changed)
        f.addRow("预设", self.preset)

        self.core_fields = {}
        specs = [
            ("ae_mm2", "Ae", 1, 5000, 3, " mm²"),
            ("amin_mm2", "Amin", 1, 5000, 3, " mm²"),
            ("le_mm", "le", 1, 1000, 3, " mm"),
            ("ve_mm3", "Ve", 1, 2e6, 1, " mm³"),
            ("al_nh", "AL (ungapped)", 1, 100000, 1, " nH/N²"),
            ("mu_e", "µe", 1, 20000, 1, ""),
            ("winding_area_mm2", "AN / winding area", 1, 10000, 2, " mm²"),
            ("mean_turn_length_mm", "lN / MLT", 1, 1000, 2, " mm"),
            ("usable_winding_width_mm", "可用绕线宽度", 1, 200, 2, " mm"),
            ("ar_uohm", "AR (datasheet)", 0, 10000, 3, " µΩ"),
        ]
        for key, label, lo, hi, dec, suffix in specs:
            w = self._dspin(lo, hi, dec, suffix); self.core_fields[key] = w; f.addRow(label, w)
        self.material = QComboBox(); self.material.addItem("N87", "TDK_N87_REF"); self.material.addItem("N97", "TDK_N97_REF")
        f.addRow("损耗材料模型", self.material)
        lay.addWidget(preset_box)

        design = QGroupBox("自动绕组设计")
        f = QFormLayout(design)
        self.max_b = self._dspin(0.05, 0.40, 3, " T"); self.max_b.setValue(0.18)
        self.j_target = self._dspin(1, 20, 2, " A/mm²"); self.j_target.setValue(6.0)
        self.j_max = self._dspin(1, 30, 2, " A/mm²"); self.j_max.setValue(8.0)
        self.strand_d = self._dspin(0.02, 0.50, 3, " mm"); self.strand_d.setValue(0.10)
        self.strand_do = self._dspin(0.02, 0.80, 3, " mm"); self.strand_do.setValue(0.112)
        self.strand_step = QSpinBox(); self.strand_step.setRange(10, 500); self.strand_step.setSingleStep(10); self.strand_step.setValue(50)
        self.fill_limit = self._dspin(0.1, 0.95, 3); self.fill_limit.setValue(0.60)
        self.insulation_area = self._dspin(0, 500, 2, " mm²"); self.insulation_area.setValue(28.0)
        self.nominal_gain = self._dspin(0.5, 1.5, 4); self.nominal_gain.setValue(1.0)
        self.workpoint_scope = QComboBox()
        self.workpoint_scope.addItem("全范围（含 Hold-up + 轻载）", "all")
        self.workpoint_scope.addItem("正常母线范围（不含 Hold-up）", "normal")
        self.workpoint_scope.addItem("仅标称满载", "nominal")
        for label, w in (
            ("Bmax 设计限制", self.max_b),
            ("目标电流密度", self.j_target),
            ("最大电流密度", self.j_max),
            ("Litz 单股铜径", self.strand_d),
            ("单股含漆外径", self.strand_do),
            ("股数步进", self.strand_step),
            ("最大窗口填充", self.fill_limit),
            ("绝缘占用面积", self.insulation_area),
            ("标称点目标增益", self.nominal_gain),
            ("设计工况范围", self.workpoint_scope),
        ):
            f.addRow(label, w)
        note = QLabel("Litz 总股数自动取 50、100、150… 的倍数（步进可改）；高电流绕组会自动提示并束数量。")
        note.setWordWrap(True); note.setStyleSheet(f"color:{theme.active_theme().text_muted};")
        lay.addWidget(design); lay.addWidget(note)
        lay.addStretch(1)
        scroll.setWidget(body)
        return scroll

    def _build_results(self) -> QWidget:
        box = QWidget(); lay = QVBoxLayout(box); lay.setContentsMargins(4, 0, 0, 0); lay.setSpacing(6)
        self.status = QLabel("等待自动设计")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("padding:8px 10px;background:#f2f4f7;border:1px solid #d0d5dd;border-radius:6px;")
        lay.addWidget(self.status)

        self.result_tabs = QTabWidget(); self.result_tabs.setDocumentMode(True)
        overview = QWidget(); ov = QHBoxLayout(overview); ov.setContentsMargins(8, 8, 8, 8)
        self.summary = QTextEdit(); self.summary.setReadOnly(True); self.summary.setMinimumWidth(380)
        self.sketch = TransformerWindingSketch()
        ov.addWidget(self.summary, 2); ov.addWidget(self.sketch, 3)
        self.result_tabs.addTab(overview, "设计结果 / 绕组")

        loss_page = QWidget(); lv = QVBoxLayout(loss_page); lv.setContentsMargins(4, 4, 4, 4)
        self.loss_figure = Figure(figsize=(9, 6)); self.loss_canvas = FigureCanvasQTAgg(self.loss_figure)
        lv.addWidget(self.loss_canvas)
        self.result_tabs.addTab(loss_page, "损耗分解")

        self.work_table = QTableWidget(0, 9)
        self.work_table.setHorizontalHeaderLabels([
            "Vbus", "Load", "Fsw", "Bpk", "Ip RMS", "Is RMS", "Core W", "Cu W", "Total W"
        ])
        self.work_table.setAlternatingRowColors(True)
        self.work_table.setSortingEnabled(False)
        self.result_tabs.addTab(self.work_table, "工作点")
        lay.addWidget(self.result_tabs, 1)
        return box

    def _preset_changed(self) -> None:
        key = self.preset.currentData()
        if key != "CUSTOM":
            self._load_preset(key)

    def _load_preset(self, key: str) -> None:
        p = self._presets[key]
        idx = self.preset.findData(key)
        if idx >= 0 and self.preset.currentIndex() != idx:
            self.preset.blockSignals(True); self.preset.setCurrentIndex(idx); self.preset.blockSignals(False)
        for name, widget in self.core_fields.items():
            widget.setValue(float(getattr(p, name)))
        midx = self.material.findData(p.material_key)
        if midx >= 0: self.material.setCurrentIndex(midx)

    def _core_input(self) -> FerriteCoreInput:
        key = self.preset.currentData()
        base = self._presets.get(key, next(iter(self._presets.values())))
        changes = {name: widget.value() for name, widget in self.core_fields.items()}
        changes.update({
            "preset_key": str(key),
            "material_key": self.material.currentData(),
            "material_grade": self.material.currentText(),
        })
        return replace(base, **changes)

    def _settings(self) -> TransformerSynthesisSettings:
        return TransformerSynthesisSettings(
            nominal_tank_gain=self.nominal_gain.value(),
            max_flux_density_t=self.max_b.value(),
            strand_copper_diameter_mm=self.strand_d.value(),
            strand_outer_diameter_mm=self.strand_do.value(),
            strand_count_step=self.strand_step.value(),
            current_density_target_a_per_mm2=self.j_target.value(),
            current_density_max_a_per_mm2=self.j_max.value(),
            max_fill_factor=self.fill_limit.value(),
            insulation_area_mm2=self.insulation_area.value(),
            workpoint_scope=self.workpoint_scope.currentData(),
        )

    def _request_analysis(self) -> None:
        self.analysis_requested.emit(self._core_input(), self._settings())

    def set_busy(self, busy: bool) -> None:
        self.run_btn.setEnabled(not busy)
        self.apply_btn.setEnabled((not busy) and self.result is not None)
        self.export_btn.setEnabled((not busy) and self.result is not None)

    def set_result(self, result: TransformerSynthesisResult) -> None:
        self.result = result; self.apply_btn.setEnabled(True); self.export_btn.setEnabled(True); self.sketch.set_result(result)
        state = "PASS" if result.feasible else "CHECK"
        color = "#067647" if result.feasible else "#b54708"
        self.status.setText(
            f"{state} · 推荐 {result.primary_turns}:{result.secondary_turns} · "
            f"Primary {result.primary_litz.strand_count}×{result.primary_litz.strand_diameter_mm:.2f} mm · "
            f"Secondary {result.secondary_litz.strand_count}×{result.secondary_litz.strand_diameter_mm:.2f} mm · "
            f"Nominal transformer loss {result.total_nominal_loss_w:.2f} W"
        )
        self.status.setStyleSheet(
            f"padding:8px 10px;background:#f8fafc;border:1px solid #d0d5dd;border-left:5px solid {color};border-radius:6px;"
        )
        self._fill_summary(result); self._plot_loss(result); self._fill_table(result)

    def _fill_summary(self, r: TransformerSynthesisResult) -> None:
        loss = r.nominal_loss
        lines = [
            "自动设计结果", "=" * 58,
            f"磁芯: {r.core.shape} / {r.core.material_grade} / {r.core.part_number}",
            f"Ae/Amin/le/Ve: {r.core.ae_mm2:.1f} / {r.core.amin_mm2:.1f} mm² / {r.core.le_mm:.1f} mm / {r.core.ve_mm3:.0f} mm³",
            f"AN={r.core.winding_area_mm2:.1f} mm², lN={r.core.mean_turn_length_mm:.1f} mm, AL={r.core.al_nh:.0f} nH/N², µe={r.core.mu_e:.0f}",
            "",
            f"目标/实际匝比: {r.target_turns_ratio:.5f} / {r.actual_turns_ratio:.5f}  (error {r.turns_ratio_error_pct:+.3f}%)",
            f"推荐匝数: Np={r.primary_turns}, Ns={r.secondary_turns}",
            f"Lm target={r.target_lm_uh:.2f} µH, ungapped estimate={r.ungapped_lm_uh:.2f} µH",
            f"AL target={r.target_al_nh:.2f} nH/N², estimated total gap={r.estimated_gap_mm:.3f} mm",
            f"Worst Bpk={r.worst_b_peak_t*1e3:.2f} mT",
            "",
            f"Primary Litz: {r.primary_litz.description}, J={r.primary_litz.current_density_a_per_mm2:.2f} A/mm², Rdc={r.primary_rdc_mohm:.3f} mΩ",
            f"Secondary Litz: {r.secondary_litz.description}, J={r.secondary_litz.current_density_a_per_mm2:.2f} A/mm², Rdc={r.secondary_rdc_mohm:.3f} mΩ",
            f"P/2 layers={r.primary_layers_per_half}, S layers={r.secondary_layers}, fill={r.fill_factor*100:.1f}%, radial build≈{r.radial_build_mm:.2f} mm",
            "",
            f"Nominal core loss: {loss.core_w:.3f} W",
            f"Nominal primary copper: {loss.primary_copper_w:.3f} W (Rac/Rdc={loss.primary_ac_factor:.2f})",
            f"Nominal secondary copper: {loss.secondary_copper_w:.3f} W (Rac/Rdc={loss.secondary_ac_factor:.2f})",
            f"Nominal total transformer loss: {loss.total_w:.3f} W",
            f"Estimated hotspot: {loss.estimated_hotspot_c:.1f} °C",
            f"Datasheet Pv reference: < {r.core.datasheet_loss_ref_w:.2f} W/set @ {r.core.datasheet_loss_ref_b_t*1e3:.0f} mT, {r.core.datasheet_loss_ref_frequency_hz/1e3:.0f} kHz, {r.core.datasheet_loss_ref_temperature_c:.0f} °C",
        ]
        if r.reasons:
            lines += ["", "需要检查:"] + [f"- {x}" for x in r.reasons]
        if r.warnings:
            lines += ["", "工程提示:"] + [f"- {x}" for x in r.warnings]
        self.summary.setPlainText("\n".join(lines))

    def _plot_loss(self, r: TransformerSynthesisResult) -> None:
        fig = self.loss_figure; fig.clear()
        ax = fig.add_subplot(111)
        l = r.nominal_loss
        labels = ["Core", "P-DC", "P-Skin", "P-Prox", "P-Bundle", "S-DC", "S-Skin", "S-Prox", "S-Bundle"]
        values = [l.core_w, l.primary_dc_w, l.primary_skin_w, l.primary_proximity_w,
                  l.primary_bundle_w + l.primary_termination_w, l.secondary_dc_w,
                  l.secondary_skin_w, l.secondary_proximity_w,
                  l.secondary_bundle_w + l.secondary_termination_w]
        ax.bar(labels, values)
        ax.set_ylabel("Loss (W)"); ax.set_title("Nominal transformer loss breakdown")
        ax.grid(True, axis="y", alpha=0.25); ax.tick_params(axis="x", rotation=28)
        fig.tight_layout(); self.loss_canvas.draw_idle()

    def _fill_table(self, r: TransformerSynthesisResult) -> None:
        self.work_table.setRowCount(len(r.workpoints))
        for row, w in enumerate(r.workpoints):
            vals = [
                f"{w.vbus_v:.0f} V", f"{w.load_fraction*100:.0f}%", f"{w.switching_frequency_hz/1e3:.2f} kHz",
                f"{w.b_peak_t*1e3:.1f} mT", f"{w.primary_rms_a:.2f} A", f"{w.secondary_rms_a:.2f} A",
                f"{w.core_loss_w:.3f}", f"{w.primary_copper_w+w.secondary_copper_w:.3f}", f"{w.total_transformer_loss_w:.3f}",
            ]
            for col, val in enumerate(vals): self.work_table.setItem(row, col, QTableWidgetItem(val))
        self.work_table.resizeColumnsToContents()

    def _apply_turns(self) -> None:
        if self.result is not None:
            self.apply_turns_requested.emit(self.result.primary_turns, self.result.secondary_turns)

    def set_nominal_spec(self, spec: LLCDesignSpec) -> None:
        # Keep transformer-specific settings independent; the current LLC spec is
        # consumed by the worker in the parent main window.
        pass
