"""Selectable PFC Bode panel with per-transfer-function visibility controls."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from llc_design.control.digital_loop import StabilityMargins
from llc_design.gui.widgets.bode_cursor import (
    BodeCursor,
    BodeCursorMeasurement,
    BodeCursorTrace,
)
from llc_design.gui import theme


@dataclass(frozen=True)
class PFCBodeCurve:
    """One selectable transfer-function trace."""

    key: str
    label: str
    response: np.ndarray
    default_visible: bool = False
    is_open_loop: bool = False


class SelectableBodePanel(QWidget):
    """Bode chart whose individual transfer functions can be enabled or hidden.

    PFC stability analysis defaults to the open-loop transfer only.  Plant,
    controller, sensing, ZOH, closed-loop and sensitivity responses remain
    available through explicit check boxes without cluttering the initial view.
    """

    cursor_changed = Signal(object)

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self.title = title
        self.figure = Figure(figsize=(10, 7))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self._frequencies_hz: np.ndarray | None = None
        self._curves: list[PFCBodeCurve] = []
        self._checkboxes: dict[str, QCheckBox] = {}
        self._cursor: BodeCursor | None = None
        self._cursor_frequency_hz: float | None = None
        self._margins: StabilityMargins | None = None
        self._open_loop_key: str | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(5)

        selector_bar = QHBoxLayout()
        self.selector_toggle = QPushButton("传递函数 ▸")
        self.selector_toggle.setCheckable(True)
        self.selector_toggle.setChecked(False)
        self.selector_toggle.setToolTip("展开/收起逐传递函数复选框；默认折叠以给 Bode 图更多垂直空间")
        selector_bar.addWidget(self.selector_toggle)
        selector_note = QLabel("默认仅显示系统开环；需要分解时再展开")
        selector_note.setStyleSheet(f"color:{theme.active_theme().text_muted};")
        selector_bar.addWidget(selector_note)
        selector_bar.addStretch(1)
        root.addLayout(selector_bar)

        self.selector_group = QGroupBox("传递函数显示")
        selector_root = QVBoxLayout(self.selector_group)
        self.selector_grid = QGridLayout()
        selector_root.addLayout(self.selector_grid)

        buttons = QHBoxLayout()
        self.open_only_button = QPushButton("仅开环")
        self.show_all_button = QPushButton("全部显示")
        self.hide_all_button = QPushButton("全部关闭")
        self.open_only_button.clicked.connect(self.show_open_loop_only)
        self.show_all_button.clicked.connect(lambda: self._set_all(True))
        self.hide_all_button.clicked.connect(lambda: self._set_all(False))
        buttons.addWidget(self.open_only_button)
        buttons.addWidget(self.show_all_button)
        buttons.addWidget(self.hide_all_button)
        buttons.addStretch(1)
        selector_root.addLayout(buttons)
        root.addWidget(self.selector_group)
        self.selector_group.setVisible(False)
        self.selector_toggle.toggled.connect(self._toggle_selector_group)

        self.empty_hint = QLabel("尚未运行分析。")
        self.empty_hint.setWordWrap(True)
        root.addWidget(self.empty_hint)
        root.addWidget(self.canvas, 1)

    def _toggle_selector_group(self, visible: bool) -> None:
        self.selector_group.setVisible(bool(visible))
        self.selector_toggle.setText("传递函数 ▾" if visible else "传递函数 ▸")

    @property
    def visible_keys(self) -> tuple[str, ...]:
        return tuple(
            curve.key for curve in self._curves
            if self._checkboxes.get(curve.key) is not None
            and self._checkboxes[curve.key].isChecked()
        )

    def set_cursor_frequency(self, frequency_hz: float | None) -> None:
        self._cursor_frequency_hz = frequency_hz
        if self._cursor is not None and frequency_hz is not None:
            self._cursor.set_frequency(frequency_hz)

    def set_curves(
        self,
        frequencies_hz: np.ndarray,
        curves: Iterable[PFCBodeCurve],
        *,
        margins: StabilityMargins | None = None,
        open_loop_key: str | None = None,
    ) -> None:
        frequencies = np.asarray(frequencies_hz, dtype=float)
        if frequencies.ndim != 1 or len(frequencies) < 2:
            raise ValueError("Bode frequency array must contain at least two points")
        curve_list = list(curves)
        if not curve_list:
            raise ValueError("at least one Bode curve is required")
        for curve in curve_list:
            response = np.asarray(curve.response, dtype=complex)
            if response.shape != frequencies.shape:
                raise ValueError(f"{curve.key}: response shape does not match frequency grid")

        previous = {key: box.isChecked() for key, box in self._checkboxes.items()}
        self._frequencies_hz = frequencies
        self._curves = curve_list
        self._margins = margins
        self._open_loop_key = open_loop_key
        self.open_only_button.setEnabled(open_loop_key is not None)
        self._rebuild_selectors(previous)
        self._redraw()

    def _clear_selector_grid(self) -> None:
        while self.selector_grid.count():
            item = self.selector_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._checkboxes.clear()

    def _rebuild_selectors(self, previous: dict[str, bool]) -> None:
        self._clear_selector_grid()
        columns = 3
        for index, curve in enumerate(self._curves):
            checkbox = QCheckBox(curve.label)
            checkbox.setChecked(previous.get(curve.key, curve.default_visible))
            checkbox.toggled.connect(self._redraw)
            self._checkboxes[curve.key] = checkbox
            self.selector_grid.addWidget(checkbox, index // columns, index % columns)


    def focus_curve(self, key: str, *, keep_open_loop: bool = True) -> None:
        """Show one selected block transfer together with the system open loop."""
        if key not in self._checkboxes:
            return
        for curve in self._curves:
            box = self._checkboxes[curve.key]
            box.blockSignals(True)
            visible = curve.key == key
            if keep_open_loop and (curve.is_open_loop or curve.key == self._open_loop_key):
                visible = True
            box.setChecked(visible)
            box.blockSignals(False)
        self._redraw()

    def set_curve_visible(self, key: str, visible: bool) -> None:
        box = self._checkboxes.get(key)
        if box is None:
            return
        box.setChecked(bool(visible))

    def show_open_loop_only(self) -> None:
        for curve in self._curves:
            box = self._checkboxes[curve.key]
            box.blockSignals(True)
            box.setChecked(curve.is_open_loop or curve.key == self._open_loop_key)
            box.blockSignals(False)
        self._redraw()

    def _set_all(self, visible: bool) -> None:
        for checkbox in self._checkboxes.values():
            checkbox.blockSignals(True)
            checkbox.setChecked(visible)
            checkbox.blockSignals(False)
        self._redraw()

    def _disconnect_cursor(self) -> None:
        if self._cursor is not None:
            self._cursor.disconnect()
            self._cursor = None

    def _redraw(self, *_args) -> None:
        self._disconnect_cursor()
        self.figure.clear()
        ax_mag = self.figure.add_subplot(211)
        ax_phase = self.figure.add_subplot(212, sharex=ax_mag)

        if self._frequencies_hz is None:
            self.empty_hint.setText("尚未运行分析。")
            self.canvas.draw_idle()
            return

        visible = [
            curve for curve in self._curves
            if self._checkboxes[curve.key].isChecked()
        ]
        traces: list[BodeCursorTrace] = []
        for curve in visible:
            response = np.asarray(curve.response, dtype=complex)
            gain_db = 20.0 * np.log10(np.maximum(np.abs(response), 1e-300))
            phase_deg = np.unwrap(np.angle(response)) * 180.0 / math.pi
            mag_line, = ax_mag.semilogx(
                self._frequencies_hz,
                gain_db,
                label=curve.label,
                linewidth=2.0 if curve.is_open_loop else 1.25,
            )
            ax_phase.semilogx(
                self._frequencies_hz,
                phase_deg,
                label=curve.label,
                color=mag_line.get_color(),
                linewidth=2.0 if curve.is_open_loop else 1.25,
            )
            traces.append(BodeCursorTrace(
                curve.label,
                self._frequencies_hz,
                gain_db,
                phase_deg,
                mag_line.get_color(),
            ))

        ax_mag.axhline(0.0, linestyle="--", linewidth=0.8)
        ax_phase.axhline(-180.0, linestyle="--", linewidth=0.8)
        ax_mag.set_ylabel("Gain (dB)")
        ax_phase.set_ylabel("Phase (deg)")
        ax_phase.set_xlabel("Frequency (Hz)")
        for axis in (ax_mag, ax_phase):
            axis.grid(True, which="both", alpha=0.3)

        if visible:
            ax_mag.legend(fontsize=8, ncol=min(3, max(1, len(visible))))
            ax_phase.legend(fontsize=8, ncol=min(3, max(1, len(visible))))
            self.empty_hint.setText(
                "单击或拖动图中竖直光标读取 Gain / Phase；上方复选框控制每条传递函数。"
            )
        else:
            ax_mag.text(
                0.5, 0.5, "未选择传递函数",
                transform=ax_mag.transAxes,
                ha="center", va="center",
            )
            self.empty_hint.setText("当前全部关闭。勾选任意传递函数后显示。")

        self._draw_margin_marker(ax_mag, ax_phase, visible)
        self.figure.suptitle(self.title)
        self.figure.tight_layout()
        self.canvas.draw_idle()

        if traces:
            self._cursor = BodeCursor(
                self.canvas,
                ax_mag,
                ax_phase,
                traces,
                initial_frequency_hz=self._cursor_frequency_hz,
                on_changed=self._on_cursor_changed,
            )
            self.canvas.draw_idle()

    def _draw_margin_marker(self, ax_mag, ax_phase, visible: list[PFCBodeCurve]) -> None:
        if self._margins is None or self._open_loop_key is None:
            return
        if not any(curve.key == self._open_loop_key for curve in visible):
            return
        crossover = self._margins.critical_gain_crossover_hz
        if crossover is None or not np.isfinite(crossover):
            return
        ax_mag.axvline(crossover, linestyle=":", linewidth=1.1)
        ax_phase.axvline(crossover, linestyle=":", linewidth=1.1)
        phase_margin = self._margins.phase_margin_deg
        label = f"fc={crossover:.6g} Hz"
        if phase_margin is not None:
            label += f"\nPM={phase_margin:.5g}°"
        ax_mag.annotate(
            label,
            xy=(crossover, 0.0),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85},
        )

    def _on_cursor_changed(self, measurement: BodeCursorMeasurement) -> None:
        self._cursor_frequency_hz = measurement.frequency_hz
        self.cursor_changed.emit(measurement)


__all__ = ["PFCBodeCurve", "SelectableBodePanel"]
