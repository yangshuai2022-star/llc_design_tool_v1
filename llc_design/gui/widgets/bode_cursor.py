"""Interactive draggable cursor shared by LLC Bode plots.

The cursor is intentionally independent from the Qt widget hierarchy.  It uses
Matplotlib canvas events, so the same implementation can be reused by the
power-stage and complete digital-loop workbenches.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Iterable

import numpy as np
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties


_CJK_FONT_CANDIDATES: tuple[str, ...] = (
    # Windows
    "Microsoft YaHei",
    "Microsoft YaHei UI",
    "SimHei",
    "SimSun",
    # macOS
    "PingFang SC",
    "Hiragino Sans GB",
    # Linux / portable CJK installations
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Micro Hei",
    "AR PL UMing CN",
)


@lru_cache(maxsize=1)
def resolve_cjk_font_properties(font_size: float = 8.0) -> FontProperties:
    """Return an installed CJK-capable font for Matplotlib cursor labels.

    The Bode legends already use the application's configured sans-serif font,
    but the original cursor annotation forced ``family='monospace'``.  Most
    default monospace fonts do not contain Chinese glyphs, which produced
    square replacement characters on Windows.  Resolve a native CJK font by
    family name and bind the annotation directly to its installed font file.

    No font file is bundled with the application.  If no candidate is found,
    the normal Matplotlib sans-serif stack is used as a final fallback.
    """

    for family in _CJK_FONT_CANDIDATES:
        try:
            font_path = font_manager.findfont(
                FontProperties(family=[family]),
                fallback_to_default=False,
            )
        except (ValueError, RuntimeError):
            continue
        if font_path:
            return FontProperties(fname=font_path, size=font_size)
    return FontProperties(family=["sans-serif"], size=font_size)


@dataclass(frozen=True)
class BodeCursorTrace:
    """One visible Bode trace sampled on a positive frequency grid."""

    label: str
    frequencies_hz: np.ndarray
    gain_db: np.ndarray
    phase_deg: np.ndarray
    color: Any = None

    def __post_init__(self) -> None:
        frequency = np.asarray(self.frequencies_hz, dtype=float)
        gain = np.asarray(self.gain_db, dtype=float)
        phase = np.asarray(self.phase_deg, dtype=float)
        if frequency.ndim != 1 or len(frequency) < 2:
            raise ValueError("Bode cursor frequency array must be one-dimensional")
        if gain.shape != frequency.shape or phase.shape != frequency.shape:
            raise ValueError("Bode cursor trace arrays must have identical shapes")
        if np.any(~np.isfinite(frequency)) or np.any(frequency <= 0.0):
            raise ValueError("Bode cursor frequencies must be finite and positive")
        if np.any(np.diff(frequency) <= 0.0):
            raise ValueError("Bode cursor frequencies must be strictly increasing")


@dataclass(frozen=True)
class BodeCursorValue:
    """Interpolated gain and phase for one trace at the cursor frequency."""

    label: str
    gain_db: float
    phase_deg: float


@dataclass(frozen=True)
class BodeCursorMeasurement:
    """All visible trace values at one frequency."""

    frequency_hz: float
    values: tuple[BodeCursorValue, ...]


def interpolate_log_frequency(
    frequencies_hz: np.ndarray,
    values: np.ndarray,
    frequency_hz: float,
) -> float:
    """Interpolate a Bode value linearly on the logarithmic frequency axis."""

    frequency = np.asarray(frequencies_hz, dtype=float)
    ordinate = np.asarray(values, dtype=float)
    if frequency.ndim != 1 or ordinate.shape != frequency.shape:
        raise ValueError("frequency and value arrays must be one-dimensional and equal")
    if len(frequency) < 2 or np.any(frequency <= 0.0):
        raise ValueError("at least two positive frequency samples are required")
    query = float(np.clip(frequency_hz, frequency[0], frequency[-1]))
    return float(np.interp(np.log10(query), np.log10(frequency), ordinate))


def format_frequency(frequency_hz: float) -> str:
    """Compact engineering-format frequency string for the GUI readout."""

    value = float(frequency_hz)
    magnitude = abs(value)
    if magnitude >= 1e6:
        return f"{value / 1e6:.7g} MHz"
    if magnitude >= 1e3:
        return f"{value / 1e3:.7g} kHz"
    return f"{value:.7g} Hz"


class BodeCursor:
    """Draggable vertical cursor with synchronized gain/phase readout.

    A left-button press anywhere in either Bode axis moves the cursor and begins
    a drag operation.  Gain and phase are interpolated on log-frequency, which
    avoids the visibly stepped readout produced by nearest-sample selection.
    """

    def __init__(
        self,
        canvas: Any,
        magnitude_axis: Any,
        phase_axis: Any,
        traces: Iterable[BodeCursorTrace],
        *,
        initial_frequency_hz: float | None = None,
        on_changed: Callable[[BodeCursorMeasurement], None] | None = None,
    ) -> None:
        self.canvas = canvas
        self.magnitude_axis = magnitude_axis
        self.phase_axis = phase_axis
        self.traces = tuple(traces)
        if not self.traces:
            raise ValueError("at least one Bode cursor trace is required")
        self.on_changed = on_changed
        self._dragging = False
        self._event_ids: list[int] = []

        self.minimum_frequency_hz = max(
            float(trace.frequencies_hz[0]) for trace in self.traces)
        self.maximum_frequency_hz = min(
            float(trace.frequencies_hz[-1]) for trace in self.traces)
        if self.minimum_frequency_hz >= self.maximum_frequency_hz:
            raise ValueError("visible Bode traces do not share a frequency range")

        if initial_frequency_hz is None or not np.isfinite(initial_frequency_hz):
            initial_frequency_hz = float(
                np.sqrt(self.minimum_frequency_hz * self.maximum_frequency_hz))
        self.frequency_hz = self._clamp_frequency(initial_frequency_hz)

        self.magnitude_line = magnitude_axis.axvline(
            self.frequency_hz, linewidth=1.25, linestyle="--", zorder=20)
        self.phase_line = phase_axis.axvline(
            self.frequency_hz, linewidth=1.25, linestyle="--", zorder=20)

        self._magnitude_markers: list[Any] = []
        self._phase_markers: list[Any] = []
        for trace in self.traces:
            magnitude_marker, = magnitude_axis.plot(
                [self.frequency_hz], [0.0], marker="o", linestyle="None",
                markersize=5.5, color=trace.color, zorder=25)
            phase_marker, = phase_axis.plot(
                [self.frequency_hz], [0.0], marker="o", linestyle="None",
                markersize=5.5, color=trace.color, zorder=25)
            self._magnitude_markers.append(magnitude_marker)
            self._phase_markers.append(phase_marker)

        annotation_font = resolve_cjk_font_properties(8.0)
        self._magnitude_annotation = magnitude_axis.text(
            0.985, 0.975, "", transform=magnitude_axis.transAxes,
            horizontalalignment="right", verticalalignment="top",
            fontproperties=annotation_font,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.88},
            zorder=30)
        self._phase_annotation = phase_axis.text(
            0.985, 0.975, "", transform=phase_axis.transAxes,
            horizontalalignment="right", verticalalignment="top",
            fontproperties=annotation_font,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.88},
            zorder=30)

        self._event_ids.extend(
            [
                canvas.mpl_connect("button_press_event", self._on_press),
                canvas.mpl_connect("motion_notify_event", self._on_motion),
                canvas.mpl_connect("button_release_event", self._on_release),
            ]
        )
        self.set_frequency(self.frequency_hz, redraw=False)

    def disconnect(self) -> None:
        """Disconnect canvas callbacks before a figure is cleared or destroyed."""

        for event_id in self._event_ids:
            self.canvas.mpl_disconnect(event_id)
        self._event_ids.clear()
        self._dragging = False

    def _clamp_frequency(self, frequency_hz: float) -> float:
        return float(np.clip(
            float(frequency_hz),
            self.minimum_frequency_hz,
            self.maximum_frequency_hz,
        ))

    def measurement(self) -> BodeCursorMeasurement:
        values = tuple(
            BodeCursorValue(
                label=trace.label,
                gain_db=interpolate_log_frequency(
                    trace.frequencies_hz, trace.gain_db, self.frequency_hz),
                phase_deg=interpolate_log_frequency(
                    trace.frequencies_hz, trace.phase_deg, self.frequency_hz),
            )
            for trace in self.traces
        )
        return BodeCursorMeasurement(self.frequency_hz, values)

    def set_frequency(self, frequency_hz: float, *, redraw: bool = True) -> None:
        """Move the cursor and update all markers/readouts."""

        self.frequency_hz = self._clamp_frequency(frequency_hz)
        measurement = self.measurement()
        self.magnitude_line.set_xdata([self.frequency_hz, self.frequency_hz])
        self.phase_line.set_xdata([self.frequency_hz, self.frequency_hz])

        gain_lines = [f"f = {format_frequency(self.frequency_hz)}"]
        phase_lines = [f"f = {format_frequency(self.frequency_hz)}"]
        for index, value in enumerate(measurement.values):
            self._magnitude_markers[index].set_data(
                [self.frequency_hz], [value.gain_db])
            self._phase_markers[index].set_data(
                [self.frequency_hz], [value.phase_deg])
            gain_lines.append(f"{value.label}: {value.gain_db:+.5g} dB")
            phase_lines.append(f"{value.label}: {value.phase_deg:+.6g}°")

        self._magnitude_annotation.set_text("\n".join(gain_lines))
        self._phase_annotation.set_text("\n".join(phase_lines))
        if self.on_changed is not None:
            self.on_changed(measurement)
        if redraw:
            self.canvas.draw_idle()

    def _event_frequency(self, event: Any) -> float | None:
        if event.inaxes not in (self.magnitude_axis, self.phase_axis):
            return None
        if event.xdata is None or not np.isfinite(event.xdata) or event.xdata <= 0.0:
            return None
        return self._clamp_frequency(float(event.xdata))

    def _on_press(self, event: Any) -> None:
        if event.button != 1:
            return
        frequency = self._event_frequency(event)
        if frequency is None:
            return
        self._dragging = True
        self.set_frequency(frequency)

    def _on_motion(self, event: Any) -> None:
        if not self._dragging:
            return
        frequency = self._event_frequency(event)
        if frequency is not None:
            self.set_frequency(frequency)

    def _on_release(self, event: Any) -> None:
        if event.button == 1:
            self._dragging = False
