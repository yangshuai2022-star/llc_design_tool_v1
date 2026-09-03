import numpy as np
import pytest

from llc_design.gui.widgets.bode_cursor import (
    BodeCursorTrace,
    format_frequency,
    interpolate_log_frequency,
)


def test_log_frequency_interpolation_uses_log_axis() -> None:
    frequencies = np.array([10.0, 100.0, 1000.0])
    values = np.array([0.0, 20.0, 40.0])
    assert interpolate_log_frequency(frequencies, values, np.sqrt(1000.0)) == pytest.approx(10.0)
    assert interpolate_log_frequency(frequencies, values, np.sqrt(100000.0)) == pytest.approx(30.0)


def test_log_frequency_interpolation_clamps_to_visible_range() -> None:
    frequencies = np.array([10.0, 100.0])
    values = np.array([-3.0, 7.0])
    assert interpolate_log_frequency(frequencies, values, 1.0) == pytest.approx(-3.0)
    assert interpolate_log_frequency(frequencies, values, 1000.0) == pytest.approx(7.0)


def test_trace_validation_rejects_mismatched_arrays() -> None:
    with pytest.raises(ValueError):
        BodeCursorTrace(
            label="bad",
            frequencies_hz=np.array([1.0, 2.0]),
            gain_db=np.array([0.0]),
            phase_deg=np.array([0.0, 0.0]),
        )


def test_frequency_formatter() -> None:
    assert format_frequency(125.0) == "125 Hz"
    assert format_frequency(12500.0) == "12.5 kHz"
    assert format_frequency(2.5e6) == "2.5 MHz"


def test_bode_cursor_updates_both_axes_and_measurement() -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from llc_design.gui.widgets.bode_cursor import BodeCursor

    figure = Figure()
    canvas = FigureCanvasAgg(figure)
    magnitude_axis = figure.add_subplot(211)
    phase_axis = figure.add_subplot(212, sharex=magnitude_axis)
    frequencies = np.array([10.0, 100.0, 1000.0])
    trace = BodeCursorTrace(
        label="plant",
        frequencies_hz=frequencies,
        gain_db=np.array([20.0, 0.0, -20.0]),
        phase_deg=np.array([0.0, -45.0, -90.0]),
    )
    cursor = BodeCursor(
        canvas, magnitude_axis, phase_axis, [trace], initial_frequency_hz=100.0)
    cursor.set_frequency(np.sqrt(10000.0), redraw=False)
    measurement = cursor.measurement()
    assert measurement.frequency_hz == pytest.approx(100.0)
    assert measurement.values[0].gain_db == pytest.approx(0.0)
    assert measurement.values[0].phase_deg == pytest.approx(-45.0)
    assert float(cursor.magnitude_line.get_xdata()[0]) == pytest.approx(100.0)
    assert float(cursor.phase_line.get_xdata()[0]) == pytest.approx(100.0)
    cursor.disconnect()


def test_bode_cursor_chinese_annotation_uses_cjk_font_without_missing_glyphs() -> None:
    import warnings

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from llc_design.gui.widgets.bode_cursor import BodeCursor

    figure = Figure(figsize=(7.0, 4.5))
    canvas = FigureCanvasAgg(figure)
    magnitude_axis = figure.add_subplot(211)
    phase_axis = figure.add_subplot(212, sharex=magnitude_axis)
    frequencies = np.array([10.0, 100.0, 1000.0])
    cursor = BodeCursor(
        canvas,
        magnitude_axis,
        phase_axis,
        [
            BodeCursorTrace(
                label="闭环输出阻抗",
                frequencies_hz=frequencies,
                gain_db=np.array([-20.0, -30.0, -40.0]),
                phase_deg=np.array([90.0, 45.0, 0.0]),
            )
        ],
        initial_frequency_hz=100.0,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        canvas.draw()

    glyph_warnings = [
        str(item.message) for item in caught
        if "Glyph" in str(item.message) and "missing from font" in str(item.message)
    ]
    assert glyph_warnings == []
    assert "闭环输出阻抗" in cursor._magnitude_annotation.get_text()
    assert cursor._magnitude_annotation.get_fontproperties().get_family() != ["monospace"]
    cursor.disconnect()
