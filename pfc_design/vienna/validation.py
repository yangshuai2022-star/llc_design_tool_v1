"""Runtime validation helpers for the Vienna PFC analysis pipeline.

The GUI runs several independent numerical stages in a worker thread.  Keeping
validation outside the GUI makes failures deterministic and testable: a bad
small-signal response, line-cycle state or switching reconstruction is reported
at the stage where it was created instead of surfacing later as an opaque
Matplotlib/Qt exception.
"""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np

from .analysis import ViennaControlLabAnalysis
from .waveforms import ViennaLineCycleWaveforms, ViennaSwitchingWaveforms


def _finite_array(name: str, value) -> np.ndarray:
    array = np.asarray(value)
    if array.size == 0:
        raise ValueError(f"{name}: empty numerical array")
    if not np.all(np.isfinite(array)):
        bad = int(np.size(array) - np.count_nonzero(np.isfinite(array)))
        raise ValueError(f"{name}: contains {bad} NaN/Inf sample(s)")
    return array


def _require_signals(signals: Mapping[str, object], keys: tuple[str, ...], *, stage: str) -> None:
    missing = [key for key in keys if key not in signals]
    if missing:
        raise KeyError(f"{stage}: missing signal(s): {', '.join(missing)}")
    lengths = []
    for key in keys:
        lengths.append(len(_finite_array(f"{stage}.{key}", signals[key])))
    if len(set(lengths)) != 1:
        raise ValueError(f"{stage}: signal lengths are inconsistent: {dict(zip(keys, lengths))}")


def validate_vienna_analysis(analysis: ViennaControlLabAnalysis) -> None:
    """Validate every response consumed by the Vienna Bode GUI."""
    freq = _finite_array("Vienna Bode frequency", analysis.frequencies_hz)
    if np.any(freq <= 0.0) or np.any(np.diff(freq) <= 0.0):
        raise ValueError("Vienna Bode frequency vector must be positive and increasing")

    loops = (
        ("current", analysis.current_loop.responses),
        ("voltage", analysis.voltage_loop.responses),
        ("balance", analysis.balance_loop.responses),
    )
    for label, responses in loops:
        for key, value in responses.items():
            arr = _finite_array(f"Vienna {label} loop.{key}", value)
            if len(arr) != len(freq):
                raise ValueError(
                    f"Vienna {label} loop.{key}: {len(arr)} points, expected {len(freq)}"
                )

    for label, response in (
        ("current sense", analysis.current_sense_response.total),
        ("phase-voltage sense", analysis.phase_voltage_sense_response.total),
        ("split-bus sense", analysis.split_bus_sense_response.total),
    ):
        arr = _finite_array(f"Vienna {label}", response)
        if len(arr) != len(freq):
            raise ValueError(f"Vienna {label}: response length does not match Bode frequency vector")


def validate_vienna_line_cycle(line: ViennaLineCycleWaveforms) -> None:
    """Validate all line-cycle signals used by the GUI and basic bus invariants."""
    time = _finite_array("Vienna line-cycle time", line.time_s)
    if len(time) < 2 or np.any(np.diff(time) <= 0.0):
        raise ValueError("Vienna line-cycle time vector must be strictly increasing")

    required = (
        "va", "vb", "vc", "va_meas", "vb_meas", "vc_meas",
        "ia", "ib", "ic", "ia_meas", "ib_meas", "ic_meas",
        "ia_ref", "ib_ref", "ic_ref",
        "mod_a", "mod_b", "mod_c", "duty_a", "duty_b", "duty_c",
        "vdc", "vdc_plus", "vdc_minus", "vdc_delta", "vdc_measured",
        "gcmd", "vloop", "balance_output", "midpoint_current",
        "input_power_total", "load_current", "bus_series_current", "sector",
    )
    _require_signals(line.signals, required, stage="Vienna line cycle")
    for key in required:
        if len(np.asarray(line.signals[key])) != len(time):
            raise ValueError(f"Vienna line cycle.{key}: length does not match time vector")

    vp = np.asarray(line.signals["vdc_plus"], dtype=float)
    vn = np.asarray(line.signals["vdc_minus"], dtype=float)
    vdc = np.asarray(line.signals["vdc"], dtype=float)
    delta = np.asarray(line.signals["vdc_delta"], dtype=float)
    if np.any(vp <= 0.0) or np.any(vn <= 0.0):
        raise ValueError("Vienna split-bus voltage became non-positive")
    if not np.allclose(vdc, vp + vn, rtol=1e-9, atol=1e-6):
        raise ValueError("Vienna split-bus invariant failed: Vdc != Vdc+ + Vdc-")
    if not np.allclose(delta, vp - vn, rtol=1e-9, atol=1e-6):
        raise ValueError("Vienna split-bus invariant failed: ΔV != Vdc+ - Vdc-")

    ia = np.asarray(line.signals["ia"], dtype=float)
    ib = np.asarray(line.signals["ib"], dtype=float)
    ic = np.asarray(line.signals["ic"], dtype=float)
    scale = max(float(np.max(np.abs(np.concatenate((ia, ib, ic))))), 1.0)
    if float(np.max(np.abs(ia + ib + ic))) > max(1e-6, 1e-7 * scale):
        raise ValueError("Vienna three-wire invariant failed: Ia + Ib + Ic is not approximately zero")

    metric_values = np.asarray([
        *line.metrics.phase_current_rms_a,
        *line.metrics.phase_current_thd_percent,
        *line.metrics.phase_power_factor,
        line.metrics.total_input_power_w,
        line.metrics.overall_power_factor,
        line.metrics.bus_voltage_average_v,
        line.metrics.bus_voltage_ripple_pp_v,
        line.metrics.midpoint_delta_average_v,
        line.metrics.midpoint_delta_pp_v,
        line.metrics.midpoint_current_rms_a,
    ], dtype=float)
    _finite_array("Vienna line-cycle metrics", metric_values)


def validate_vienna_switching(switching: ViennaSwitchingWaveforms) -> None:
    """Validate switching signals consumed by the local-workpoint plot."""
    time = _finite_array("Vienna switching time", switching.time_s)
    if len(time) < 2 or np.any(np.diff(time) <= 0.0):
        raise ValueError("Vienna switching time vector must be strictly increasing")
    required = [
        "carrier", "sector", "midpoint_current", "upper_cap_current", "lower_cap_current",
    ]
    for phase in "abc":
        required += [
            f"gate_{phase}", f"vconv_{phase}", f"current_{phase}",
            f"upper_diode_{phase}", f"lower_diode_{phase}", f"duty_{phase}",
        ]
    _require_signals(switching.signals, tuple(required), stage="Vienna switching")
    for key in required:
        if len(np.asarray(switching.signals[key])) != len(time):
            raise ValueError(f"Vienna switching.{key}: length does not match time vector")


def validate_vienna_pipeline(
    analysis: ViennaControlLabAnalysis,
    line: ViennaLineCycleWaveforms,
    switching: ViennaSwitchingWaveforms,
) -> None:
    validate_vienna_analysis(analysis)
    validate_vienna_line_cycle(line)
    validate_vienna_switching(switching)


__all__ = [
    "validate_vienna_analysis",
    "validate_vienna_line_cycle",
    "validate_vienna_switching",
    "validate_vienna_pipeline",
]
