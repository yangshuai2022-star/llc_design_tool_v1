"""Shared measurement extraction for V8 LLC model results."""

from __future__ import annotations

import math

import numpy as np

from .types import LLCAnalysisRequest, ModelMetrics
from ..dynamics.waveforms import WaveformBundle


def _signal_stat(bundle: WaveformBundle, key: str, attribute: str, default: float = 0.0) -> float:
    signal = bundle.signals.get(key)
    if signal is None:
        return float(default)
    return float(getattr(signal.statistics, attribute))


def metrics_from_waveform(
    request: LLCAnalysisRequest,
    bundle: WaveformBundle,
    *,
    input_phase_deg: float,
    normalized_gain: float | None = None,
    input_power_w: float | None = None,
    modeled_series_loss_w: float = 0.0,
    modeled_rectifier_drop_loss_w: float = 0.0,
) -> ModelMetrics:
    """Measure a synchronized waveform bundle using one common convention."""

    spec = request.spec
    vout = _signal_stat(bundle, "v_output", "average", request.output_voltage_target_v)
    pout = vout * vout / request.load_resistance_ohm
    if normalized_gain is None:
        normalized_gain = (
            spec.turns_ratio * (vout + spec.rectifier_equivalent_drop_v)
            / (spec.bridge_gain * request.bus_voltage_v)
        )
    if input_power_w is None:
        vbridge = bundle.signal("v_bridge").values
        ir = bundle.signal("i_resonant").values
        input_power_w = float(np.mean(vbridge * ir))

    ir_rms = _signal_stat(bundle, "i_resonant", "rms")
    ir_peak = _signal_stat(bundle, "i_resonant", "absolute_peak")
    im_rms = _signal_stat(bundle, "i_magnetizing", "rms")
    im_peak = _signal_stat(bundle, "i_magnetizing", "absolute_peak")
    iload_rms = _signal_stat(bundle, "i_primary_load", "rms")
    iload_peak = _signal_stat(bundle, "i_primary_load", "absolute_peak")
    isec_rms = _signal_stat(bundle, "i_transformer_secondary", "rms")
    isec_peak = _signal_stat(bundle, "i_transformer_secondary", "absolute_peak")
    irect_avg = _signal_stat(bundle, "i_rectified", "average")
    vcr_rms = _signal_stat(bundle, "v_resonant_cap", "rms")
    vcr_peak = _signal_stat(bundle, "v_resonant_cap", "absolute_peak")
    ico_rms = _signal_stat(bundle, "i_output_cap", "rms")

    accounted_output_w = pout + modeled_series_loss_w + modeled_rectifier_drop_loss_w
    balance_error = float(input_power_w - accounted_output_w)
    balance_percent = 100.0 * balance_error / max(abs(float(input_power_w)), 1e-12)

    return ModelMetrics(
        switching_frequency_hz=float(bundle.switching_frequency_hz),
        output_voltage_v=vout,
        output_power_w=pout,
        normalized_gain=float(normalized_gain),
        input_phase_deg=float(input_phase_deg),
        input_power_w=float(input_power_w),
        resonant_current_rms_a=ir_rms,
        resonant_current_peak_a=ir_peak,
        magnetizing_current_rms_a=im_rms,
        magnetizing_current_peak_a=im_peak,
        primary_load_current_rms_a=iload_rms,
        primary_load_current_peak_a=iload_peak,
        secondary_current_rms_a=isec_rms,
        secondary_current_peak_a=isec_peak,
        rectifier_current_average_a=irect_avg,
        resonant_capacitor_rms_v=vcr_rms,
        resonant_capacitor_peak_v=vcr_peak,
        output_capacitor_current_rms_a=ico_rms,
        power_balance_error_w=balance_error,
        power_balance_error_percent=balance_percent,
    )


def relative_error_percent(value: float, reference: float) -> float:
    """Signed relative error with a stable zero-reference convention."""

    if math.isclose(reference, 0.0, abs_tol=1e-15):
        return 0.0 if math.isclose(value, 0.0, abs_tol=1e-15) else math.copysign(math.inf, value)
    return 100.0 * (value - reference) / reference
