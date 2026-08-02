"""Simple reconstructed secondary rectifier/output-capacitor waveforms."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class OutputRippleResult:
    capacitor_current_rms_a: float
    capacitor_current_peak_a: float
    charge_pp_c: float
    capacitive_ripple_vpp: float
    esr_ripple_vpp: float
    total_ripple_vpp: float
    capacitor_loss_w: float
    required_capacitance_f: float


def output_capacitor_ripple(output_current_a: float, switching_frequency_hz: float,
                            capacitance_f: float, esr_ohm: float,
                            ripple_limit_vpp: float,
                            samples: int = 4096) -> OutputRippleResult:
    """Numerically integrate one full-wave rectified current lobe.

    The sinusoidal secondary current peak is selected so its rectified average
    equals the DC output current. This is an FHA waveform estimate.
    """
    if min(output_current_a, switching_frequency_hz, capacitance_f) <= 0:
        raise ValueError("current, frequency and capacitance must be positive")
    ripple_period = 1.0 / (2.0 * switching_frequency_hz)
    t = np.linspace(0.0, ripple_period, samples, endpoint=False)
    i_peak = math.pi * output_current_a / 2.0
    i_rect = i_peak * np.sin(math.pi * t / ripple_period)
    i_cap = i_rect - output_current_a
    dt = ripple_period / samples
    charge = np.cumsum(i_cap) * dt
    charge -= charge.mean()
    voltage_c = charge / capacitance_f
    voltage_total = voltage_c + esr_ohm * i_cap

    i_rms = float(np.sqrt(np.mean(i_cap**2)))
    q_pp = float(charge.max() - charge.min())
    cap_ripple = float(voltage_c.max() - voltage_c.min())
    esr_ripple = float((esr_ohm * i_cap).max() - (esr_ohm * i_cap).min())
    total_ripple = float(voltage_total.max() - voltage_total.min())
    loss = i_rms**2 * esr_ohm

    # Conservative split: reserve the directly calculated ESR pp component,
    # then allocate the remaining ripple budget to capacitive charge swing.
    cap_budget = ripple_limit_vpp - esr_ripple
    required_c = math.inf if cap_budget <= 0 else q_pp / cap_budget

    return OutputRippleResult(
        capacitor_current_rms_a=i_rms,
        capacitor_current_peak_a=float(max(abs(i_cap.min()), abs(i_cap.max()))),
        charge_pp_c=q_pp,
        capacitive_ripple_vpp=cap_ripple,
        esr_ripple_vpp=esr_ripple,
        total_ripple_vpp=total_ripple,
        capacitor_loss_w=loss,
        required_capacitance_f=required_c,
    )
