"""LLC operating-point solution from the common FHA tank circuit."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .spec import LLCDesignSpec
from .tank import (FHA_RECTIFIER_K, TankDesign, bridge_fundamental_rms_v,
                   equivalent_ac_load_ohm, gain_vector, solve_frequency,
                   tank_state, target_gain)


@dataclass(frozen=True)
class LLCOperatingPoint:
    vbus_v: float
    load_fraction: float
    pout_w: float
    output_current_a: float
    rac_ohm: float
    q_effective: float
    required_gain: float
    switching_frequency_hz: float
    normalized_frequency: float
    achieved_gain: float
    branch: str
    input_impedance_ohm: complex
    input_phase_deg: float
    bridge_fundamental_rms_v: float
    transformer_fundamental_rms_v: float
    transformer_square_equivalent_v: float
    resonant_current_rms_a: float
    resonant_current_peak_a: float
    magnetizing_current_rms_a: float
    magnetizing_current_peak_a: float
    reflected_load_current_rms_a: float
    secondary_current_rms_a: float
    secondary_current_peak_a: float
    commutation_current_a: float
    estimated_input_power_w: float
    available_gain_min: float
    available_gain_max: float

    @property
    def inductive(self) -> bool:
        return self.input_phase_deg > 0.0


def solve_operating_point(spec: LLCDesignSpec, tank: TankDesign,
                          vbus_v: float, load_fraction: float) -> LLCOperatingPoint:
    modeled_fraction = max(load_fraction, spec.minimum_modeled_load_fraction)
    pout = spec.pout_w * modeled_fraction
    io = pout / spec.vout_v
    transferred_power = pout * (1.0 + spec.rectifier_equivalent_drop_v / spec.vout_v)
    rac = equivalent_ac_load_ohm(
        spec.turns_ratio, spec.vout_v + spec.rectifier_equivalent_drop_v,
        transferred_power)
    required = target_gain(spec, vbus_v)
    solution = solve_frequency(tank, spec, rac, required)
    state = tank_state(tank, solution.frequency_hz, rac)

    v_bridge = bridge_fundamental_rms_v(spec, vbus_v)
    i_res_phasor = v_bridge / state.z_input_ohm
    v_parallel = i_res_phasor * state.z_parallel_ohm
    i_load = v_parallel / rac
    i_mag = v_parallel / (1j * 2.0 * math.pi * solution.frequency_hz * tank.lm_h)

    i_res_rms = abs(i_res_phasor)
    i_res_peak = math.sqrt(2.0) * i_res_rms
    i_mag_rms = abs(i_mag)
    i_mag_peak = math.sqrt(2.0) * i_mag_rms
    i_load_rms = abs(i_load)
    i_sec_rms = spec.turns_ratio * i_load_rms
    i_sec_peak = math.sqrt(2.0) * i_sec_rms

    phase_rad = math.radians(state.input_phase_deg)
    fundamental_transition_current = abs(i_res_peak * math.sin(phase_rad))
    commutation_current = max(0.75 * i_mag_peak, fundamental_transition_current)
    input_power = (v_parallel.real * i_load.real + v_parallel.imag * i_load.imag)

    # Equivalent square-wave transformer voltage corresponding to its FHA fundamental.
    v_square_eq = abs(v_parallel) / FHA_RECTIFIER_K

    # Sample allowed frequency range to report reachable gain.
    freqs = np.geomspace(spec.minimum_frequency_hz, spec.maximum_frequency_hz, 800)
    gains = gain_vector(tank, freqs, rac)

    return LLCOperatingPoint(
        vbus_v=vbus_v,
        load_fraction=load_fraction,
        pout_w=spec.pout_w * load_fraction,
        output_current_a=spec.output_current_a * load_fraction,
        rac_ohm=rac,
        q_effective=tank.zr_ohm / rac,
        required_gain=required,
        switching_frequency_hz=solution.frequency_hz,
        normalized_frequency=solution.frequency_hz / tank.fr_hz,
        achieved_gain=solution.achieved_gain,
        branch=solution.branch,
        input_impedance_ohm=state.z_input_ohm,
        input_phase_deg=state.input_phase_deg,
        bridge_fundamental_rms_v=v_bridge,
        transformer_fundamental_rms_v=abs(v_parallel),
        transformer_square_equivalent_v=v_square_eq,
        resonant_current_rms_a=i_res_rms,
        resonant_current_peak_a=i_res_peak,
        magnetizing_current_rms_a=i_mag_rms,
        magnetizing_current_peak_a=i_mag_peak,
        reflected_load_current_rms_a=i_load_rms,
        secondary_current_rms_a=i_sec_rms,
        secondary_current_peak_a=i_sec_peak,
        commutation_current_a=commutation_current,
        estimated_input_power_w=input_power,
        available_gain_min=float(gains.min()),
        available_gain_max=float(gains.max()),
    )
