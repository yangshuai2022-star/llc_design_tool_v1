"""LLC Q / gain / ZVS operating-region analysis.

V7 separates two ZVS concepts:
1. theoretical inductive region: Im{Zin} > 0;
2. engineering commutation margin: available dead-time charge and stored
   inductive energy versus MOSFET Qoss/Coss requirements.

The map uses the same FHA tank and device records as the rest of the LLC tool,
so Q, gain, working points and ZVS diagnostics stay internally consistent.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from .operating_point import LLCOperatingPoint, solve_operating_point
from .spec import LLCDesignSpec
from .tank import (
    TankDesign,
    bridge_fundamental_rms_v,
    design_tank,
    equivalent_ac_load_ohm,
    tank_state,
)
from ..models.devices import DeviceDatabase
from ..models.primary_bridge import primary_bridge_loss


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class LLCQZVSMap:
    frequencies_hz: FloatArray
    normalized_frequency: FloatArray
    load_fractions: FloatArray
    q_effective: FloatArray
    gain: FloatArray
    input_phase_deg: FloatArray
    commutation_current_a: FloatArray
    zvs_charge_margin: FloatArray
    zvs_energy_margin: FloatArray
    zvs_margin: FloatArray
    theoretical_inductive: NDArray[np.bool_]
    zvs_safe: NDArray[np.bool_]
    zvs_warning: NDArray[np.bool_]


@dataclass(frozen=True)
class LLCQZVSWorkPoint:
    vbus_v: float
    load_fraction: float
    q_effective: float
    frequency_hz: float
    normalized_frequency: float
    gain: float
    phase_deg: float
    commutation_current_a: float
    zvs_charge_margin: float
    zvs_energy_margin: float
    zvs_margin: float
    zvs_status: str


@dataclass(frozen=True)
class LLCQZVSAnalysis:
    spec: LLCDesignSpec
    tank: TankDesign
    map: LLCQZVSMap
    workpoints: tuple[LLCQZVSWorkPoint, ...]
    warnings: tuple[str, ...]


def _commutation_metrics(
    spec: LLCDesignSpec,
    tank: TankDesign,
    frequency_hz: float,
    rac_ohm: float,
    vbus_v: float,
    qoss_c: float,
    coss_f: float,
) -> tuple[float, float, float, float, float, float]:
    state = tank_state(tank, frequency_hz, rac_ohm)
    v_bridge = bridge_fundamental_rms_v(spec, vbus_v)
    i_res = v_bridge / state.z_input_ohm
    v_parallel = i_res * state.z_parallel_ohm
    i_mag = v_parallel / (1j * 2.0 * math.pi * frequency_hz * tank.lm_h)
    i_res_peak = math.sqrt(2.0) * abs(i_res)
    i_mag_peak = math.sqrt(2.0) * abs(i_mag)
    phase_rad = math.radians(state.input_phase_deg)
    transition_current = abs(i_res_peak * math.sin(phase_rad))
    commutation_current = max(0.75 * i_mag_peak, transition_current)

    npar = spec.primary_parallel_devices
    q_required = 2.0 * npar * max(qoss_c, coss_f * vbus_v)
    charge_margin = commutation_current * spec.primary_deadtime_s / max(q_required, 1e-15)
    e_required = 2.0 * npar * 0.5 * coss_f * vbus_v * vbus_v
    available_energy = 0.5 * (tank.lr_h + tank.lm_h) * commutation_current**2
    energy_margin = available_energy / max(e_required, 1e-15)
    return (
        state.gain,
        state.input_phase_deg,
        commutation_current,
        charge_margin,
        energy_margin,
        min(charge_margin, energy_margin),
    )


def build_q_zvs_analysis(
    spec: LLCDesignSpec,
    *,
    load_fractions: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 1.00, 1.20),
    vbus_points: tuple[float, ...] | None = None,
    frequency_points: int = 360,
) -> LLCQZVSAnalysis:
    spec.validate()
    if frequency_points < 80:
        raise ValueError("frequency_points must be >= 80")
    loads = np.asarray(load_fractions, dtype=float)
    if np.any(loads <= 0.0):
        raise ValueError("load fractions must be positive")

    tank = design_tank(spec)
    device = DeviceDatabase().get_primary(spec.primary_device)
    frequencies = np.geomspace(spec.minimum_frequency_hz, spec.maximum_frequency_hz, frequency_points)
    fn = frequencies / tank.fr_hz
    shape = (len(loads), len(frequencies))
    gain_map = np.empty(shape)
    phase_map = np.empty(shape)
    current_map = np.empty(shape)
    q_margin_map = np.empty(shape)
    e_margin_map = np.empty(shape)
    margin_map = np.empty(shape)
    q_effective = np.empty(len(loads))

    for li, load in enumerate(loads):
        modeled = max(float(load), spec.minimum_modeled_load_fraction)
        transferred_power = spec.pout_w * modeled * (
            1.0 + spec.rectifier_equivalent_drop_v / spec.vout_v
        )
        rac = equivalent_ac_load_ohm(
            spec.turns_ratio,
            spec.vout_v + spec.rectifier_equivalent_drop_v,
            transferred_power,
        )
        q_effective[li] = tank.zr_ohm / rac
        for fi, frequency in enumerate(frequencies):
            values = _commutation_metrics(
                spec, tank, float(frequency), rac, spec.vbus_nom_v,
                device.qoss_c, device.coss_er_f,
            )
            gain_map[li, fi], phase_map[li, fi], current_map[li, fi], q_margin_map[li, fi], e_margin_map[li, fi], margin_map[li, fi] = values

    inductive = phase_map > 0.0
    safe = inductive & (margin_map >= spec.primary_zvs_margin_required)
    warning = inductive & (margin_map >= 1.0) & ~safe

    map_result = LLCQZVSMap(
        frequencies_hz=frequencies,
        normalized_frequency=fn,
        load_fractions=loads,
        q_effective=q_effective,
        gain=gain_map,
        input_phase_deg=phase_map,
        commutation_current_a=current_map,
        zvs_charge_margin=q_margin_map,
        zvs_energy_margin=e_margin_map,
        zvs_margin=margin_map,
        theoretical_inductive=inductive,
        zvs_safe=safe,
        zvs_warning=warning,
    )

    buses = vbus_points or (
        spec.vbus_hold_end_v,
        spec.vbus_min_normal_v,
        spec.vbus_nom_v,
        spec.vbus_max_v,
    )
    workpoints: list[LLCQZVSWorkPoint] = []
    warnings: list[str] = []
    for vbus in buses:
        for load in loads:
            try:
                op: LLCOperatingPoint = solve_operating_point(spec, tank, float(vbus), float(load))
                loss = primary_bridge_loss(spec, tank, op, device)
                margin = min(loss.zvs_charge_margin, loss.zvs_energy_margin)
                if op.input_phase_deg <= 0.0:
                    status = "CAPACITIVE"
                elif margin < 1.0:
                    status = "ZVS_FAIL"
                elif margin < spec.primary_zvs_margin_required:
                    status = "ZVS_WARNING"
                else:
                    status = "ZVS_SAFE"
                workpoints.append(LLCQZVSWorkPoint(
                    vbus_v=float(vbus),
                    load_fraction=float(load),
                    q_effective=op.q_effective,
                    frequency_hz=op.switching_frequency_hz,
                    normalized_frequency=op.normalized_frequency,
                    gain=op.achieved_gain,
                    phase_deg=op.input_phase_deg,
                    commutation_current_a=op.commutation_current_a,
                    zvs_charge_margin=loss.zvs_charge_margin,
                    zvs_energy_margin=loss.zvs_energy_margin,
                    zvs_margin=margin,
                    zvs_status=status,
                ))
                if status != "ZVS_SAFE":
                    warnings.append(
                        f"{vbus:.0f} V / {load*100:.0f}% load: {status}, margin={margin:.3f}"
                    )
            except Exception as exc:
                warnings.append(f"{vbus:.0f} V / {load*100:.0f}% load: unsolved ({exc})")

    return LLCQZVSAnalysis(
        spec=spec,
        tank=tank,
        map=map_result,
        workpoints=tuple(workpoints),
        warnings=tuple(warnings),
    )


__all__ = [
    "LLCQZVSAnalysis",
    "LLCQZVSMap",
    "LLCQZVSWorkPoint",
    "build_q_zvs_analysis",
]
