"""Resonant, output and DC-link capacitor calculations."""

from __future__ import annotations

from dataclasses import dataclass
import math

from ..core.bus_capacitor import (hold_end_voltage_v, hold_time_s,
                                  required_capacitance_f)
from ..core.operating_point import LLCOperatingPoint
from ..core.spec import LLCDesignSpec, PrimaryTopology
from ..core.tank import TankDesign
from ..core.waveform import OutputRippleResult, output_capacitor_ripple


@dataclass(frozen=True)
class ResonantCapacitorResult:
    current_rms_a: float
    voltage_rms_v: float
    voltage_dc_bias_v: float
    voltage_peak_v: float
    loss_w: float
    voltage_margin: float
    current_margin: float


@dataclass(frozen=True)
class BusCapacitorResult:
    requested_hold_time_s: float
    predicted_end_voltage_v: float
    hold_time_to_limit_s: float
    required_capacitance_f: float
    installed_capacitance_f: float
    energy_start_j: float
    energy_end_j: float


def resonant_capacitor_result(spec: LLCDesignSpec, tank: TankDesign,
                              op: LLCOperatingPoint) -> ResonantCapacitorResult:
    xc = 1.0 / (2.0 * math.pi * op.switching_frequency_hz * tank.cr_f)
    vrms = op.resonant_current_rms_a * xc
    ac_peak = math.sqrt(2.0) * vrms
    # The series resonant capacitor blocks the bridge DC component: half bridge
    # sees a Vbus/2 DC bias, full bridge (symmetric square wave) sees ~0.
    dc_bias = 0.0 if spec.primary_topology == PrimaryTopology.FULL_BRIDGE else 0.5 * op.vbus_v
    vpk = dc_bias + ac_peak
    loss = op.resonant_current_rms_a**2 * spec.resonant_cap_esr_ohm
    allowed_v = spec.resonant_cap_voltage_rating_v * spec.capacitor_voltage_derating
    return ResonantCapacitorResult(
        current_rms_a=op.resonant_current_rms_a,
        voltage_rms_v=vrms, voltage_dc_bias_v=dc_bias, voltage_peak_v=vpk,
        loss_w=loss,
        voltage_margin=allowed_v / max(vpk, 1e-9),
        current_margin=spec.resonant_cap_rms_current_rating_a /
                       max(op.resonant_current_rms_a, 1e-9),
    )


def output_capacitor_result(spec: LLCDesignSpec,
                            op: LLCOperatingPoint) -> OutputRippleResult:
    return output_capacitor_ripple(
        output_current_a=op.output_current_a,
        switching_frequency_hz=op.switching_frequency_hz,
        capacitance_f=spec.output_capacitance_f,
        esr_ohm=spec.output_cap_esr_ohm,
        ripple_limit_vpp=spec.output_ripple_limit_vpp,
    )


def bus_capacitor_result(spec: LLCDesignSpec,
                         estimated_input_power_w: float | None = None) -> BusCapacitorResult:
    pin = estimated_input_power_w or spec.pout_w / spec.efficiency_assumption
    vend = hold_end_voltage_v(spec.bus_capacitance_f, spec.vbus_nom_v,
                             pin, spec.requested_hold_time_s)
    thold = hold_time_s(spec.bus_capacitance_f, spec.vbus_nom_v,
                        spec.vbus_hold_end_v, pin)
    creq = required_capacitance_f(spec.vbus_nom_v, spec.vbus_hold_end_v,
                                 pin, spec.requested_hold_time_s)
    return BusCapacitorResult(
        requested_hold_time_s=spec.requested_hold_time_s,
        predicted_end_voltage_v=vend,
        hold_time_to_limit_s=thold,
        required_capacitance_f=creq,
        installed_capacitance_f=spec.bus_capacitance_f,
        energy_start_j=0.5 * spec.bus_capacitance_f * spec.vbus_nom_v**2,
        energy_end_j=0.5 * spec.bus_capacitance_f * spec.vbus_hold_end_v**2,
    )
