"""Full-bridge synchronous-rectifier loss model."""

from __future__ import annotations

from dataclasses import dataclass
import math

from ..core.operating_point import LLCOperatingPoint
from ..core.spec import LLCDesignSpec, MosfetSpec


@dataclass(frozen=True)
class SynchronousRectifierLoss:
    conduction_w: float
    deadtime_diode_w: float
    turnoff_w: float
    coss_w: float
    gate_drive_w: float
    total_w: float
    device_rms_current_a: float
    device_peak_current_a: float
    voltage_stress_v: float


def synchronous_rectifier_loss(spec: LLCDesignSpec, op: LLCOperatingPoint,
                                device: MosfetSpec) -> SynchronousRectifierLoss:
    npar = spec.sr_parallel_devices_per_position
    rds_path = 2.0 * device.rds_at(spec.sr_junction_temperature_c) / npar
    p_cond = op.secondary_current_rms_a**2 * rds_path

    omega = 2.0 * math.pi * op.switching_frequency_hz
    timing_window = max(spec.sr_deadtime_s + spec.sr_turnoff_advance_s, 0.0)
    # Two current zero-crossing commutations per switching cycle, two diode drops
    # in the full bridge. Near zero crossing, i ~= Ipk*omega*t.
    p_diode = (4.0 * device.body_diode_vf_v * 0.5
               * op.secondary_current_peak_a * omega * timing_window**2
               * op.switching_frequency_hz)

    i_off_position = (op.secondary_current_peak_a
                      * math.sin(min(omega * spec.sr_turnoff_advance_s,
                                     math.pi / 2.0)) / npar)
    eoff = (device.eoff_ref_j
            * spec.vout_v / max(device.eoff_ref_v, 1e-9)
            * i_off_position / max(device.eoff_ref_i, 1e-9))
    p_eoff = 4.0 * npar * eoff * op.switching_frequency_hz

    v_stress = spec.sr_voltage_overshoot_factor * spec.vout_v
    eoss = 0.5 * device.coss_er_f * v_stress**2
    p_coss = (4.0 * npar * eoss * op.switching_frequency_hz
              * spec.sr_coss_dissipation_factor)
    p_gate = 4.0 * npar * device.qg_c * device.gate_voltage_v * op.switching_frequency_hz

    total = p_cond + p_diode + p_eoff + p_coss + p_gate
    return SynchronousRectifierLoss(
        conduction_w=p_cond, deadtime_diode_w=p_diode, turnoff_w=p_eoff,
        coss_w=p_coss, gate_drive_w=p_gate, total_w=total,
        device_rms_current_a=op.secondary_current_rms_a / (math.sqrt(2.0) * npar),
        device_peak_current_a=op.secondary_current_peak_a / npar,
        voltage_stress_v=v_stress,
    )
