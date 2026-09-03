"""Primary half/full-bridge MOSFET losses and simplified ZVS margin."""

from __future__ import annotations

from dataclasses import dataclass
import math

from ..core.operating_point import LLCOperatingPoint
from ..core.spec import LLCDesignSpec, MosfetSpec
from ..core.tank import TankDesign


@dataclass(frozen=True)
class PrimaryBridgeLoss:
    conduction_w: float
    turnoff_w: float
    gate_drive_w: float
    residual_coss_w: float
    deadtime_diode_w: float
    total_w: float
    zvs_charge_margin: float
    zvs_energy_margin: float
    zvs_residual_fraction: float
    device_rms_current_a: float
    device_peak_current_a: float
    voltage_stress_v: float


def primary_bridge_loss(spec: LLCDesignSpec, tank: TankDesign,
                        op: LLCOperatingPoint,
                        device: MosfetSpec) -> PrimaryBridgeLoss:
    npar = spec.primary_parallel_devices
    rds = device.rds_at(spec.primary_junction_temperature_c) / npar
    p_cond = (op.resonant_current_rms_a**2
              * spec.bridge_series_devices * rds)

    # LLC turn-off occurs on the magnetizing/commutation current, not the
    # resonant current peak; Eoff must scale with the physical current.
    device_peak = (op.commutation_current_a
                   * spec.primary_turnoff_current_factor / npar)
    eoff_device = (device.eoff_ref_j
                   * op.vbus_v / max(device.eoff_ref_v, 1e-9)
                   * device_peak / max(device.eoff_ref_i, 1e-9))
    p_eoff = spec.bridge_device_count * npar * eoff_device * op.switching_frequency_hz
    p_gate = (spec.bridge_device_count * npar * device.qg_c
              * device.gate_voltage_v * op.switching_frequency_hz)

    # Per commutating bridge leg, both device output capacitances must move.
    q_required = 2.0 * npar * max(device.qoss_c, device.coss_er_f * op.vbus_v)
    zvs_q_margin = (op.commutation_current_a * spec.primary_deadtime_s
                    / max(q_required, 1e-15))
    eoss_device = 0.5 * device.coss_er_f * op.vbus_v**2
    e_required = 2.0 * npar * eoss_device
    available_energy = 0.5 * (tank.lr_h + tank.lm_h) * op.commutation_current_a**2
    zvs_e_margin = available_energy / max(e_required, 1e-15)
    effective_margin = min(zvs_q_margin, zvs_e_margin)
    residual = min(1.0, max(0.0, 1.0 - effective_margin))**2
    p_coss = (spec.bridge_device_count * npar * eoss_device
              * op.switching_frequency_hz * residual)

    charge_time = q_required / max(op.commutation_current_a, 1e-9)
    diode_time = max(spec.primary_deadtime_s - charge_time, 0.0)
    commutations_per_cycle = 2 * spec.bridge_series_devices
    p_diode = (commutations_per_cycle * device.body_diode_vf_v
               * op.commutation_current_a * diode_time
               * op.switching_frequency_hz)

    total = p_cond + p_eoff + p_gate + p_coss + p_diode
    return PrimaryBridgeLoss(
        conduction_w=p_cond, turnoff_w=p_eoff, gate_drive_w=p_gate,
        residual_coss_w=p_coss, deadtime_diode_w=p_diode, total_w=total,
        zvs_charge_margin=zvs_q_margin, zvs_energy_margin=zvs_e_margin,
        zvs_residual_fraction=residual,
        device_rms_current_a=(op.resonant_current_rms_a /
                              (math.sqrt(2.0) * npar)),
        device_peak_current_a=op.resonant_current_peak_a / npar,
        voltage_stress_v=op.vbus_v,
    )
