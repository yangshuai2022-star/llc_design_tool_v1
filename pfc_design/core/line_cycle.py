"""Line-cycle trace for dual-timescale PFC loss kernel.

Generates a half-line-cycle grid of per-angle electrical quantities
using midpoint sampling (theta never hits 0 or pi).  All loss models
consume this trace for line-cycle-integrated loss computation.

Formulas follow the DOC/pfc_loss_kernel_change_log_v2.md spec.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .operating_point import OperatingPoint
    from .spec import DesignSpec


@dataclass
class LineCycleTrace:
    """Per-angle electrical trace over a half line cycle [0, pi].

    All current fields are per-phase unless marked ``_total``.
    Angular arrays use midpoint sampling: theta_i = (i + 0.5) * pi / N.
    """

    # -- grid -----------------------------------------------------------
    theta: np.ndarray          # midpoint theta, (0, pi) exclusive
    theta_deg: np.ndarray      # theta in degrees
    dtheta: float              # pi / N_pts

    # -- voltage --------------------------------------------------------
    vm: float                  # peak line voltage (V)
    sin_theta: np.ndarray      # sin(theta), clamped to sin_min
    vin_abs: np.ndarray        # |Vm * sin(theta)|  (V)

    # -- duty cycle -----------------------------------------------------
    duty: np.ndarray           # 1 - vin_abs/Vout, clipped [0.02, 0.98]

    # -- per-phase currents ---------------------------------------------
    i_avg_phase: np.ndarray    # instantaneous average current (A)
    delta_i_pp: np.ndarray     # switching ripple pk-pk (A)
    i_peak: np.ndarray         # i_avg + delta_i_pp/2 (A)
    i_valley: np.ndarray       # i_avg - delta_i_pp/2, clamped >= 0 (A)
    i_rms_local_sq: np.ndarray # i_avg^2 + delta_i_pp^2 / 12 (A²)

    # -- per-phase scalars ----------------------------------------------
    iin_rms_phase: float       # per-phase RMS input current (A)
    iin_pk_phase: float        # per-phase peak input current (A)

    # -- inductance -----------------------------------------------------
    L_uh: float                # effective inductance used for trace (μH)

    # -- DCM / numerical risk flags -------------------------------------
    near_zero_or_dcm: np.ndarray  # True where duty > 0.95


# ── Integration helpers ────────────────────────────────────────────────

def average_over_half_cycle(values: np.ndarray, theta: np.ndarray) -> float:
    """Half-line-cycle average:  (1/pi) * ∫[0,π] P(θ) dθ.

    Uses midpoint-rule integration for equally spaced midpoint samples
    (theta_i = (i+0.5)*pi/N).  Equivalent to np.mean(values) for the
    standard midpoint grid.
    """
    values = np.asarray(values, dtype=float)
    theta = np.asarray(theta, dtype=float)
    if values.size != theta.size:
        raise ValueError("values and theta must have the same size")
    dtheta = np.pi / theta.size
    return float(np.sum(values) * dtheta / np.pi)


def integrate_over_half_cycle(values: np.ndarray, theta: np.ndarray) -> float:
    """Raw integral ∫[0,π] P(θ) dθ  (midpoint rule, before dividing by π)."""
    values = np.asarray(values, dtype=float)
    theta = np.asarray(theta, dtype=float)
    if values.size != theta.size:
        raise ValueError("values and theta must have the same size")
    dtheta = np.pi / theta.size
    return float(np.sum(values) * dtheta)


# ── Factory ────────────────────────────────────────────────────────────

def build_line_cycle_trace(
    op: OperatingPoint,
    spec: DesignSpec,
    L_uh: float,
    n_pts: int = 512,
) -> LineCycleTrace:
    """Build a LineCycleTrace from an operating point and inductance.

    Args:
        op: OperatingPoint (provides vin_rms, vm, etc.)
        spec: DesignSpec (provides vout, fsw, pout_total, n_phases, eta)
        L_uh: effective inductance in μH
        n_pts: number of midpoint-sampled theta points (default 512)
    """
    if L_uh <= 0:
        raise ValueError(f"L_uh must be positive, got {L_uh}")

    vm = op.vm
    vout = spec.vout
    fsw = spec.fsw
    n_phases = spec.n_phases
    eta = spec.eta_target
    vin_rms = op.vin_rms
    L_H = L_uh * 1e-6  # μH → H

    # Midpoint sampling: theta_i = (i + 0.5) * pi / N  (avoids 0 and pi)
    theta = (np.arange(n_pts) + 0.5) * np.pi / n_pts
    dtheta = np.pi / n_pts

    # sin_theta with numerical floor to avoid divide-by-zero pathologies
    sin_theta = np.maximum(np.sin(theta), 0.002)
    theta_deg = np.degrees(theta)
    vin_abs = vm * sin_theta

    # Duty cycle — boost PFC
    duty = np.clip(1.0 - vin_abs / vout, 0.02, 0.98)

    # Per-phase line-frequency current
    pin_total = spec.pout_total / eta
    iin_rms_phase = pin_total / n_phases / vin_rms
    iin_pk_phase = np.sqrt(2) * iin_rms_phase
    i_avg_phase = iin_pk_phase * sin_theta

    # Switching ripple: ΔI_pp(θ) = Vin(θ) * D(θ) / (L * fsw)
    delta_i_pp = vin_abs * duty / (L_H * fsw)

    # Peak / valley (valley clamped ≥ 0 to avoid negative loss)
    i_peak = i_avg_phase + 0.5 * delta_i_pp
    i_valley = np.maximum(i_avg_phase - 0.5 * delta_i_pp, 0.0)

    # Local switch-period RMS current squared
    i_rms_local_sq = i_avg_phase**2 + delta_i_pp**2 / 12.0

    # Near zero-crossing / DCM risk marker
    near_zero_or_dcm = duty > 0.95

    return LineCycleTrace(
        theta=theta,
        theta_deg=theta_deg,
        dtheta=dtheta,
        vm=vm,
        sin_theta=sin_theta,
        vin_abs=vin_abs,
        duty=duty,
        i_avg_phase=i_avg_phase,
        delta_i_pp=delta_i_pp,
        i_peak=i_peak,
        i_valley=i_valley,
        i_rms_local_sq=i_rms_local_sq,
        iin_rms_phase=iin_rms_phase,
        iin_pk_phase=iin_pk_phase,
        L_uh=L_uh,
        near_zero_or_dcm=near_zero_or_dcm,
    )
