"""Topology-aware electrical models for fixed-phase multiphase LLC systems.

V8.3 deliberately does *not* model a multiphase converter by taking one
single-phase full-power waveform and shifting it in time.

Implemented electrical topologies:

* 2-phase / 90 deg: two input-parallel/output-parallel LLC cells.  The cells
  share one output voltage and one switching frequency; their powers are
  solved from their individual resonant-tank gain curves.  Component mismatch
  therefore produces current/power imbalance naturally.
* 3-phase / 120 deg: three switching legs, Y-connected resonant tanks and
  transformer phases, and one shared three-phase six-pulse rectifier.  The FHA
  load is Rac=(6/pi^2)*n^2*Rdc and the topology gain is M=n*Vo/Vin.  High-
  fidelity HB/TD use dedicated three-phase hybrid state equations rather than
  copies of the single-phase solver.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Mapping

import numpy as np
from scipy.optimize import brentq, least_squares

from ..core.spec import LLCDesignSpec, TankParameterMode, PrimaryTopology
from ..core.tank import (
    TankDesign,
    design_tank,
    equivalent_ac_load_ohm,
    gain,
    target_gain,
    tank_state,
    solve_frequency,
    find_gain_roots,
)


class MultiphaseTopology(str, Enum):
    TWO_PHASE_PARALLEL_90 = "2P_PARALLEL_90"
    THREE_PHASE_STAR_120 = "3P_STAR_120"


@dataclass(frozen=True)
class PhaseElectricalPoint:
    index: int
    phase_offset_deg: float
    output_power_w: float
    share_percent: float
    q_effective: float
    achieved_gain: float
    required_gain: float
    input_phase_deg: float
    resonant_current_rms_a: float
    tank: TankDesign
    resonant_current_angle_deg: float = 0.0
    effective_source_rms_v: float = 0.0
    turns_ratio: float = 0.0
    primary_turns: int = 0
    secondary_turns: int = 0
    reflected_load_current_rms_a: float = 0.0
    reflected_load_current_angle_deg: float = 0.0


@dataclass(frozen=True)
class InterleavedElectricalPoint:
    phase_count: int
    phase_offsets_deg: tuple[float, ...]
    switching_frequency_hz: float
    output_voltage_v: float
    total_output_power_w: float
    phases: tuple[PhaseElectricalPoint, ...]
    warnings: tuple[str, ...]
    exact_common_solution: bool = True
    solver_residual: float = 0.0
    topology: MultiphaseTopology = MultiphaseTopology.TWO_PHASE_PARALLEL_90


@dataclass(frozen=True)
class SystemGainCurve:
    frequency_hz: np.ndarray
    normalized_gain: np.ndarray  # physical DC conversion ratio Vo/Vin
    output_voltage_v: np.ndarray
    phase_power_w: tuple[np.ndarray, ...]
    total_power_w: float
    topology: MultiphaseTopology = MultiphaseTopology.TWO_PHASE_PARALLEL_90


def topology_for_phase_count(phase_count: int) -> MultiphaseTopology:
    if phase_count == 2:
        return MultiphaseTopology.TWO_PHASE_PARALLEL_90
    if phase_count == 3:
        return MultiphaseTopology.THREE_PHASE_STAR_120
    raise ValueError("V8 multiphase LLC supports fixed 2-phase or 3-phase systems")


def fixed_phase_offsets_deg(phase_count: int) -> tuple[float, ...]:
    if phase_count == 2:
        return (0.0, 90.0)
    if phase_count == 3:
        return (0.0, 120.0, 240.0)
    raise ValueError("V8 multiphase LLC supports fixed 2-phase or 3-phase systems")


def _rectifier_dc_proxy(spec: LLCDesignSpec, output_voltage_v: float, power_w: float) -> tuple[float, float]:
    """Return (Vrect, Rdc-equivalent) including the configured rectifier drop.

    The same power convention is used by the single-phase operating-point
    model so manual/auto comparisons stay coherent.
    """
    vo = max(float(output_voltage_v), 1e-12)
    p = max(float(power_w), 1e-12)
    vrect = vo + spec.rectifier_equivalent_drop_v
    transferred = p * (1.0 + spec.rectifier_equivalent_drop_v / vo)
    return vrect, vrect * vrect / transferred


# ---------------------------------------------------------------------------
# 2-phase, 90-degree, independent parallel LLC cells
# ---------------------------------------------------------------------------

def _parallel_phase_specs(spec: LLCDesignSpec, phase_count: int, overrides: Mapping[int, Mapping[str, object]] | None) -> tuple[LLCDesignSpec, ...]:
    base = spec.clone(pout_w=spec.pout_w / phase_count)
    ov = overrides or {}
    return tuple(base.clone(**dict(ov.get(i, {}))) if i in ov else base for i in range(phase_count))


def _parallel_rac_for_power(spec: LLCDesignSpec, output_voltage_v: float, power_w: float) -> float:
    vrect, rdc = _rectifier_dc_proxy(spec, output_voltage_v, power_w)
    # equivalent_ac_load_ohm expects V/P; using Rdc directly is equivalent.
    del vrect
    return (8.0 / math.pi**2) * spec.turns_ratio**2 * rdc


def solve_phase_power_at_frequency(
    spec: LLCDesignSpec,
    tank: TankDesign,
    *,
    frequency_hz: float,
    vbus_v: float,
    output_voltage_v: float,
    max_power_factor: float = 2.5,
) -> float | None:
    required = target_gain(spec.clone(vout_v=output_voltage_v), vbus_v)
    rated = spec.pout_w
    pmin = max(rated * 1e-5, 1e-6)
    pmax = max(rated * max_power_factor, pmin * 10)
    grid = np.geomspace(pmin, pmax, 180)
    err = np.empty_like(grid)
    for i, p in enumerate(grid):
        err[i] = gain(tank, frequency_hz, _parallel_rac_for_power(spec, output_voltage_v, float(p))) - required
    roots: list[float] = []
    for i in range(len(grid) - 1):
        y0, y1 = float(err[i]), float(err[i + 1])
        if abs(y0) < 1e-10:
            roots.append(float(grid[i]))
        if y0 * y1 < 0:
            roots.append(
                float(
                    brentq(
                        lambda p: gain(tank, frequency_hz, _parallel_rac_for_power(spec, output_voltage_v, p)) - required,
                        float(grid[i]),
                        float(grid[i + 1]),
                        xtol=1e-8,
                        rtol=1e-10,
                    )
                )
            )
    if not roots:
        gnl = gain(tank, frequency_hz, _parallel_rac_for_power(spec, output_voltage_v, pmin))
        if abs(gnl - required) <= max(2e-4, abs(required) * 2e-4):
            return 0.0
        return None
    return min(roots, key=lambda p: abs(math.log(max(p, 1e-12) / rated)))


def _solve_parallel_electrical_point(
    spec: LLCDesignSpec,
    *,
    vbus_v: float,
    load_fraction: float,
    phase_spec_overrides: Mapping[int, Mapping[str, object]] | None,
) -> InterleavedElectricalPoint:
    phase_count = 2
    offsets = fixed_phase_offsets_deg(phase_count)
    phase_specs = _parallel_phase_specs(spec, phase_count, phase_spec_overrides)
    tanks = tuple(design_tank(ps) for ps in phase_specs)
    target_power = spec.pout_w * float(load_fraction)
    vo = spec.vout_v
    fmin = max(ps.minimum_frequency_hz for ps in phase_specs)
    fmax = min(ps.maximum_frequency_hz for ps in phase_specs)
    p_rated = np.asarray([ps.pout_w for ps in phase_specs], dtype=float)
    p0 = np.full(phase_count, target_power / phase_count, dtype=float)

    fseeds = []
    for ps, t, p in zip(phase_specs, tanks, p0):
        rac = _parallel_rac_for_power(ps, vo, float(p))
        try:
            fseeds.append(solve_frequency(t, ps, rac, target_gain(ps, vbus_v)).frequency_hz)
        except Exception:
            fseeds.append(t.fr_hz)
    f0 = float(np.clip(np.mean(fseeds), fmin, fmax))
    x0 = np.r_[math.log(f0), np.log(np.maximum(p0, 1e-6))]
    lower = np.r_[math.log(fmin), np.log(np.maximum(p_rated * 1e-5, 1e-6))]
    upper = np.r_[math.log(fmax), np.log(np.maximum(p_rated * 4.0, 1.0))]

    def residual(x: np.ndarray) -> np.ndarray:
        fs = float(math.exp(x[0])); powers = np.exp(x[1:])
        out = []
        for ps, t, p in zip(phase_specs, tanks, powers):
            required = target_gain(ps, vbus_v)
            g = gain(t, fs, _parallel_rac_for_power(ps, vo, float(p)))
            out.append((g - required) / max(abs(required) * 0.005, 5e-4))
        out.append((float(np.sum(powers)) - target_power) / max(target_power * 0.005, 1.0))
        return np.asarray(out, dtype=float)

    sol = least_squares(residual, x0, bounds=(lower, upper), xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=2500, x_scale="jac")
    fs = float(math.exp(sol.x[0]))
    powers = tuple(float(v) for v in np.exp(sol.x[1:]))
    rnorm = float(np.linalg.norm(residual(sol.x), ord=np.inf))
    exact = bool(sol.success and rnorm <= 2e-3)
    if not sol.success and rnorm > 1.0:
        raise RuntimeError(f"2P common-frequency solve failed: residual={rnorm:.3e}, message={sol.message}")

    total = float(sum(powers)); avg = total / phase_count; phase_points = []; warnings = []
    if not exact:
        warnings.append(
            f"No exact common-frequency 2P load-sharing solution (normalized residual {rnorm:.3e}); "
            "displayed powers are the bounded best-fit and indicate current-sharing/collapse risk."
        )
    for i, (off, ps, t, p) in enumerate(zip(offsets, phase_specs, tanks, powers)):
        rac = _parallel_rac_for_power(ps, vo, max(p, 1e-9)); st = tank_state(t, fs, rac); required = target_gain(ps, vbus_v)
        vbridge_rms = (2.0 * math.sqrt(2.0) / math.pi) * ps.bridge_gain * vbus_v
        ir = abs(vbridge_rms / st.z_input_ohm)
        phase_points.append(PhaseElectricalPoint(i + 1, off, p, 100 * p / max(total, 1e-12), t.zr_ohm / rac, st.gain, required, st.input_phase_deg, ir, t, off - st.input_phase_deg, vbridge_rms, ps.turns_ratio, ps.primary_turns, ps.secondary_turns))
    imbalance = (max(powers) - min(powers)) / max(avg, 1e-12) * 100
    if imbalance > 2.0:
        warnings.append(f"Resonant-tank mismatch produces {imbalance:.3f}% phase-power imbalance at the common switching frequency.")
    return InterleavedElectricalPoint(phase_count, offsets, fs, vo, total, tuple(phase_points), tuple(warnings), exact, rnorm, MultiphaseTopology.TWO_PHASE_PARALLEL_90)


def _solve_parallel_gain_curve(
    spec: LLCDesignSpec,
    *,
    vbus_v: float,
    load_fraction: float,
    phase_spec_overrides: Mapping[int, Mapping[str, object]] | None,
    points: int,
) -> SystemGainCurve:
    phase_count = 2
    target_power = spec.pout_w * load_fraction
    phase_specs = _parallel_phase_specs(spec, phase_count, phase_spec_overrides)
    tanks = tuple(design_tank(ps) for ps in phase_specs)
    fmin = max(ps.minimum_frequency_hz for ps in phase_specs); fmax = min(ps.maximum_frequency_hz for ps in phase_specs)
    freqs = np.geomspace(fmin, fmax, int(points)); vo_arr = np.full_like(freqs, np.nan); p_arrays = [np.full_like(freqs, np.nan) for _ in range(phase_count)]
    p_rated = np.asarray([ps.pout_w for ps in phase_specs], dtype=float)
    prev = np.r_[math.log(max(spec.vout_v, 0.1)), np.log(np.full(phase_count, target_power / phase_count))]
    lower = np.r_[math.log(max(0.03 * spec.vout_v, 0.02)), np.log(np.maximum(p_rated * 1e-5, 1e-6))]
    upper = np.r_[math.log(max(3.0 * spec.vout_v, 2.0)), np.log(np.maximum(p_rated * 5.0, 1.0))]
    for k, f in enumerate(freqs):
        def residual(x: np.ndarray) -> np.ndarray:
            vo = float(math.exp(x[0])); powers = np.exp(x[1:]); out = []
            for ps, t, p in zip(phase_specs, tanks, powers):
                required = target_gain(ps.clone(vout_v=vo), vbus_v)
                g = gain(t, float(f), _parallel_rac_for_power(ps, vo, float(p)))
                out.append((g - required) / max(abs(required) * 0.01, 1e-3))
            out.append((float(np.sum(powers)) - target_power) / max(target_power * 0.01, 1.0))
            return np.asarray(out, dtype=float)
        candidates = []
        nominal = np.r_[math.log(max(spec.vout_v, 0.1)), np.log(np.full(phase_count, target_power / phase_count))]
        for seed in (prev, nominal):
            sol = least_squares(residual, seed, bounds=(lower, upper), xtol=3e-11, ftol=3e-11, gtol=3e-11, max_nfev=800, x_scale="jac")
            rn = float(np.linalg.norm(residual(sol.x), ord=np.inf)); candidates.append((rn, sol))
        rn, sol = min(candidates, key=lambda z: z[0])
        if (not sol.success) or rn > 0.08:
            continue
        prev = sol.x.copy(); vo = float(math.exp(sol.x[0])); powers = np.exp(sol.x[1:]); vo_arr[k] = vo
        for i, p in enumerate(powers): p_arrays[i][k] = float(p)
    return SystemGainCurve(freqs, vo_arr / vbus_v, vo_arr, tuple(p_arrays), target_power, MultiphaseTopology.TWO_PHASE_PARALLEL_90)


# ---------------------------------------------------------------------------
# 3-phase, 120-degree, Y-connected tanks + shared 6-pulse bridge rectifier
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ThreePhaseDesignContext:
    """Balanced fixed-120° three-phase LLC design context.

    This topology is *not* three independent single-phase rectifiers.  Three
    resonant branches / transformer phases are Y connected and feed one shared
    three-phase full-bridge rectifier.  The topology-specific FHA reflected
    load is therefore

        Rac = (6/pi^2) * n^2 * Rdc

    and the normalized DC gain is M = n*Vrect/Vin.

    ``spec`` is a topology-context copy. AUTO_DESIGN may round Np/Ns for the
    three-phase resonance point; USER_DEFINED preserves the customer's turns
    exactly. ``tank`` always uses the topology-specific Rac when synthesizing Q.
    """
    spec: LLCDesignSpec
    tank: TankDesign
    rac_ohm: float
    dc_load_ohm: float
    total_power_w: float


def three_phase_equivalent_ac_load_ohm(turns_ratio: float, dc_load_ohm: float) -> float:
    """FHA load reflected into one phase of the shared 3P bridge topology.

    For the balanced Y/Y three-phase LLC with a common six-pulse rectifier,
    the rectifier-side equivalent resistance is 6*Rdc/pi^2. Reflection through
    the transformer adds n^2.
    """
    if turns_ratio <= 0 or dc_load_ohm <= 0:
        raise ValueError("turns ratio and dc load must be positive")
    return turns_ratio**2 * (6.0 / math.pi**2) * dc_load_ohm


def _tank_with_rac(spec: LLCDesignSpec, rac_ohm: float) -> TankDesign:
    """Synthesize/validate one resonant phase using topology-specific Rac."""
    mode = TankParameterMode(spec.parameter_mode)
    if mode == TankParameterMode.USER_DEFINED:
        spec.validate()
        assert spec.user_lr_h is not None and spec.user_cr_f is not None and spec.user_lm_h is not None
        lr = float(spec.user_lr_h); cr = float(spec.user_cr_f); lm = float(spec.user_lm_h)
        zr = math.sqrt(lr / cr); fr = 1.0 / (2.0 * math.pi * math.sqrt(lr * cr)); ln = lm / lr; q = zr / rac_ohm
        return TankDesign(lr, cr, lm, rac_ohm, zr, ln, q, fr)
    spec.validate()
    zr = spec.q_full_load * rac_ohm; wr = 2.0 * math.pi * spec.resonant_frequency_hz
    lr = zr / wr; cr = 1.0 / (wr * zr); lm = spec.ln_ratio * lr
    return TankDesign(lr, cr, lm, rac_ohm, zr, spec.ln_ratio, spec.q_full_load, spec.resonant_frequency_hz)


def three_phase_design_context(
    spec: LLCDesignSpec,
    *,
    total_power_w: float | None = None,
) -> ThreePhaseDesignContext:
    """Return the balanced 3P topology spec/tank without single-phase fan-out."""
    p = float(spec.pout_w if total_power_w is None else total_power_w)
    if p <= 0:
        raise ValueError("three-phase total power must be positive")

    # Each primary phase is driven by one half-bridge leg, but the *three-phase*
    # DC gain is M=n*Vrect/Vin.  Do not use the single-phase half-bridge 0.5 gain
    # when choosing the transformer ratio.  USER_DEFINED never changes Np/Ns.
    if TankParameterMode(spec.parameter_mode) == TankParameterMode.AUTO_DESIGN:
        ns = max(int(spec.secondary_turns), 1)
        n_target = spec.vbus_nom_v / max(spec.vout_v + spec.rectifier_equivalent_drop_v, 1e-12)
        np_auto = max(1, int(round(n_target * ns)))
        ps = spec.clone(
            primary_topology=PrimaryTopology.HALF_BRIDGE,
            primary_turns=np_auto,
            secondary_turns=ns,
            pout_w=p,
        )
    else:
        ps = spec.clone(primary_topology=PrimaryTopology.HALF_BRIDGE, pout_w=p)

    _vrect, rdc = _rectifier_dc_proxy(ps, ps.vout_v, p)
    rac = three_phase_equivalent_ac_load_ohm(ps.turns_ratio, rdc)
    tank = _tank_with_rac(ps, rac)
    return ThreePhaseDesignContext(ps, tank, rac, rdc, p)


def _select_3p_fha_frequency(tank: TankDesign, *, rac_ohm: float, required_m: float,
                             fmin_hz: float, fmax_hz: float) -> float:
    roots = find_gain_roots(tank, rac_ohm, required_m, fmin_hz, fmax_hz)
    if not roots:
        freqs = np.geomspace(fmin_hz, fmax_hz, 800)
        vals = np.asarray([gain(tank, float(f), rac_ohm) for f in freqs])
        raise RuntimeError(
            f"3P FHA required M={required_m:.5f} is outside available "
            f"{float(np.min(vals)):.5f}..{float(np.max(vals)):.5f}"
        )
    if required_m <= 1.0:
        preferred = [f for f in roots if f >= tank.fr_hz * (1.0 - 1e-8)]
        return float(min(preferred) if preferred else min(roots, key=lambda f: abs(f - tank.fr_hz)))
    preferred = [f for f in roots if f <= tank.fr_hz * (1.0 + 1e-8)]
    return float(max(preferred) if preferred else min(roots, key=lambda f: abs(f - tank.fr_hz)))


def _solve_star_electrical_point(
    spec: LLCDesignSpec,
    *,
    vbus_v: float,
    load_fraction: float,
    phase_spec_overrides: Mapping[int, Mapping[str, object]] | None,
) -> InterleavedElectricalPoint:
    if phase_spec_overrides:
        raise NotImplementedError(
            "The shared 3P bridge high-accuracy model currently assumes balanced tank parameters. "
            "Per-phase mismatch cannot be represented by independent phase Rac values; use the balanced "
            "3P solver until the full asymmetric diode-complementarity model is available."
        )
    ptotal = spec.pout_w * float(load_fraction)
    ctx = three_phase_design_context(spec, total_power_w=ptotal)
    ps, tank, rac = ctx.spec, ctx.tank, ctx.rac_ohm
    vrect = spec.vout_v + spec.rectifier_equivalent_drop_v
    required_m = ps.turns_ratio * vrect / vbus_v
    fs = _select_3p_fha_frequency(
        tank, rac_ohm=rac, required_m=required_m,
        fmin_hz=ps.minimum_frequency_hz, fmax_hz=ps.maximum_frequency_hz,
    )
    st = tank_state(tank, fs, rac)
    source_rms = math.sqrt(2.0) / math.pi * vbus_v
    ir_rms = abs(source_rms / st.z_input_ohm)
    vp = (source_rms / st.z_input_ohm) * st.z_parallel_ohm
    iload = vp / rac
    q = tank.zr_ohm / rac
    phase_power = ptotal / 3.0
    phases = tuple(
        PhaseElectricalPoint(
            i + 1, off, phase_power, 100.0 / 3.0, q,
            st.gain, required_m, st.input_phase_deg, ir_rms, tank,
            off - st.input_phase_deg, source_rms, ps.turns_ratio,
            ps.primary_turns, ps.secondary_turns,
            abs(iload), math.degrees(math.atan2(iload.imag, iload.real)) + off,
        )
        for i, off in enumerate(fixed_phase_offsets_deg(3))
    )
    warnings = (
        "3P FHA uses the balanced Y-connected resonant tanks and one shared six-pulse rectifier; "
        "it is not a three-cell single-phase fan-out.",
    )
    return InterleavedElectricalPoint(
        3, fixed_phase_offsets_deg(3), fs, spec.vout_v, ptotal, phases,
        warnings, True, abs(st.gain - required_m), MultiphaseTopology.THREE_PHASE_STAR_120,
    )


def _solve_star_gain_curve(
    spec: LLCDesignSpec,
    *,
    vbus_v: float,
    load_fraction: float,
    phase_spec_overrides: Mapping[int, Mapping[str, object]] | None,
    points: int,
) -> SystemGainCurve:
    if phase_spec_overrides:
        raise NotImplementedError("3P shared-bridge gain curve currently supports balanced phase parameters only")
    pnom = spec.pout_w * float(load_fraction)
    ctx = three_phase_design_context(spec, total_power_w=pnom)
    ps, tank, rac = ctx.spec, ctx.tank, ctx.rac_ohm
    freqs = np.geomspace(ps.minimum_frequency_hz, ps.maximum_frequency_hz, int(points))
    m = np.asarray([gain(tank, float(f), rac) for f in freqs], dtype=float)
    vo = np.maximum(m * vbus_v / ps.turns_ratio - spec.rectifier_equivalent_drop_v, 0.0)
    rload = spec.vout_v**2 / max(pnom, 1e-30)
    ptotal = vo * vo / rload
    pphase = tuple((ptotal / 3.0).copy() for _ in range(3))
    return SystemGainCurve(
        freqs, vo / vbus_v, vo, pphase, pnom,
        MultiphaseTopology.THREE_PHASE_STAR_120,
    )


# ---------------------------------------------------------------------------
# Public dispatch
# ---------------------------------------------------------------------------

def solve_interleaved_electrical_point(
    spec: LLCDesignSpec,
    phase_count: int,
    *,
    vbus_v: float | None = None,
    load_fraction: float = 1.0,
    phase_spec_overrides: Mapping[int, Mapping[str, object]] | None = None,
) -> InterleavedElectricalPoint:
    vbus = float(spec.vbus_nom_v if vbus_v is None else vbus_v)
    topology = topology_for_phase_count(phase_count)
    if topology == MultiphaseTopology.TWO_PHASE_PARALLEL_90:
        return _solve_parallel_electrical_point(spec, vbus_v=vbus, load_fraction=load_fraction, phase_spec_overrides=phase_spec_overrides)
    return _solve_star_electrical_point(spec, vbus_v=vbus, load_fraction=load_fraction, phase_spec_overrides=phase_spec_overrides)


def solve_system_gain_curve(
    spec: LLCDesignSpec,
    phase_count: int,
    *,
    vbus_v: float | None = None,
    load_fraction: float = 1.0,
    phase_spec_overrides: Mapping[int, Mapping[str, object]] | None = None,
    points: int = 180,
) -> SystemGainCurve:
    """Return the topology-specific *system* DC gain curve Vo/Vin.

    2P solves common-output load sharing for two independent cells. 3P uses
    the Y/Y shared-six-pulse-rectifier topology-specific FHA relation. The
    curve is never copied from a single-phase full-power model.
    """
    vbus = float(spec.vbus_nom_v if vbus_v is None else vbus_v)
    topology = topology_for_phase_count(phase_count)
    if topology == MultiphaseTopology.TWO_PHASE_PARALLEL_90:
        return _solve_parallel_gain_curve(spec, vbus_v=vbus, load_fraction=load_fraction, phase_spec_overrides=phase_spec_overrides, points=points)
    return _solve_star_gain_curve(spec, vbus_v=vbus, load_fraction=load_fraction, phase_spec_overrides=phase_spec_overrides, points=points)


__all__ = [
    "MultiphaseTopology", "PhaseElectricalPoint", "InterleavedElectricalPoint", "SystemGainCurve",
    "fixed_phase_offsets_deg", "topology_for_phase_count", "three_phase_equivalent_ac_load_ohm",
    "ThreePhaseDesignContext", "three_phase_design_context",
    "solve_phase_power_at_frequency", "solve_interleaved_electrical_point", "solve_system_gain_curve",
]
