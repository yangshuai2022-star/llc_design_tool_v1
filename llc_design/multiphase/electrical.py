"""Topology-aware electrical models for fixed-phase multiphase LLC systems.

V8.3 deliberately does *not* model a multiphase converter by taking one
single-phase full-power waveform and shifting it in time.

Implemented electrical topologies:

* 2-phase / 90 deg: two input-parallel/output-parallel LLC cells.  The cells
  share one output voltage and one switching frequency; their powers are
  solved from their individual resonant-tank gain curves.  Component mismatch
  therefore produces current/power imbalance naturally.
* 3-phase / 120 deg: three-leg, floating-neutral Y-connected LLC primary
  represented by a per-phase FHA network.  The three branch currents are
  coupled by the star neutral and the topology-specific per-phase FHA load relation
  Rac_phase = (24/pi^2) * n^2 * Rload for the implemented center-tap/full-wave
  shared-output topology.  This is a different electrical model, not three
  copies of a single-phase LLC.

The 3-phase FHA load relation and floating-neutral coupling are consistent
with recent three-phase LLC literature.  High-fidelity HB/TD extensions should
build on this topology model rather than aliasing the single-phase solver.
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
# 3-phase, 120-degree, Y-connected floating-neutral LLC
# ---------------------------------------------------------------------------

def three_phase_equivalent_ac_load_ohm(turns_ratio: float, dc_load_ohm: float) -> float:
    """Per-phase FHA load for the V8.3 three-module Y-LLC topology.

    The implemented topology follows the classic interleaved Y-LLC structure:
    three half-bridge LLC modules, transformer primaries connected to a floating
    star point, and each secondary full-wave/center-tap rectifier feeding the
    same DC output.  In the balanced case each phase supplies one third of the
    total DC current, hence its effective DC load is ``3*Rload`` and

        Rac_phase = (8/pi^2) * n^2 * (3 Rload)
                  = (24/pi^2) * n^2 * Rload.

    This is intentionally *not* the 6/pi^2 relation used by a different
    three-phase six-pulse rectifier topology.
    """
    if turns_ratio <= 0 or dc_load_ohm <= 0:
        raise ValueError("turns ratio and dc load must be positive")
    return turns_ratio**2 * (24.0 / math.pi**2) * dc_load_ohm


def _tank_with_rac(spec: LLCDesignSpec, rac_ohm: float) -> TankDesign:
    """Synthesize/validate one resonant phase using a topology-specific Rac."""
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


def _star_phase_specs_and_tanks(
    spec: LLCDesignSpec,
    total_power_w: float,
    overrides: Mapping[int, Mapping[str, object]] | None,
) -> tuple[tuple[LLCDesignSpec, ...], tuple[TankDesign, ...], float]:
    # A 3P Y-LLC is physically driven by three half-bridge legs.  AUTO mode
    # therefore synthesizes a topology-correct turns ratio instead of reusing
    # a single/full-bridge ratio.  USER_DEFINED keeps the customer's Np/Ns
    # exactly and merely validates it with the 3P model.
    if TankParameterMode(spec.parameter_mode) == TankParameterMode.AUTO_DESIGN:
        ns = max(int(spec.secondary_turns), 1)
        n_target = spec.vbus_nom_v / max(2.0 * (spec.vout_v + spec.rectifier_equivalent_drop_v), 1e-12)
        np_auto = max(1, int(round(n_target * ns)))
        base = spec.clone(primary_topology=PrimaryTopology.HALF_BRIDGE, primary_turns=np_auto, secondary_turns=ns)
    else:
        base = spec.clone(primary_topology=PrimaryTopology.HALF_BRIDGE)

    # Balanced nominal design point: each rectifier supplies Ptotal/3, so each
    # phase sees three times the system DC load.  This sets the design Q.
    _, rdc_sys = _rectifier_dc_proxy(base, base.vout_v, total_power_w)
    rac_nom = three_phase_equivalent_ac_load_ohm(base.turns_ratio, rdc_sys)
    ov = overrides or {}; specs = []; tanks = []
    for i in range(3):
        ps = base.clone(**dict(ov.get(i, {}))) if i in ov else base.clone()
        # A per-phase override may change Np/Ns, therefore Rac must follow it.
        rac_i = rac_nom if abs(ps.turns_ratio - base.turns_ratio) < 1e-15 else three_phase_equivalent_ac_load_ohm(ps.turns_ratio, rdc_sys)
        specs.append(ps); tanks.append(_tank_with_rac(ps, rac_i))
    return tuple(specs), tuple(tanks), rac_nom


@dataclass(frozen=True)
class _StarFHAState:
    output_voltage_v: float
    total_power_w: float
    phase_power_w: tuple[float, ...]
    phase_current_rms_a: tuple[float, ...]
    phase_gain: tuple[float, ...]
    phase_impedance_angle_deg: tuple[float, ...]
    neutral_voltage_rms_v: complex
    q_effective: tuple[float, ...]
    phase_current_angle_deg: tuple[float, ...]
    effective_source_rms_v: tuple[float, ...]
    phase_rac_ohm: tuple[float, ...]
    phase_power_error_w: tuple[float, ...]
    reflected_load_current_rms_a: tuple[float, ...]
    reflected_load_current_angle_deg: tuple[float, ...]


def _star_fha_state(
    spec: LLCDesignSpec,
    phase_specs: tuple[LLCDesignSpec, ...],
    tanks: tuple[TankDesign, ...],
    *,
    frequency_hz: float,
    vbus_v: float,
    output_voltage_v: float,
    phase_power_w: tuple[float, float, float],
) -> _StarFHAState:
    """Evaluate the coupled star network for an assumed power split.

    Each secondary rectifier feeds the same DC output voltage.  Its FHA load is
    therefore a nonlinear function of that phase's *actual* power share.  The
    caller solves ``Pcalc_i == Passumed_i``; this is what lets component
    mismatch change current sharing without inventing three independent output
    voltages.
    """
    vo = max(float(output_voltage_v), 1e-9)
    w = 2.0 * math.pi * frequency_hz
    # Fundamental RMS of a +/-Vin/2 half-bridge pole square wave.
    vfund = math.sqrt(2.0) / math.pi * vbus_v
    angles = np.deg2rad(np.asarray((0.0, -120.0, 120.0)))
    vs = vfund * np.exp(1j * angles)

    rac = np.asarray([
        _parallel_rac_for_power(ps, vo, max(float(p), 1e-9))
        for ps, p in zip(phase_specs, phase_power_w)
    ], dtype=float)
    ztot = []; zpar = []
    for t, r in zip(tanks, rac):
        zs = 1j * w * t.lr_h + 1.0 / (1j * w * t.cr_f)
        zlm = 1j * w * t.lm_h
        zp = 1.0 / (1.0 / r + 1.0 / zlm)
        zpar.append(zp); ztot.append(zs + zp)
    ztot = np.asarray(ztot, dtype=complex); zpar = np.asarray(zpar, dtype=complex)

    # Floating-star KCL: sum_i (Vi - Vn)/Zi = 0.
    y = 1.0 / ztot
    vn = complex(np.sum(vs * y) / np.sum(y))
    veff = vs - vn
    currents = veff / ztot
    vp = currents * zpar

    # AC load power is the rectifier-input transferred power.  Convert back to
    # useful DC output power after the configured equivalent rectifier drop.
    p_ac = np.abs(vp) ** 2 / rac
    dc_scale = vo / max(vo + spec.rectifier_equivalent_drop_v, 1e-30)
    p_calc = p_ac * dc_scale
    pgain = np.divide(np.abs(vp), np.maximum(np.abs(veff), 1e-30))
    iload = vp / rac
    q = tuple(float(t.zr_ohm / r) for t, r in zip(tanks, rac))
    assumed = np.asarray(phase_power_w, dtype=float)
    return _StarFHAState(
        vo,
        float(np.sum(p_calc)),
        tuple(float(v) for v in p_calc),
        tuple(float(abs(v)) for v in currents),
        tuple(float(v) for v in pgain),
        tuple(float(np.degrees(np.angle(z))) for z in ztot),
        vn,
        q,
        tuple(float(np.degrees(np.angle(v))) for v in currents),
        tuple(float(abs(v)) for v in veff),
        tuple(float(v) for v in rac),
        tuple(float(v) for v in (p_calc - assumed)),
        tuple(float(abs(v)) for v in iload),
        tuple(float(np.degrees(np.angle(v))) for v in iload),
    )


def _solve_star_electrical_point(
    spec: LLCDesignSpec,
    *,
    vbus_v: float,
    load_fraction: float,
    phase_spec_overrides: Mapping[int, Mapping[str, object]] | None,
) -> InterleavedElectricalPoint:
    target_power = spec.pout_w * float(load_fraction)
    phase_specs, tanks, _ = _star_phase_specs_and_tanks(spec, target_power, phase_spec_overrides)
    fmin = max(ps.minimum_frequency_hz for ps in phase_specs); fmax = min(ps.maximum_frequency_hz for ps in phase_specs)
    p0 = np.full(3, target_power / 3.0, dtype=float)

    # Balanced single-module-at-1/3-load gives a strong frequency seed.  The
    # nonlinear solve then permits mismatch and floating-neutral coupling.
    try:
        ps0 = phase_specs[0].clone(pout_w=target_power / 3.0)
        t0 = tanks[0]
        fseed = solve_frequency(t0, ps0, _parallel_rac_for_power(ps0, spec.vout_v, target_power / 3.0), target_gain(ps0, vbus_v)).frequency_hz
    except Exception:
        fseed = float(np.mean([t.fr_hz for t in tanks]))
    x0 = np.r_[math.log(float(np.clip(fseed, fmin, fmax))), np.log(np.maximum(p0, 1e-6))]
    lower = np.r_[math.log(fmin), np.log(np.full(3, max(target_power * 1e-6, 1e-6)))]
    upper = np.r_[math.log(fmax), np.log(np.full(3, max(target_power * 3.0, 1.0)))]

    def residual(x: np.ndarray) -> np.ndarray:
        fs = float(math.exp(x[0])); powers = tuple(float(v) for v in np.exp(x[1:]))
        st = _star_fha_state(spec, phase_specs, tanks, frequency_hz=fs, vbus_v=vbus_v, output_voltage_v=spec.vout_v, phase_power_w=powers)
        scale = max(target_power / 3.0 * 0.005, 0.5)
        out = [(pc - pa) / scale for pc, pa in zip(st.phase_power_w, powers)]
        out.append((sum(powers) - target_power) / max(target_power * 0.005, 1.0))
        return np.asarray(out, dtype=float)

    sol = least_squares(residual, x0, bounds=(lower, upper), xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=3000, x_scale="jac")
    fs = float(math.exp(sol.x[0])); powers = tuple(float(v) for v in np.exp(sol.x[1:]))
    state = _star_fha_state(spec, phase_specs, tanks, frequency_hz=fs, vbus_v=vbus_v, output_voltage_v=spec.vout_v, phase_power_w=powers)
    rnorm = float(np.linalg.norm(residual(sol.x), ord=np.inf)); exact = bool(sol.success and rnorm <= 2e-3)
    warnings: list[str] = []
    if not exact:
        warnings.append(f"3P Y-LLC self-consistent load-share solution residual is {rnorm:.3e}; result is a bounded best-fit.")

    total = float(sum(powers)); avg = total / 3.0; phase_points = []
    for i, (off, t, p, q, g, angle, ir) in enumerate(zip(fixed_phase_offsets_deg(3), tanks, powers, state.q_effective, state.phase_gain, state.phase_impedance_angle_deg, state.phase_current_rms_a)):
        phase_points.append(PhaseElectricalPoint(i + 1, off, p, 100.0 * p / max(total, 1e-12), q, g, g, angle, ir, t, state.phase_current_angle_deg[i], state.effective_source_rms_v[i], phase_specs[i].turns_ratio, phase_specs[i].primary_turns, phase_specs[i].secondary_turns, state.reflected_load_current_rms_a[i], state.reflected_load_current_angle_deg[i]))
    imbalance = (max(powers) - min(powers)) / max(avg, 1e-12) * 100.0
    if imbalance > 2.0:
        warnings.append(f"3P floating-neutral tank mismatch produces {imbalance:.3f}% phase-power imbalance.")
    neutral_ref = math.sqrt(2.0) / math.pi * vbus_v
    if abs(state.neutral_voltage_rms_v) > 0.01 * neutral_ref:
        warnings.append(f"Floating-star neutral displacement is {abs(state.neutral_voltage_rms_v):.3f} Vrms; phase coupling is active.")
    return InterleavedElectricalPoint(3, fixed_phase_offsets_deg(3), fs, spec.vout_v, total, tuple(phase_points), tuple(warnings), exact, rnorm, MultiphaseTopology.THREE_PHASE_STAR_120)


def _solve_star_gain_curve(
    spec: LLCDesignSpec,
    *,
    vbus_v: float,
    load_fraction: float,
    phase_spec_overrides: Mapping[int, Mapping[str, object]] | None,
    points: int,
) -> SystemGainCurve:
    target_power_nom = spec.pout_w * float(load_fraction)
    phase_specs, tanks, _ = _star_phase_specs_and_tanks(spec, target_power_nom, phase_spec_overrides)
    fmin = max(ps.minimum_frequency_hz for ps in phase_specs); fmax = min(ps.maximum_frequency_hz for ps in phase_specs)
    freqs = np.geomspace(fmin, fmax, int(points)); vo_arr = np.full_like(freqs, np.nan); p_arrays = [np.full_like(freqs, np.nan) for _ in range(3)]
    # The user-selected work point defines the physical output resistance used
    # for a frequency sweep; output voltage/power are allowed to move.
    rload = spec.vout_v**2 / max(target_power_nom, 1e-12)
    pseed = np.full(3, target_power_nom / 3.0, dtype=float)
    prev = np.r_[math.log(max(spec.vout_v, 0.1)), np.log(np.maximum(pseed, 1e-6))]
    lower = np.r_[math.log(max(0.03 * spec.vout_v, 0.02)), np.log(np.full(3, max(target_power_nom * 1e-7, 1e-6)))]
    upper = np.r_[math.log(max(3.0 * spec.vout_v, 2.0)), np.log(np.full(3, max(target_power_nom * 4.0, 1.0)))]

    for k, f in enumerate(freqs):
        def residual(x: np.ndarray) -> np.ndarray:
            vo = float(math.exp(x[0])); powers = tuple(float(v) for v in np.exp(x[1:]))
            st = _star_fha_state(spec, phase_specs, tanks, frequency_hz=float(f), vbus_v=vbus_v, output_voltage_v=vo, phase_power_w=powers)
            p_load = vo * vo / rload
            phase_scale = max(p_load / 3.0 * 0.01, 0.25)
            out = [(pc - pa) / phase_scale for pc, pa in zip(st.phase_power_w, powers)]
            out.append((sum(powers) - p_load) / max(p_load * 0.01, 0.5))
            return np.asarray(out, dtype=float)

        nominal = np.r_[math.log(max(spec.vout_v, 0.1)), np.log(np.maximum(pseed, 1e-6))]
        candidates = []
        for seed in (prev, nominal):
            sol = least_squares(residual, seed, bounds=(lower, upper), xtol=3e-11, ftol=3e-11, gtol=3e-11, max_nfev=1000, x_scale="jac")
            rn = float(np.linalg.norm(residual(sol.x), ord=np.inf)); candidates.append((rn, sol))
        rn, sol = min(candidates, key=lambda z: z[0])
        if (not sol.success) or rn > 0.08:
            continue
        prev = sol.x.copy(); vo = float(math.exp(sol.x[0])); powers = np.exp(sol.x[1:]); vo_arr[k] = vo
        for i, pwr in enumerate(powers): p_arrays[i][k] = float(pwr)
    return SystemGainCurve(freqs, vo_arr / vbus_v, vo_arr, tuple(p_arrays), target_power_nom, MultiphaseTopology.THREE_PHASE_STAR_120)


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

    2P solves common-output load sharing for two independent cells.  3P uses
    the Y-connected floating-neutral FHA network and topology-specific per-phase Rac relation.
    The curve is never copied from a single-phase full-power model.
    """
    vbus = float(spec.vbus_nom_v if vbus_v is None else vbus_v)
    topology = topology_for_phase_count(phase_count)
    if topology == MultiphaseTopology.TWO_PHASE_PARALLEL_90:
        return _solve_parallel_gain_curve(spec, vbus_v=vbus, load_fraction=load_fraction, phase_spec_overrides=phase_spec_overrides, points=points)
    return _solve_star_gain_curve(spec, vbus_v=vbus, load_fraction=load_fraction, phase_spec_overrides=phase_spec_overrides, points=points)


__all__ = [
    "MultiphaseTopology", "PhaseElectricalPoint", "InterleavedElectricalPoint", "SystemGainCurve",
    "fixed_phase_offsets_deg", "topology_for_phase_count", "three_phase_equivalent_ac_load_ohm",
    "solve_phase_power_at_frequency", "solve_interleaved_electrical_point", "solve_system_gain_curve",
]
