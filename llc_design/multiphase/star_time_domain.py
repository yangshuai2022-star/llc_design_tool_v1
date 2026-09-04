"""High-accuracy normalized time-domain model for the fixed 120° three-phase Y LLC.

This module implements the balanced three-phase LLC topology described by
Shen et al., IET Electric Power Applications 15 (2021) 114-127:

* three half-bridge legs, fixed 120 degree phase displacement;
* three Y-connected resonant branches / transformers;
* one shared three-phase six-pulse bridge rectifier;
* the interaction states 1/3P/N, 2/3P/N, O-P/O-N and O.

The solver is deliberately separate from the single-phase LLC time-domain
engine.  It solves the preferred above-resonance and below-resonance state
sequences using the normalized piecewise analytical state equations and
steady-state constraints.  No single-phase waveform fan-out is used.

Normalization
-------------
Vb = (2/3) Vin
Ib = Vb / Z0
Z0 = sqrt(Lr/Cr)
theta = w0*t, w0 = 1/sqrt(Lr*Cr)
lambda = Lr/Lm
M = n*Vrect/Vin

The physical total transferred power is evaluated from the instantaneous
magnetizing clamp voltage and reflected load current rather than relying on a
paper-specific shorthand power integral.  This also makes the implementation
clear when an equivalent rectifier drop is configured.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from ..core.spec import LLCDesignSpec
from ..core.tank import TankDesign


class ThreePhaseMode(str, Enum):
    ABOVE_RESONANCE = "ABOVE_RESONANCE"
    BELOW_RESONANCE = "BELOW_RESONANCE"


@dataclass(frozen=True)
class ThreePhaseSegment:
    label: str
    theta_start: float
    theta_end: float
    source_norm: float
    clamp_norm: float | None


@dataclass(frozen=True)
class ThreePhaseTDResult:
    converged: bool
    mode: ThreePhaseMode
    switching_frequency_hz: float
    resonant_frequency_hz: float
    normalized_half_period: float
    normalized_gain: float          # M = n*Vrect/Vin
    dc_gain_vo_vin: float           # useful Vo/Vin
    output_voltage_v: float
    output_power_w: float
    transferred_power_w: float
    residual_norm: float
    event_theta: float
    event_name: str
    initial_state: tuple[float, float, float]
    time_s: NDArray[np.float64]
    theta: NDArray[np.float64]
    mcr: NDArray[np.float64]
    jlr: NDArray[np.float64]
    jlm: NDArray[np.float64]
    mlm: NDArray[np.float64]
    source_norm: NDArray[np.float64]
    load_current_norm: NDArray[np.float64]
    i_resonant_a: NDArray[np.float64]
    i_magnetizing_a: NDArray[np.float64]
    i_reflected_load_a: NDArray[np.float64]
    i_secondary_phase_a: NDArray[np.float64]
    v_cr_v: NDArray[np.float64]
    v_lm_v: NDArray[np.float64]
    v_phase_input_v: NDArray[np.float64]
    i_output_rectified_a: NDArray[np.float64]
    segments: tuple[ThreePhaseSegment, ...]
    warnings: tuple[str, ...] = ()

    @property
    def fr_ratio(self) -> float:
        return self.switching_frequency_hz / self.resonant_frequency_hz

    @property
    def resonant_current_rms_a(self) -> float:
        return float(np.sqrt(np.mean(self.i_resonant_a**2)))

    @property
    def resonant_current_peak_a(self) -> float:
        return float(np.max(np.abs(self.i_resonant_a)))

    @property
    def secondary_current_rms_a(self) -> float:
        return float(np.sqrt(np.mean(self.i_secondary_phase_a**2)))


@dataclass(frozen=True)
class _NormState:
    mcr: float
    jlr: float
    jlm: float


def _clamped_end(s: _NormState, dtheta: float, source: float, clamp: float, lam: float) -> _NormState:
    c = math.cos(dtheta)
    sn = math.sin(dtheta)
    eq = source - clamp
    dm = s.mcr - eq
    return _NormState(
        eq + dm * c + s.jlr * sn,
        -dm * sn + s.jlr * c,
        s.jlm + lam * clamp * dtheta,
    )


def _open_end(s: _NormState, dtheta: float, source: float, lam: float) -> _NormState:
    # Lr and Lm are in series in O.  lambda=Lr/Lm, so normalized natural
    # frequency k=sqrt(Lr/(Lr+Lm))=sqrt(lambda/(1+lambda)).
    k = math.sqrt(lam / (1.0 + lam))
    kd = k * dtheta
    c = math.cos(kd)
    sn = math.sin(kd)
    dm = s.mcr - source
    mcr = source + dm * c + (s.jlr / k) * sn
    jlr = -k * dm * sn + s.jlr * c
    return _NormState(mcr, jlr, jlr)


def _sample_clamped(s: _NormState, tau: NDArray[np.float64], source: float, clamp: float, lam: float):
    c = np.cos(tau); sn = np.sin(tau); eq = source - clamp; dm = s.mcr - eq
    mcr = eq + dm*c + s.jlr*sn
    jlr = -dm*sn + s.jlr*c
    jlm = s.jlm + lam*clamp*tau
    mlm = np.full_like(tau, clamp)
    return mcr, jlr, jlm, mlm


def _sample_open(s: _NormState, tau: NDArray[np.float64], source: float, lam: float):
    k = math.sqrt(lam/(1.0+lam)); kd=k*tau; c=np.cos(kd); sn=np.sin(kd); dm=s.mcr-source
    mcr=source+dm*c+(s.jlr/k)*sn
    jlr=-k*dm*sn+s.jlr*c
    jlm=jlr.copy()
    mlm=(source-mcr)/(1.0+lam)
    return mcr,jlr,jlm,mlm


def _segments_above(zeta: float, alpha: float, M: float) -> tuple[ThreePhaseSegment, ...]:
    beta=zeta/3.0; gamma=2.0*zeta/3.0
    return (
        ThreePhaseSegment("1/3N",0.0,alpha,0.5,-0.5*M),
        ThreePhaseSegment("1/3P",alpha,beta,0.5,+0.5*M),
        ThreePhaseSegment("2/3P",beta,gamma,1.0,+1.0*M),
        ThreePhaseSegment("1/3P",gamma,zeta,0.5,+0.5*M),
    )


def _segments_below(zeta: float, epsilon: float, M: float) -> tuple[ThreePhaseSegment, ...]:
    beta=zeta/3.0; delta=2.0*zeta/3.0
    alpha=epsilon-2.0*zeta/3.0
    gamma=epsilon-zeta/3.0
    return (
        ThreePhaseSegment("1/3P",0.0,alpha,0.5,+0.5*M),
        ThreePhaseSegment("O-P",alpha,beta,0.5,+0.75*M),
        ThreePhaseSegment("2/3P",beta,gamma,1.0,+1.0*M),
        ThreePhaseSegment("O-P",gamma,delta,0.5,+0.75*M),
        ThreePhaseSegment("1/3P",delta,epsilon,0.5,+0.5*M),
        ThreePhaseSegment("O",epsilon,zeta,0.5,None),
    )


def _propagate(s0: _NormState, segments: Iterable[ThreePhaseSegment], lam: float) -> tuple[_NormState, list[tuple[ThreePhaseSegment,_NormState]]]:
    s=s0; history=[]
    for seg in segments:
        history.append((seg,s))
        dt=seg.theta_end-seg.theta_start
        if seg.clamp_norm is None:
            s=_open_end(s,dt,seg.source_norm,lam)
        else:
            s=_clamped_end(s,dt,seg.source_norm,seg.clamp_norm,lam)
    return s,history


# ---------------------------------------------------------------------------
# Reduced periodic solver
# ---------------------------------------------------------------------------
#
# Every topology interval in the normalized 3P model is affine in the state
# x=[mCr, jLr, jLm]^T.  For fixed switching half-period, conversion ratio and
# commutation instant the complete half-cycle therefore has the form
#
#     x(zeta) = A x(0) + b .
#
# Half-wave symmetry requires x(zeta)=-x(0), so the initial state is obtained
# from one 3x3 linear solve instead of being included in the nonlinear search:
#
#     (A + I) x(0) = -b .
#
# The nonlinear problem is then only two-dimensional:
#   regulated operating point : [zeta, commutation fraction]
#   fixed switching frequency : [M,    commutation fraction]
#
# This reduction is important.  The earlier alpha implementation solved five
# coupled unknowns with a generic least-squares search; it could return a small
# local compromise without satisfying the paper's periodic equations exactly.
# The reduced formulation enforces periodicity to numerical precision by
# construction and leaves only the physical event and power balance equations.


def _affine_clamped(dtheta: float, source: float, clamp: float, lam: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    c = math.cos(dtheta)
    sn = math.sin(dtheta)
    eq = source - clamp
    a = np.asarray(
        [
            [c, sn, 0.0],
            [-sn, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    b = np.asarray([eq * (1.0 - c), eq * sn, lam * clamp * dtheta], dtype=float)
    return a, b


def _affine_open(dtheta: float, source: float, lam: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    k = math.sqrt(lam / (1.0 + lam))
    kd = k * dtheta
    c = math.cos(kd)
    sn = math.sin(kd)
    row = np.asarray([-k * sn, c, 0.0], dtype=float)
    a = np.asarray(
        [
            [c, sn / k, 0.0],
            row,
            row,
        ],
        dtype=float,
    )
    b2 = k * source * sn
    b = np.asarray([source * (1.0 - c), b2, b2], dtype=float)
    return a, b


def _half_cycle_affine(segments: Iterable[ThreePhaseSegment], lam: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    a_total = np.eye(3, dtype=float)
    b_total = np.zeros(3, dtype=float)
    for seg in segments:
        dtheta = seg.theta_end - seg.theta_start
        if dtheta < -1e-13:
            raise ValueError(f"negative 3P state interval in {seg.label}")
        if seg.clamp_norm is None:
            a, b = _affine_open(max(dtheta, 0.0), seg.source_norm, lam)
        else:
            a, b = _affine_clamped(max(dtheta, 0.0), seg.source_norm, float(seg.clamp_norm), lam)
        b_total = a @ b_total + b
        a_total = a @ a_total
    return a_total, b_total


def _periodic_initial_state(segments: tuple[ThreePhaseSegment, ...], lam: float) -> tuple[_NormState, float]:
    a, b = _half_cycle_affine(segments, lam)
    mat = a + np.eye(3, dtype=float)
    cond = float(np.linalg.cond(mat))
    if not np.isfinite(cond) or cond > 1e12:
        raise np.linalg.LinAlgError(f"ill-conditioned 3P half-cycle periodic map (cond={cond:.3e})")
    x = np.linalg.solve(mat, -b)
    return _NormState(float(x[0]), float(x[1]), float(x[2])), cond


def _event_theta(mode: ThreePhaseMode, zeta: float, event_frac: float) -> float:
    # Keep a tiny margin away from collapsed states.  The limit at resonance is
    # still approached continuously, but exactly zero-duration subintervals
    # make the event equation rank deficient and are not useful numerically.
    u = float(np.clip(event_frac, 1e-5, 1.0 - 1e-5))
    if mode == ThreePhaseMode.ABOVE_RESONANCE:
        return u * (zeta / 3.0)
    return (2.0 / 3.0 + u / 3.0) * zeta


def _segments_for(mode: ThreePhaseMode, zeta: float, event_frac: float, M: float) -> tuple[ThreePhaseSegment, ...]:
    event = _event_theta(mode, zeta, event_frac)
    if mode == ThreePhaseMode.ABOVE_RESONANCE:
        return _segments_above(zeta, event, M)
    return _segments_below(zeta, event, M)


@dataclass(frozen=True)
class _ReducedPoint:
    s0: _NormState
    segments: tuple[ThreePhaseSegment, ...]
    event_theta: float
    event_residual: float
    power_norm: float
    periodic_error: float
    map_condition: float
    complementarity_margin: float


def _complementarity_margin(s0: _NormState, segments: tuple[ThreePhaseSegment, ...], lam: float) -> float:
    """Minimum normalized rectifier-current sign margin on clamped intervals.

    Positive values mean vLm and (jLr-jLm) have the same sign everywhere that
    the shared bridge clamps the representative transformer phase.  O intervals
    are checked separately by the analytical O state, where jLr == jLm.
    """
    _, history = _propagate(s0, segments, lam)
    margin = math.inf
    for seg, state in history:
        if seg.theta_end <= seg.theta_start:
            continue
        tau = np.linspace(0.0, seg.theta_end - seg.theta_start, 13)
        if seg.clamp_norm is None:
            _, jr, jm, _ = _sample_open(state, tau, seg.source_norm, lam)
            margin = min(margin, -float(np.max(np.abs(jr - jm))))
            continue
        _, jr, jm, _ = _sample_clamped(state, tau, seg.source_norm, float(seg.clamp_norm), lam)
        signed = np.sign(float(seg.clamp_norm)) * (jr - jm)
        margin = min(margin, float(np.min(signed)))
    if math.isinf(margin):
        return 0.0
    return margin


def _reduced_point(mode: ThreePhaseMode, zeta: float, event_frac: float, M: float, lam: float) -> _ReducedPoint:
    segments = _segments_for(mode, zeta, event_frac, M)
    s0, cond = _periodic_initial_state(segments, lam)
    send, hist = _propagate(s0, segments, lam)
    event = _event_theta(mode, zeta, event_frac)
    se = _state_at(hist, event, lam)
    event_residual = float(se.jlr - se.jlm)
    pnorm = _normalized_power(s0, segments, lam, zeta)
    periodic = max(abs(send.mcr + s0.mcr), abs(send.jlr + s0.jlr), abs(send.jlm + s0.jlm))
    cmargin = _complementarity_margin(s0, segments, lam)
    return _ReducedPoint(s0, segments, event, event_residual, pnorm, float(periodic), cond, cmargin)


def _state_at(history: list[tuple[ThreePhaseSegment,_NormState]], theta: float, lam: float) -> _NormState:
    for seg,s in history:
        if seg.theta_start-1e-13 <= theta <= seg.theta_end+1e-13:
            dt=max(0.0,min(theta,seg.theta_end)-seg.theta_start)
            if seg.clamp_norm is None:
                return _open_end(s,dt,seg.source_norm,lam)
            return _clamped_end(s,dt,seg.source_norm,seg.clamp_norm,lam)
    raise ValueError("theta outside segment history")


# Fixed Gauss-Legendre quadrature nodes.  Piecewise waveforms are smooth inside
# each interval, so this is both fast and much more stable than integrating a
# sampled absolute-value waveform inside the nonlinear solve.
_GX,_GW=np.polynomial.legendre.leggauss(16)


def _normalized_power(s0: _NormState, segments: tuple[ThreePhaseSegment,...], lam: float, zeta: float) -> float:
    """Total 3-phase transferred power normalized by Vb^2/Z0."""
    _,history=_propagate(s0,segments,lam)
    integ=0.0
    for seg,s in history:
        if seg.clamp_norm is None or seg.theta_end <= seg.theta_start:
            continue
        a,b=seg.theta_start,seg.theta_end
        tau=0.5*(b-a)*(_GX+1.0)
        _,jr,jm,_=_sample_clamped(s,tau,seg.source_norm,float(seg.clamp_norm),lam)
        load=np.abs(jr-jm)
        integ += 0.5*(b-a)*float(np.sum(_GW * (abs(float(seg.clamp_norm))*load)))
    # All three phases are identical time shifts.
    return 3.0*integ/max(zeta,1e-30)


def _residual_regulated(x: NDArray[np.float64], *, mode: ThreePhaseMode, zeta_bounds: tuple[float,float], M: float, pnorm_target: float, lam: float) -> NDArray[np.float64]:
    m0,jr0,jm0,zeta,event_frac=x
    # event_frac is a bounded 0..1 coordinate, converted to the mode-specific
    # physical event interval.  This avoids singular bounds around zeta.
    if mode==ThreePhaseMode.ABOVE_RESONANCE:
        alpha=(0.015+0.970*event_frac)*(zeta/3.0)
        segs=_segments_above(zeta,alpha,M); event_theta=alpha
    else:
        epsilon=(2.0/3.0 + (0.015+0.970*event_frac)/3.0)*zeta
        segs=_segments_below(zeta,epsilon,M); event_theta=epsilon
    s0=_NormState(m0,jr0,jm0); send,hist=_propagate(s0,segs,lam); se=_state_at(hist,event_theta,lam)
    pnorm=_normalized_power(s0,segs,lam,zeta)
    return np.asarray([
        send.mcr+m0,
        send.jlr+jr0,
        send.jlm+jm0,
        se.jlr-se.jlm,
        (pnorm-pnorm_target)/max(abs(pnorm_target),0.05),
    ],dtype=float)


def _residual_frequency(x: NDArray[np.float64], *, mode: ThreePhaseMode, zeta: float, load_r_ohm: float, vbus_v: float, turns_ratio: float, rectifier_drop_v: float, zo: float, lam: float) -> NDArray[np.float64]:
    m0,jr0,jm0,M,event_frac=x
    if mode==ThreePhaseMode.ABOVE_RESONANCE:
        alpha=(0.015+0.970*event_frac)*(zeta/3.0); segs=_segments_above(zeta,alpha,M); event_theta=alpha
    else:
        epsilon=(2.0/3.0+(0.015+0.970*event_frac)/3.0)*zeta; segs=_segments_below(zeta,epsilon,M); event_theta=epsilon
    s0=_NormState(m0,jr0,jm0); send,hist=_propagate(s0,segs,lam); se=_state_at(hist,event_theta,lam)
    vrect=M*vbus_v/max(turns_ratio,1e-30)
    vo=max(vrect-rectifier_drop_v,1e-9)
    pout=vo*vo/max(load_r_ohm,1e-30)
    ptransfer=pout*vrect/max(vo,1e-30)
    vb=(2.0/3.0)*vbus_v
    pnorm_target=ptransfer*zo/(vb*vb)
    pnorm=_normalized_power(s0,segs,lam,zeta)
    return np.asarray([
        send.mcr+m0,
        send.jlr+jr0,
        send.jlm+jm0,
        se.jlr-se.jlm,
        (pnorm-pnorm_target)/max(abs(pnorm_target),0.05),
    ],dtype=float)


def _multi_seed_solve(fun, lower: NDArray[np.float64], upper: NDArray[np.float64], seeds: list[NDArray[np.float64]], max_nfev: int=3500):
    best=None
    for seed in seeds:
        seed=np.minimum(np.maximum(np.asarray(seed,dtype=float),lower+1e-9),upper-1e-9)
        sol=least_squares(fun,seed,bounds=(lower,upper),xtol=2e-12,ftol=2e-12,gtol=2e-12,max_nfev=max_nfev,x_scale="jac")
        rn=float(np.linalg.norm(fun(sol.x),ord=np.inf))
        item=(rn,sol)
        if best is None or rn<best[0]: best=item
        if sol.success and rn<2e-6: break
    assert best is not None
    return best


def _default_state_seeds(M: float, zeta: float, mode: ThreePhaseMode) -> list[NDArray[np.float64]]:
    # State magnitudes are O(1) in the normalized system.  Several phase-shifted
    # seeds make the solve robust without embedding a single-phase solution.
    if mode==ThreePhaseMode.ABOVE_RESONANCE:
        ef=0.55
    else:
        ef=0.55
    return [
        np.asarray([0.0,-0.7,-0.45,zeta,ef]),
        np.asarray([0.2,-1.0,-0.55,zeta,0.35]),
        np.asarray([-0.2,-0.5,-0.35,zeta,0.72]),
        np.asarray([0.4,-1.3,-0.8,zeta,0.50]),
    ]


@dataclass(frozen=True)
class _ReducedSolve:
    success: bool
    residual_norm: float
    zeta: float
    M: float
    event_frac: float
    point: _ReducedPoint


def _solve_reduced_multiseed(fun, *, lower: NDArray[np.float64], upper: NDArray[np.float64], seeds: list[NDArray[np.float64]], decode, max_nfev: int = 1200) -> _ReducedSolve:
    best: _ReducedSolve | None = None
    for seed in seeds:
        x0 = np.minimum(np.maximum(np.asarray(seed, dtype=float), lower + 1e-8), upper - 1e-8)
        try:
            sol = least_squares(
                fun,
                x0,
                bounds=(lower, upper),
                xtol=5e-13,
                ftol=5e-13,
                gtol=5e-13,
                max_nfev=max_nfev,
                x_scale="jac",
            )
            zeta, M, event_frac, point = decode(sol.x)
            r = np.asarray(fun(sol.x), dtype=float)
            rn = float(np.linalg.norm(r, ord=np.inf))
            # A negative complementarity margin is a physically invalid diode
            # state even if the two algebraic equations happen to be small.
            physical = point.complementarity_margin >= -2e-5
            item = _ReducedSolve(bool(sol.success and physical), rn, zeta, M, event_frac, point)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
        if best is None or (item.success, -item.residual_norm) > (best.success, -best.residual_norm):
            best = item
        if item.success and item.residual_norm < 2e-8:
            break
    if best is None:
        raise RuntimeError("3P reduced periodic solve failed for all initial seeds")
    return best


def _regulated_mode_solve(spec: LLCDesignSpec,tank: TankDesign,*,vbus_v:float,output_power_w:float,mode:ThreePhaseMode,seed:NDArray[np.float64]|None=None):
    n=spec.turns_ratio
    vrect=spec.vout_v+spec.rectifier_equivalent_drop_v
    M=n*vrect/vbus_v
    vb=(2.0/3.0)*vbus_v
    ptransfer=output_power_w*vrect/spec.vout_v
    pnorm_target=ptransfer*tank.zr_ohm/(vb*vb)
    lam=tank.lr_h/tank.lm_h
    # zeta=pi*fr/fs. Above resonance fs>fr -> zeta<pi.
    if mode==ThreePhaseMode.ABOVE_RESONANCE:
        zlo=max(math.pi*tank.fr_hz/spec.maximum_frequency_hz,0.15*math.pi)
        zhi=min(math.pi*tank.fr_hz/max(spec.minimum_frequency_hz,tank.fr_hz*(1+1e-6)),0.999999*math.pi)
    else:
        zlo=max(math.pi*tank.fr_hz/min(spec.maximum_frequency_hz,tank.fr_hz*(1-1e-6)),1.000001*math.pi)
        zhi=min(math.pi*tank.fr_hz/spec.minimum_frequency_hz,2.5*math.pi)
    if not (zlo<zhi):
        return None

    def decode(x: NDArray[np.float64]):
        zeta=float(x[0]); ef=float(x[1]); point=_reduced_point(mode,zeta,ef,M,lam)
        return zeta,M,ef,point

    def fun(x: NDArray[np.float64]) -> NDArray[np.float64]:
        try:
            _,_,_,p=decode(x)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            return np.asarray([1e3,1e3,1e3],dtype=float)
        event_scale=max(0.25,math.sqrt(max(abs(pnorm_target),1e-6)))
        power_scale=max(abs(pnorm_target),0.05)
        invalid=min(0.0,p.complementarity_margin)
        return np.asarray([
            p.event_residual/event_scale,
            (p.power_norm-pnorm_target)/power_scale,
            4.0*invalid,
        ],dtype=float)

    lower=np.asarray([zlo,1e-5],dtype=float); upper=np.asarray([zhi,1.0-1e-5],dtype=float)
    zgrid=np.linspace(zlo+0.04*(zhi-zlo),zhi-0.04*(zhi-zlo),7)
    egrid=(0.08,0.22,0.38,0.55,0.72,0.88)
    seeds=[np.asarray([z,e],dtype=float) for z in zgrid for e in egrid]
    if seed is not None and len(seed)>=2:
        # Accept either the new [zeta,event] continuation seed or the old 5-D
        # alpha seed for compatibility with callers developed during V8.3.
        if len(seed)>=5: seeds.insert(0,np.asarray([float(seed[3]),float(seed[4])]))
        else: seeds.insert(0,np.asarray(seed[:2],dtype=float))
    return _solve_reduced_multiseed(fun,lower=lower,upper=upper,seeds=seeds,decode=decode)


def _at_frequency_mode_solve(spec: LLCDesignSpec,tank:TankDesign,*,vbus_v:float,frequency_hz:float,load_r_ohm:float,mode:ThreePhaseMode,seed:NDArray[np.float64]|None=None):
    zeta=math.pi*tank.fr_hz/frequency_hz
    if mode==ThreePhaseMode.ABOVE_RESONANCE and zeta>=math.pi*(1+2e-5): return None
    if mode==ThreePhaseMode.BELOW_RESONANCE and zeta<=math.pi*(1-2e-5): return None
    lam=tank.lr_h/tank.lm_h
    M0=max(0.08,min(2.5,spec.turns_ratio*(spec.vout_v+spec.rectifier_equivalent_drop_v)/vbus_v))

    def pnorm_target_for(M: float) -> float:
        vrect=M*vbus_v/max(spec.turns_ratio,1e-30)
        vo=max(vrect-spec.rectifier_equivalent_drop_v,1e-9)
        pout=vo*vo/max(load_r_ohm,1e-30)
        ptransfer=pout*vrect/max(vo,1e-30)
        vb=(2.0/3.0)*vbus_v
        return ptransfer*tank.zr_ohm/(vb*vb)

    def decode(x: NDArray[np.float64]):
        M=float(x[0]); ef=float(x[1]); point=_reduced_point(mode,zeta,ef,M,lam)
        return zeta,M,ef,point

    def fun(x: NDArray[np.float64]) -> NDArray[np.float64]:
        M=float(x[0])
        try:
            _,_,_,p=decode(x)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            return np.asarray([1e3,1e3,1e3],dtype=float)
        target=pnorm_target_for(M)
        event_scale=max(0.25,math.sqrt(max(abs(target),1e-6)))
        power_scale=max(abs(target),0.05)
        invalid=min(0.0,p.complementarity_margin)
        return np.asarray([
            p.event_residual/event_scale,
            (p.power_norm-target)/power_scale,
            4.0*invalid,
        ],dtype=float)

    lower=np.asarray([0.02,1e-5],dtype=float); upper=np.asarray([3.5,1.0-1e-5],dtype=float)
    mgrid=np.unique(np.clip(np.asarray([0.35*M0,0.55*M0,0.8*M0,M0,1.2*M0,1.55*M0,2.0*M0,0.65,1.0,1.35,1.8]),0.025,3.45))
    egrid=(0.08,0.22,0.40,0.58,0.76,0.90)
    seeds=[np.asarray([m,e],dtype=float) for m in mgrid for e in egrid]
    if seed is not None and len(seed)>=2:
        if len(seed)>=5: seeds.insert(0,np.asarray([float(seed[3]),float(seed[4])]))
        else: seeds.insert(0,np.asarray(seed[:2],dtype=float))
    return _solve_reduced_multiseed(fun,lower=lower,upper=upper,seeds=seeds,decode=decode)


def _build_result(spec:LLCDesignSpec,tank:TankDesign,*,vbus_v:float,output_power_w:float,mode:ThreePhaseMode,zeta:float,M:float,event_frac:float,s0:_NormState,residual_norm:float,samples_per_cycle:int) -> ThreePhaseTDResult:
    event=_event_theta(mode,zeta,event_frac)
    if mode==ThreePhaseMode.ABOVE_RESONANCE:
        segs=_segments_above(zeta,event,M); ename="alpha"
    else:
        segs=_segments_below(zeta,event,M); ename="epsilon"
    fs=math.pi*tank.fr_hz/zeta
    n_total=max(258,int(samples_per_cycle)); n_total += (-n_total) % 6
    n_half=n_total//2; theta_h=np.linspace(0.0,zeta,n_half,endpoint=False)
    _,hist=_propagate(s0,segs,tank.lr_h/tank.lm_h)
    mcr=np.zeros(n_half); jr=np.zeros(n_half); jm=np.zeros(n_half); mlm=np.zeros(n_half); src=np.zeros(n_half)
    for k,th in enumerate(theta_h):
        for seg,ss in hist:
            if seg.theta_start-1e-12<=th<seg.theta_end+1e-12:
                tau=np.asarray([th-seg.theta_start])
                if seg.clamp_norm is None:
                    a,b,c,d=_sample_open(ss,tau,seg.source_norm,tank.lr_h/tank.lm_h)
                else:
                    a,b,c,d=_sample_clamped(ss,tau,seg.source_norm,float(seg.clamp_norm),tank.lr_h/tank.lm_h)
                mcr[k]=a[0];jr[k]=b[0];jm[k]=c[0];mlm[k]=d[0];src[k]=seg.source_norm;break
    # Half-wave antisymmetry creates the second half-cycle exactly.
    theta=np.r_[theta_h,theta_h+zeta]
    mcr=np.r_[mcr,-mcr];jr=np.r_[jr,-jr];jm=np.r_[jm,-jm];mlm=np.r_[mlm,-mlm];src=np.r_[src,-src]
    vb=(2.0/3.0)*vbus_v; ib=vb/tank.zr_ohm; vrect=M*vbus_v/spec.turns_ratio; vo=max(vrect-spec.rectifier_equivalent_drop_v,0.0)
    load=jr-jm
    ir=jr*ib; im=jm*ib; iload=load*ib; isec=iload*spec.turns_ratio
    vcr=mcr*vb; vlm=mlm*vb; vphase=src*vb
    # Reconstruct the common six-pulse bridge current from three 120°-shifted
    # secondary phase currents. For a balanced three-wire set ia+ib+ic=0 and
    # the DC-link current equals the sum of the positive phase currents, i.e.
    # 0.5*(|ia|+|ib|+|ic|).  Do not normalize this waveform to the requested
    # load: any mismatch is a model-consistency diagnostic and must stay visible.
    n=len(isec); shift=n//3
    ib_phase=np.roll(isec,shift); ic_phase=np.roll(isec,2*shift)
    iout=0.5*(np.abs(isec)+np.abs(ib_phase)+np.abs(ic_phase))
    phase_kcl_peak=float(np.max(np.abs(isec+ib_phase+ic_phase)))
    time=theta/(2.0*math.pi*tank.fr_hz)
    ptransfer=output_power_w*vrect/max(vo,1e-30) if vo>0 else output_power_w
    warnings=[]
    if residual_norm>2e-4: warnings.append(f"3P TD residual {residual_norm:.3e} exceeds preferred 2e-4 tolerance.")
    expected_iout=output_power_w/max(vo,1e-30) if vo>0.0 else 0.0
    iout_mean=float(np.mean(iout))
    if expected_iout>1e-9:
        iout_err=(iout_mean-expected_iout)/expected_iout
        if abs(iout_err)>0.02:
            warnings.append(f"Shared-bridge reconstructed DC current differs from the solved power point by {100*iout_err:.2f}%; inspect 3P phase-current/KCL consistency.")
    if phase_kcl_peak>max(0.02*max(float(np.max(np.abs(isec))),1e-9),0.05):
        warnings.append(f"120-degree phase-current reconstruction has {phase_kcl_peak:.3f} A peak KCL residual; waveform is retained without artificial rescaling.")
    if mode==ThreePhaseMode.BELOW_RESONANCE:
        warnings.append("Below-resonance result includes the topology-specific O-P interaction and O discontinuous-conduction interval.")
    # Near fr one event interval collapses and the preferred piecewise model is
    # evaluated as a limiting sequence; a few 1e-3 normalized residual is then
    # expected even though voltage/power regulation remains accurate.
    return ThreePhaseTDResult(residual_norm<5e-3,mode,fs,tank.fr_hz,zeta,M,vo/vbus_v,vo,output_power_w,ptransfer,residual_norm,event,ename,(s0.mcr,s0.jlr,s0.jlm),time,theta,mcr,jr,jm,mlm,src,load,ir,im,iload,isec,vcr,vlm,vphase,iout,segs,tuple(warnings))


def solve_three_phase_td_regulated(spec:LLCDesignSpec,tank:TankDesign,*,vbus_v:float|None=None,load_fraction:float=1.0,samples_per_cycle:int=1024) -> ThreePhaseTDResult:
    """Solve the 3P 120° operating point at the requested Vo and load.

    Both preferred above- and below-resonance branches are attempted when the
    configured frequency range permits them.  The converged branch with the
    lowest normalized residual is returned; a branch inconsistent with its
    frequency region is never silently accepted.
    """
    vin=float(spec.vbus_nom_v if vbus_v is None else vbus_v); pout=spec.pout_w*float(load_fraction)
    candidates=[]
    for mode in (ThreePhaseMode.ABOVE_RESONANCE,ThreePhaseMode.BELOW_RESONANCE):
        item=_regulated_mode_solve(spec,tank,vbus_v=vin,output_power_w=pout,mode=mode)
        if item is None: continue
        zeta=float(item.zeta); fs=math.pi*tank.fr_hz/zeta
        if spec.minimum_frequency_hz<=fs<=spec.maximum_frequency_hz:
            candidates.append((item.residual_norm,not item.success,mode,item))
    if not candidates:
        raise RuntimeError("No admissible 3P TD branch inside the configured frequency range")
    rn,_,mode,item=min(candidates,key=lambda z:(z[1],z[0]))
    # Include the exact periodic-map error in the reported residual. It should
    # normally be near machine precision in the reduced formulation.
    report=max(float(rn),float(item.point.periodic_error),max(0.0,-float(item.point.complementarity_margin)))
    return _build_result(spec,tank,vbus_v=vin,output_power_w=pout,mode=mode,zeta=item.zeta,M=item.M,event_frac=item.event_frac,s0=item.point.s0,residual_norm=report,samples_per_cycle=samples_per_cycle)


def solve_three_phase_td_at_frequency(spec:LLCDesignSpec,tank:TankDesign,frequency_hz:float,*,vbus_v:float|None=None,load_resistance_ohm:float|None=None,samples_per_cycle:int=1024) -> ThreePhaseTDResult:
    """Solve the topology-specific DC gain at one switching frequency.

    The load is a physical DC resistance.  If omitted it is the rated system
    load ``Vo^2/Pout``.  This is the primitive used for a true 3P gain curve.
    """
    vin=float(spec.vbus_nom_v if vbus_v is None else vbus_v)
    rload=float(spec.vout_v**2/spec.pout_w if load_resistance_ohm is None else load_resistance_ohm)
    mode=ThreePhaseMode.ABOVE_RESONANCE if frequency_hz>=tank.fr_hz else ThreePhaseMode.BELOW_RESONANCE
    item=_at_frequency_mode_solve(spec,tank,vbus_v=vin,frequency_hz=float(frequency_hz),load_r_ohm=rload,mode=mode)
    if item is None: raise RuntimeError("Requested frequency is inconsistent with selected 3P mode")
    M=float(item.M); vrect=M*vin/spec.turns_ratio; vo=max(vrect-spec.rectifier_equivalent_drop_v,0.0); pout=vo*vo/rload
    report=max(float(item.residual_norm),float(item.point.periodic_error),max(0.0,-float(item.point.complementarity_margin)))
    return _build_result(spec,tank,vbus_v=vin,output_power_w=pout,mode=mode,zeta=item.zeta,M=M,event_frac=item.event_frac,s0=item.point.s0,residual_norm=report,samples_per_cycle=samples_per_cycle)


def solve_three_phase_td_gain_curve(spec:LLCDesignSpec,tank:TankDesign,*,vbus_v:float|None=None,load_fraction:float=1.0,frequencies_hz:NDArray[np.float64]|None=None,points:int=80) -> tuple[NDArray[np.float64],NDArray[np.float64],NDArray[np.float64],NDArray[np.float64]]:
    """Return (frequency, M=n*Vrect/Vin, Vo/Vin, Vo) using the true 3P TD model."""
    vin=float(spec.vbus_nom_v if vbus_v is None else vbus_v)
    rload=spec.vout_v**2/max(spec.pout_w*float(load_fraction),1e-30)
    freqs=np.geomspace(spec.minimum_frequency_hz,spec.maximum_frequency_hz,int(points)) if frequencies_hz is None else np.asarray(frequencies_hz,dtype=float)
    M=np.full_like(freqs,np.nan); dc=np.full_like(freqs,np.nan); vo=np.full_like(freqs,np.nan)
    for i,f in enumerate(freqs):
        try:
            r=solve_three_phase_td_at_frequency(spec,tank,float(f),vbus_v=vin,load_resistance_ohm=rload,samples_per_cycle=256)
        except Exception:
            continue
        if r.converged:
            M[i]=r.normalized_gain;dc[i]=r.dc_gain_vo_vin;vo[i]=r.output_voltage_v
    return freqs,M,dc,vo


__all__=[
    "ThreePhaseMode","ThreePhaseSegment","ThreePhaseTDResult",
    "solve_three_phase_td_regulated","solve_three_phase_td_at_frequency","solve_three_phase_td_gain_curve",
]
