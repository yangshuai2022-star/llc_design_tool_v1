"""Fourier-collocation harmonic-balance model for the fixed 120° 3-phase LLC.

The model balances the same topology-specific hybrid equations used by
``star_time_domain`` in a truncated odd-harmonic Fourier basis.  It is not a
single-phase HB waveform copied three times.  The 1/3P/N, 2/3P/N, O-P/O-N and
O state laws are enforced directly at collocation points, while the event
boundary and DC conversion gain are solved together with the Fourier states.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares, brentq

from ..core.spec import LLCDesignSpec
from ..core.tank import TankDesign
from .star_time_domain import (
    ThreePhaseMode, ThreePhaseSegment, ThreePhaseTDResult,
    solve_three_phase_td_at_frequency, solve_three_phase_td_regulated,
    _segments_above, _segments_below,
)


@dataclass(frozen=True)
class ThreePhaseHBResult:
    converged: bool
    mode: ThreePhaseMode
    switching_frequency_hz: float
    resonant_frequency_hz: float
    normalized_gain: float
    dc_gain_vo_vin: float
    output_voltage_v: float
    output_power_w: float
    residual_norm: float
    collocation_rms: float
    collocation_peak: float
    constraint_peak: float
    max_harmonic: int
    harmonics: tuple[int, ...]
    event_theta: float
    time_s: NDArray[np.float64]
    theta: NDArray[np.float64]
    mcr: NDArray[np.float64]
    jlr: NDArray[np.float64]
    jlm: NDArray[np.float64]
    mlm: NDArray[np.float64]
    source_norm: NDArray[np.float64]
    i_resonant_a: NDArray[np.float64]
    i_magnetizing_a: NDArray[np.float64]
    i_secondary_phase_a: NDArray[np.float64]
    v_cr_v: NDArray[np.float64]
    v_lm_v: NDArray[np.float64]
    i_output_rectified_a: NDArray[np.float64]
    segments: tuple[ThreePhaseSegment, ...]
    warnings: tuple[str, ...] = ()

    @property
    def resonant_current_rms_a(self) -> float:
        return float(np.sqrt(np.mean(self.i_resonant_a**2)))


@dataclass(frozen=True)
class ThreePhaseHBConfig:
    # H=11 is a practical interactive compromise for the discontinuous
    # below-resonance current waveform. Higher odd orders (15/21) are exposed
    # for verification sweeps.
    max_harmonic: int = 11
    collocation_points: int = 96
    samples_per_cycle: int = 1026
    max_nfev: int = 1000
    collocation_rms_tolerance: float = 8.0e-2
    constraint_tolerance: float = 2.0e-2
    constraint_weight: float = 8.0


def _harmonics(max_harmonic:int)->NDArray[np.int64]:
    m=max(1,int(max_harmonic))
    if m%2==0: m-=1
    return np.arange(1,m+1,2,dtype=int)


def _basis(theta:NDArray[np.float64],zeta:float,h:NDArray[np.int64]):
    om=math.pi/zeta
    arg=np.outer(theta, h.astype(float)*om)
    return np.cos(arg),np.sin(arg),-np.sin(arg)*(h*om),np.cos(arg)*(h*om)


def _eval(coeff:NDArray[np.float64],theta:NDArray[np.float64],zeta:float,h:NDArray[np.int64]):
    k=len(h); ac=coeff[:k]; bs=coeff[k:]
    c,s,dc,ds=_basis(theta,zeta,h)
    value=c@ac+s@bs
    deriv=dc@ac+ds@bs
    return value,deriv


def _fit_coeff(theta:NDArray[np.float64],values:NDArray[np.float64],zeta:float,h:NDArray[np.int64]):
    c,s,_,_=_basis(theta,zeta,h); A=np.c_[c,s]
    return np.linalg.lstsq(A,values,rcond=None)[0]


def _segment_arrays(theta:NDArray[np.float64],segments:tuple[ThreePhaseSegment,...]):
    src=np.zeros_like(theta); clamp=np.full_like(theta,np.nan); labels=np.empty(theta.size,dtype=object)
    for seg in segments:
        mask=(theta>=seg.theta_start-1e-14)&(theta<seg.theta_end-1e-14)
        src[mask]=seg.source_norm; labels[mask]=seg.label
        if seg.clamp_norm is not None: clamp[mask]=seg.clamp_norm
    # Last point guard (normally endpoint=False).
    unset=np.equal(labels,None)  # noqa: E711
    if np.any(unset):
        seg=segments[-1];src[unset]=seg.source_norm;labels[unset]=seg.label
        if seg.clamp_norm is not None:clamp[unset]=seg.clamp_norm
    return src,clamp,labels


def _event_from_frac(mode:ThreePhaseMode,zeta:float,event_frac:float,M:float):
    if mode==ThreePhaseMode.ABOVE_RESONANCE:
        ev=(0.015+0.970*event_frac)*(zeta/3.0);segs=_segments_above(zeta,ev,M)
    else:
        ev=(2.0/3.0+(0.015+0.970*event_frac)/3.0)*zeta;segs=_segments_below(zeta,ev,M)
    return ev,segs


def _power_norm(theta:NDArray[np.float64],jr:NDArray[np.float64],jm:NDArray[np.float64],clamp:NDArray[np.float64],zeta:float):
    active=np.isfinite(clamp); integrand=np.zeros_like(theta); integrand[active]=np.abs(clamp[active])*np.abs(jr[active]-jm[active])
    return 3.0*float(np.trapezoid(integrand,theta))/max(zeta,1e-30)


def _td_seed(spec:LLCDesignSpec,tank:TankDesign,frequency_hz:float,vbus_v:float,load_r_ohm:float,h:NDArray[np.int64]):
    td=solve_three_phase_td_at_frequency(spec,tank,frequency_hz,vbus_v=vbus_v,load_resistance_ohm=load_r_ohm,samples_per_cycle=max(1024,8*len(h)*32))
    zeta=math.pi*tank.fr_hz/frequency_hz
    # TD result contains exactly one full period, half-wave antisymmetric.
    cm=_fit_coeff(td.theta,td.mcr,zeta,h); cr=_fit_coeff(td.theta,td.jlr,zeta,h); ci=_fit_coeff(td.theta,td.jlm,zeta,h)
    if td.mode==ThreePhaseMode.ABOVE_RESONANCE:
        ef=(td.event_theta/(zeta/3.0)-0.015)/0.970
    else:
        ef=((td.event_theta/zeta-2.0/3.0)*3.0-0.015)/0.970
    ef=float(np.clip(ef,0.0,1.0))
    return td,np.r_[cm,cr,ci,td.normalized_gain,ef]


def solve_three_phase_hb_at_frequency(spec:LLCDesignSpec,tank:TankDesign,frequency_hz:float,*,vbus_v:float|None=None,load_resistance_ohm:float|None=None,config:ThreePhaseHBConfig|None=None) -> ThreePhaseHBResult:
    cfg=config or ThreePhaseHBConfig(); vin=float(spec.vbus_nom_v if vbus_v is None else vbus_v); rload=float(spec.vout_v**2/spec.pout_w if load_resistance_ohm is None else load_resistance_ohm)
    zeta=math.pi*tank.fr_hz/float(frequency_hz); mode=ThreePhaseMode.ABOVE_RESONANCE if frequency_hz>=tank.fr_hz else ThreePhaseMode.BELOW_RESONANCE
    h=_harmonics(cfg.max_harmonic); K=len(h); lam=tank.lr_h/tank.lm_h
    td,x0=_td_seed(spec,tank,float(frequency_hz),vin,rload,h)
    # Collocation points on the positive half-cycle.  Remove points close to
    # switching/event boundaries on each residual evaluation to avoid asking a
    # truncated Fourier series to reproduce a discontinuous derivative.
    theta_base=(np.arange(cfg.collocation_points,dtype=float)+0.5)*zeta/cfg.collocation_points

    def residual(x:NDArray[np.float64]):
        cm=x[0:2*K]; cr=x[2*K:4*K]; ci=x[4*K:6*K]; M=float(x[-2]); ef=float(x[-1])
        ev,segs=_event_from_frac(mode,zeta,ef,M)
        bounds=np.asarray([s.theta_start for s in segs]+[segs[-1].theta_end])
        dist=np.min(np.abs(theta_base[:,None]-bounds[None,:]),axis=1)
        # least_squares requires a residual vector with a fixed dimension.
        # Keep every collocation point and smoothly de-emphasize points nearest
        # a hybrid-state boundary, where a finite Fourier basis cannot reproduce
        # the discontinuous state derivative exactly (Gibbs phenomenon).
        th=theta_base
        cell=zeta/cfg.collocation_points
        weight=np.clip(dist/max(0.60*cell,1e-30),0.15,1.0)
        m,dm=_eval(cm,th,zeta,h);jr,djr=_eval(cr,th,zeta,h);jm,djm=_eval(ci,th,zeta,h)
        src,clamp,_=_segment_arrays(th,segs); out=[]
        openmask=~np.isfinite(clamp); activemask=~openmask
        # Weight each differential residual to comparable O(1) scales.
        out.extend((weight*(dm-jr)).tolist())
        r2=np.empty_like(th);r3=np.empty_like(th)
        r2[activemask]=djr[activemask]-(src[activemask]-m[activemask]-clamp[activemask])
        r3[activemask]=djm[activemask]-lam*clamp[activemask]
        if np.any(openmask):
            k2=lam/(1.0+lam)
            r2[openmask]=djr[openmask]-k2*(src[openmask]-m[openmask])
            r3[openmask]=jm[openmask]-jr[openmask]
        out.extend((weight*r2).tolist());out.extend((weight*r3).tolist())
        # Event complementarity.
        je,_=_eval(cr,np.asarray([ev]),zeta,h);me,_=_eval(ci,np.asarray([ev]),zeta,h);out.append(float(je[0]-me[0]))
        # Physical load power constraint for the DC load resistance.
        dense=np.linspace(0.0,zeta,max(512,8*cfg.collocation_points),endpoint=False)
        jr_d,_=_eval(cr,dense,zeta,h);jm_d,_=_eval(ci,dense,zeta,h);_,cl_d,_=_segment_arrays(dense,segs)
        pnorm=_power_norm(dense,jr_d,jm_d,cl_d,zeta)
        vrect=M*vin/spec.turns_ratio;vo=max(vrect-spec.rectifier_equivalent_drop_v,1e-9);pout=vo*vo/rload;ptransfer=pout*vrect/vo;vb=(2.0/3.0)*vin;pt=ptransfer*tank.zr_ohm/(vb*vb)
        out.append(cfg.constraint_weight*(pnorm-pt)/max(abs(pt),0.05))
        return np.asarray(out,dtype=float)

    lower=np.r_[np.full(6*K,-8.0),0.03,0.0];upper=np.r_[np.full(6*K,8.0),3.5,1.0]
    sol=least_squares(residual,np.minimum(np.maximum(x0,lower+1e-8),upper-1e-8),bounds=(lower,upper),xtol=3e-11,ftol=3e-11,gtol=3e-11,max_nfev=cfg.max_nfev,x_scale="jac")
    rvec=residual(sol.x)
    differential=rvec[:-2]
    constraints=rvec[-2:]/max(cfg.constraint_weight,1e-30)
    coll_rms=float(np.sqrt(np.mean(differential*differential))) if differential.size else 0.0
    coll_peak=float(np.max(np.abs(differential))) if differential.size else 0.0
    constraint_peak=float(np.max(np.abs(constraints))) if constraints.size else 0.0
    # The pointwise peak is dominated by Gibbs error at hybrid-state
    # derivative jumps. Report it separately; convergence is based on the RMS
    # differential balance plus the physical event/power constraints.
    rn=max(coll_rms,constraint_peak)
    M=float(sol.x[-2]);ef=float(sol.x[-1]);ev,segs=_event_from_frac(mode,zeta,ef,M)
    # Full-period reconstructed waveforms. Use a multiple of six so an exact
    # 120 degree circular shift exists in the sampled representation.
    n=max(258,int(cfg.samples_per_cycle)); n += (-n) % 6
    theta=np.arange(n,dtype=float)*(2.0*zeta/n)
    cm=sol.x[0:2*K];cr=sol.x[2*K:4*K];ci=sol.x[4*K:6*K]
    m,_=_eval(cm,theta,zeta,h);jr,_=_eval(cr,theta,zeta,h);jm,_=_eval(ci,theta,zeta,h)
    # State schedule for second half is the antisymmetric image of first half.
    th_half=np.mod(theta,zeta); src_h,cl_h,_=_segment_arrays(th_half,segs);sgn=np.where(theta<zeta,1.0,-1.0);src=src_h*sgn;clamp=cl_h*sgn
    openmask=~np.isfinite(clamp);mlm=clamp.copy();mlm[openmask]=(src[openmask]-m[openmask])/(1.0+lam)
    vb=(2.0/3.0)*vin;ib=vb/tank.zr_ohm;vrect=M*vin/spec.turns_ratio;vo=max(vrect-spec.rectifier_equivalent_drop_v,0.0);pout=vo*vo/rload
    ir=jr*ib;im=jm*ib;isec=(jr-jm)*ib*spec.turns_ratio;vcr=m*vb;vlm=mlm*vb
    shift=n//3;ab=np.abs(isec);iout_raw=ab+np.roll(ab,shift)+np.roll(ab,2*shift)
    # The state model gives the rectified-current waveform shape. Normalize its
    # mean to the physical DC load current; this is the same power constraint
    # used by the hybrid model and avoids assuming a two-diode 0.5 factor when
    # O-P/O intervals alter the conduction pattern.
    target_iout=pout/max(vo,1e-30) if vo>0.0 else 0.0
    iout=iout_raw*(target_iout/max(float(np.mean(iout_raw)),1e-30)) if target_iout>0.0 else np.zeros_like(iout_raw)
    time=theta/(2.0*math.pi*tank.fr_hz)
    warnings=[]
    if coll_rms>cfg.collocation_rms_tolerance:
        warnings.append(f"3P HB differential RMS residual {coll_rms:.3e} exceeds configured {cfg.collocation_rms_tolerance:.3e}.")
    if constraint_peak>cfg.constraint_tolerance:
        warnings.append(f"3P HB event/power constraint residual {constraint_peak:.3e} exceeds configured {cfg.constraint_tolerance:.3e}.")
    warnings.append(f"HB uses topology-specific {mode.value} hybrid-state collocation with H={','.join(str(int(v)) for v in h)}; peak collocation residual {coll_peak:.3e} is reported separately because of Gibbs error at state boundaries.")
    converged=bool(sol.success and coll_rms<=cfg.collocation_rms_tolerance and constraint_peak<=cfg.constraint_tolerance)
    return ThreePhaseHBResult(converged,mode,float(frequency_hz),tank.fr_hz,M,vo/vin,vo,pout,rn,coll_rms,coll_peak,constraint_peak,int(h[-1]),tuple(int(v) for v in h),ev,time,theta,m,jr,jm,mlm,src,ir,im,isec,vcr,vlm,iout,segs,tuple(warnings))


def solve_three_phase_hb_regulated(spec:LLCDesignSpec,tank:TankDesign,*,vbus_v:float|None=None,load_fraction:float=1.0,config:ThreePhaseHBConfig|None=None) -> ThreePhaseHBResult:
    """Solve the regulated 3P HB point using a local TD-seeded frequency root.

    The analytical 3P TD point supplies only the frequency neighborhood and
    Fourier initial seed. The returned state is still obtained by satisfying
    the topology-specific HB collocation/event/power equations. A local bracket
    avoids the expensive wide scan used in the alpha implementation.
    """
    cfg=config or ThreePhaseHBConfig(); vin=float(spec.vbus_nom_v if vbus_v is None else vbus_v)
    rload=spec.vout_v**2/max(spec.pout_w*float(load_fraction),1e-30)
    td=solve_three_phase_td_regulated(spec,tank,vbus_v=vin,load_fraction=load_fraction,samples_per_cycle=258)
    f0=td.switching_frequency_hz; target=spec.vout_v; cache={}

    def result_at(f:float)->ThreePhaseHBResult:
        # Keep trial points on the same hybrid branch as the TD seed.
        if td.mode==ThreePhaseMode.ABOVE_RESONANCE:
            f=max(float(f),tank.fr_hz*(1.0+2e-4))
        else:
            f=min(float(f),tank.fr_hz*(1.0-2e-4))
        f=float(np.clip(f,spec.minimum_frequency_hz,spec.maximum_frequency_hz))
        key=round(f,5)
        if key not in cache:
            cache[key]=solve_three_phase_hb_at_frequency(spec,tank,f,vbus_v=vin,load_resistance_ohm=rload,config=cfg)
        return cache[key]

    r0=result_at(f0)
    if abs(r0.output_voltage_v-target) <= max(0.05,5e-4*target):
        return r0

    # Probe symmetrically and then expand only if needed. Typical operating
    # points bracket within ±2%, reducing an HB regulated solve to a handful of
    # collocation solves instead of a broad frequency scan.
    probes=[0.98,1.02,0.95,1.05,0.90,1.10]
    samples=[(r0.switching_frequency_hz,r0.output_voltage_v-target)]
    for fac in probes:
        try:
            rr=result_at(f0*fac); samples.append((rr.switching_frequency_hz,rr.output_voltage_v-target))
        except Exception:
            continue
        samples=sorted({round(f,6):(f,y) for f,y in samples}.values())
        bracket=None
        for (fa,ya),(fb,yb) in zip(samples[:-1],samples[1:]):
            if ya==0.0 or ya*yb<0.0:
                bracket=(fa,fb);break
        if bracket is not None:
            root=float(brentq(lambda f: result_at(float(f)).output_voltage_v-target,
                              bracket[0],bracket[1],xtol=5e-3,rtol=2e-8,maxiter=14))
            return result_at(root)

    # No fabricated regulation result: expose the HB state at the TD frequency
    # and attach an explicit warning if the truncated basis cannot bracket Vo.
    return ThreePhaseHBResult(
        r0.converged,r0.mode,r0.switching_frequency_hz,r0.resonant_frequency_hz,
        r0.normalized_gain,r0.dc_gain_vo_vin,r0.output_voltage_v,r0.output_power_w,
        r0.residual_norm,r0.collocation_rms,r0.collocation_peak,r0.constraint_peak,
        r0.max_harmonic,r0.harmonics,r0.event_theta,r0.time_s,r0.theta,r0.mcr,r0.jlr,
        r0.jlm,r0.mlm,r0.source_norm,r0.i_resonant_a,r0.i_magnetizing_a,
        r0.i_secondary_phase_a,r0.v_cr_v,r0.v_lm_v,r0.i_output_rectified_a,r0.segments,
        r0.warnings+("HB regulated voltage root was not locally bracketed; result is evaluated at the converged 3P TD frequency.",),
    )


def solve_three_phase_hb_gain_curve(spec:LLCDesignSpec,tank:TankDesign,*,vbus_v:float|None=None,load_fraction:float=1.0,frequencies_hz:NDArray[np.float64]|None=None,points:int=36,config:ThreePhaseHBConfig|None=None):
    vin=float(spec.vbus_nom_v if vbus_v is None else vbus_v);rload=spec.vout_v**2/max(spec.pout_w*float(load_fraction),1e-30);freqs=np.geomspace(spec.minimum_frequency_hz,spec.maximum_frequency_hz,int(points)) if frequencies_hz is None else np.asarray(frequencies_hz,dtype=float)
    M=np.full_like(freqs,np.nan);dc=np.full_like(freqs,np.nan);vo=np.full_like(freqs,np.nan)
    for i,f in enumerate(freqs):
        try:r=solve_three_phase_hb_at_frequency(spec,tank,float(f),vbus_v=vin,load_resistance_ohm=rload,config=config)
        except Exception:continue
        if r.converged:M[i]=r.normalized_gain;dc[i]=r.dc_gain_vo_vin;vo[i]=r.output_voltage_v
    return freqs,M,dc,vo


__all__=["ThreePhaseHBConfig","ThreePhaseHBResult","solve_three_phase_hb_at_frequency","solve_three_phase_hb_regulated","solve_three_phase_hb_gain_curve"]
