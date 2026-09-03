"""Golden-waveform magnetic analysis for LLC V8.

Unlike the legacy FHA magnetic reconstruction, this module consumes the actual
V8 model waveform.  Transformer copper harmonics, iGSE core loss, leakage,
inter-winding capacitance and thermal iteration therefore all refer to the
same electrical operating point used by ZVS/SR analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from ..analysis.types import LLCModelResult
from ..core.spec import LLCDesignSpec
from .transformer import TransformerDesign
from .litz import StackLayer, WindingLossBreakdown, distribute_turns, layered_litz_stack_loss

MU0=4e-7*math.pi
EPS0=8.8541878128e-12


@dataclass(frozen=True)
class EnhancedMagneticsResult:
    core_loss_w: float
    primary_copper_w: float
    secondary_copper_w: float
    total_loss_w: float
    primary_ac_factor: float
    secondary_ac_factor: float
    b_peak_t: float
    b_peak_min_area_t: float
    leakage_inductance_h: float
    primary_secondary_capacitance_f: float
    hotspot_c: float
    primary_loss: WindingLossBreakdown
    secondary_loss: WindingLossBreakdown
    warnings: tuple[str,...]=()


def _one_cycle(result: LLCModelResult, key: str) -> tuple[np.ndarray,np.ndarray]:
    fs=result.metrics.switching_frequency_hz; t=np.asarray(result.waveform.time_s); x=np.asarray(result.waveform.signal(key).values)
    dt=float(np.median(np.diff(t))); n=max(32,min(len(x),int(round((1/fs)/dt))))
    return t[:n]-t[0],x[:n]


def _flux_from_voltage(time: np.ndarray, voltage: np.ndarray, turns: int, area_m2: float) -> np.ndarray:
    dt=float(np.median(np.diff(time))); v=voltage-float(np.mean(voltage)); flux=np.cumsum(v)*dt/(turns*area_m2)
    drift=np.linspace(0,float(flux[-1]),len(flux),endpoint=False); flux=flux-drift; flux-=float(np.mean(flux)); return flux


def _stack(design: TransformerDesign, primary: np.ndarray, secondary: np.ndarray) -> tuple[StackLayer,...]:
    p1=design.primary_turns//2; p2=design.primary_turns-p1
    p1l=distribute_turns(p1,design.primary_turns_per_layer); p2l=distribute_turns(p2,design.primary_turns_per_layer)
    sl=distribute_turns(design.secondary_turns,design.secondary_turns_per_layer)
    p=tuple(float(v) for v in primary); s=tuple(float(v) for v in secondary); layers=[]
    for turns in p1l: layers.append(StackLayer('primary',turns,turns*design.core.mlt_primary_mm*1e-3,design.primary_wire,p))
    for turns in sl: layers.append(StackLayer('secondary',turns,turns*design.core.mlt_secondary_mm*1e-3,design.secondary_wire,s))
    for turns in p2l: layers.append(StackLayer('primary',turns,turns*design.core.mlt_primary_mm*1e-3,design.primary_wire,p))
    return tuple(layers)


def estimate_leakage_from_stack(design: TransformerDesign) -> float:
    """1-D field-energy leakage estimate for the physical P/2-S-P/2 stack."""
    p1=distribute_turns(design.primary_turns//2,design.primary_turns_per_layer)
    p2=distribute_turns(design.primary_turns-design.primary_turns//2,design.primary_turns_per_layer)
    ss=distribute_turns(design.secondary_turns,design.secondary_turns_per_layer)
    # (turns, current for 1 A primary terminal excitation, thickness, MLT)
    rows=[]
    tp=design.primary_wire.equivalent_outer_diameter_m; ts=design.secondary_wire.equivalent_outer_diameter_m
    for n in p1: rows.append((n,1.0,tp,design.core.mlt_primary_mm*1e-3))
    for n in ss: rows.append((n,-design.primary_turns/design.secondary_turns,ts,design.core.mlt_secondary_mm*1e-3))
    for n in p2: rows.append((n,1.0,tp,design.core.mlt_primary_mm*1e-3))
    ww=max(design.core.window_width_m,1e-9); mmf=0.0; energy=0.0
    for turns,current,thickness,mlt in rows:
        m0=mmf; m1=mmf+turns*current
        # Integral of a linearly varying H^2 across layer thickness.
        h2=(m0*m0+m0*m1+m1*m1)/(3.0*ww*ww)
        volume=ww*thickness*mlt
        energy += 0.5*MU0*h2*volume
        mmf=m1
    return max(2.0*energy,0.0)  # I_primary = 1 A


def estimate_interwinding_capacitance(design: TransformerDesign, insulation_thickness_m: float=0.20e-3, relative_permittivity: float=3.4) -> float:
    """First-order primary-secondary interface capacitance.

    The estimate intentionally reports a physical winding-interface quantity,
    not a fitted LLC tank capacitance.  It is suitable for high-frequency
    sensitivity studies and should be replaced by measured impedance data when
    available.
    """
    interfaces=2 if design.primary_layers_per_half>0 and design.secondary_layers>0 else 0
    overlap=design.core.window_width_m*0.5*(design.core.mlt_primary_mm+design.core.mlt_secondary_mm)*1e-3
    return interfaces*EPS0*relative_permittivity*overlap/max(insulation_thickness_m,1e-9)


def analyze_golden_magnetics(spec: LLCDesignSpec, result: LLCModelResult, design: TransformerDesign, *, insulation_thickness_m: float=0.20e-3, relative_permittivity: float=3.4) -> EnhancedMagneticsResult:
    time,primary=_one_cycle(result,'i_transformer_primary')
    _,secondary=_one_cycle(result,'i_transformer_secondary')
    _,vp=_one_cycle(result,'v_transformer_primary')
    if len(secondary)!=len(primary):
        secondary=np.interp(time,*_one_cycle(result,'i_transformer_secondary'))
    b=_flux_from_voltage(time,vp,design.primary_turns,design.core.ae_m2)
    bpk=float(np.max(np.abs(b))); bmin=bpk*design.core.ae_m2/design.core.amin_m2
    fs=result.metrics.switching_frequency_hz
    leakage=estimate_leakage_from_stack(design); cps=estimate_interwinding_capacitance(design,insulation_thickness_m,relative_permittivity)
    temp=max(spec.winding_temperature_c,spec.ambient_temperature_c); last=None
    for _ in range(spec.magnetic_thermal_max_iterations):
        core=design.core.core_loss_waveform_w(time,b,temp)
        losses=layered_litz_stack_loss(_stack(design,primary,secondary),fs,design.core.window_width_m,temp,
            max_harmonic=spec.litz_max_harmonic,transposition_quality=spec.litz_transposition_quality,
            sub_bundle_coupling_factor=spec.litz_sub_bundle_coupling_factor,termination_resistance_fraction=spec.winding_termination_resistance_fraction,
            calibration_factor=spec.litz_proximity_correction*spec.transformer_proximity_severity)
        p=losses['primary']; s=losses['secondary']; total=core+p.total_w+s.total_w
        pred=min(max(spec.ambient_temperature_c+total*spec.transformer_rth_k_per_w,spec.ambient_temperature_c),220.0)
        updated=0.55*temp+0.45*pred; last=(core,p,s,total)
        if abs(updated-temp)<=spec.magnetic_thermal_tolerance_c: temp=updated; break
        temp=updated
    core,p,s,total=last if last else (0.0,None,None,0.0)
    assert p is not None and s is not None
    warnings=list(design.core.loss_range_warnings(fs,bpk))
    if bmin>spec.transformer_max_b_t: warnings.append('Minimum-area Bpk exceeds configured transformer flux limit.')
    if temp>spec.magnetic_hotspot_limit_c: warnings.append('Estimated transformer hotspot exceeds configured limit.')
    return EnhancedMagneticsResult(float(core),p.total_w,s.total_w,float(total),p.effective_ac_factor,s.effective_ac_factor,bpk,bmin,leakage,cps,temp,p,s,tuple(warnings))
