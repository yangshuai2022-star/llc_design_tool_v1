"""Fixed-angle 2P/3P interleaved LLC system analysis.

Unlike V8.2, this implementation does *not* solve one 1/N-power cell at its own
regulated frequency and then blindly phase-shift it.  The electrical layer
first solves the common switching frequency and self-consistent phase powers at
the shared output node.  Fixed 90°/120° offsets are applied only after that.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
import math
import numpy as np
from numpy.typing import NDArray

from ..analysis import FidelityLevel, LLCAnalysisRequest, LLCModelResult, solve_fha, solve_harmonic_balance, solve_time_domain
from ..analysis.harmonic_balance import HarmonicBalanceConfig
from ..analysis.time_domain import TimeDomainConfig
from ..core.spec import LLCDesignSpec
from ..dynamics.waveforms import WaveformBundle, WaveformSignal, signal_statistics
from .electrical import (
    InterleavedElectricalPoint, SystemGainCurve, MultiphaseTopology, fixed_phase_offsets_deg,
    solve_interleaved_electrical_point, solve_system_gain_curve,
)


@dataclass(frozen=True)
class InterleavedPhaseResult:
    index: int
    phase_offset_deg: float
    electrical: LLCModelResult | None
    output_power_w: float
    share_percent: float
    q_effective: float = 0.0
    input_phase_deg: float = 0.0


@dataclass(frozen=True)
class InterleavedLLCResult:
    phase_count: int
    phase_offsets_deg: tuple[float,...]
    phases: tuple[InterleavedPhaseResult,...]
    waveform: WaveformBundle
    output_capacitor_rms_a: float
    output_ripple_vpp: float
    input_bus_ripple_rms_a: float
    current_share_imbalance_percent: float
    total_output_power_w: float
    common_switching_frequency_hz: float = 0.0
    electrical_point: InterleavedElectricalPoint | None = None
    gain_curve: SystemGainCurve | None = None
    warnings: tuple[str,...]=()


def _shift_periodic(values: NDArray[np.float64], shift_fraction: float) -> NDArray[np.float64]:
    n=len(values); x=np.arange(n,dtype=float); source=np.mod(x-shift_fraction*n,n)
    xp=np.arange(n+1,dtype=float); fp=np.r_[values,values[0]]
    return np.interp(source,xp,fp).astype(float)


def _one_cycle(result: LLCModelResult,key: str) -> NDArray[np.float64]:
    fs=result.metrics.switching_frequency_hz; t=result.waveform.time_s; dt=float(np.median(np.diff(t))); n=max(32,min(len(t),int(round((1/fs)/dt))))
    return np.asarray(result.waveform.signal(key).values[:n],dtype=float)


def _output_network(irect: NDArray[np.float64], fs: float, vdc: float, rload: float, c: float, esr: float) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    n=len(irect); dt=1/(fs*n); ac=irect-float(np.mean(irect)); sp=np.fft.rfft(ac); freqs=np.fft.rfftfreq(n,dt); vo_sp=np.zeros_like(sp,dtype=complex)
    for k in range(1,len(sp)):
        w=2*math.pi*freqs[k]; zc=esr+1/(1j*w*c); zp=1/(1/rload+1/zc); vo_sp[k]=sp[k]*zp
    vo=vdc+np.fft.irfft(vo_sp,n=n); iload=vo/rload; ico=irect-iload
    return vo,iload,ico


def _solve_phase_waveform(pspec: LLCDesignSpec, *, common_fs: float, power_w: float, vbus_v: float, fidelity: FidelityLevel, samples_per_cycle: int) -> LLCModelResult:
    load_fraction=max(power_w/max(pspec.pout_w,1e-12),1e-6)
    req=LLCAnalysisRequest(pspec,vbus_v=vbus_v,load_fraction=min(load_fraction,1.5),frequency_hz=common_fs,regulate_output=False,waveform_cycles=1,samples_per_cycle=samples_per_cycle)
    if fidelity == FidelityLevel.FHA:
        return solve_fha(req)
    if fidelity == FidelityLevel.SWITCHED_TIME_DOMAIN:
        return solve_time_domain(req, TimeDomainConfig(samples_per_cycle=max(256,min(samples_per_cycle,1024)),output_cycles=1))
    return solve_harmonic_balance(req,HarmonicBalanceConfig(samples_per_cycle=max(256,samples_per_cycle),output_cycles=1,max_harmonic=7))


def _synthesize_three_phase_star_waveforms(
    spec: LLCDesignSpec,
    electrical: InterleavedElectricalPoint,
    *,
    vbus_v: float,
    load_fraction: float,
    samples_per_cycle: int,
) -> tuple[WaveformBundle, float, float, float]:
    """Build a topology-aware 3P Y-LLC FHA waveform bundle.

    The electrical point already contains the floating-neutral coupled phase
    currents.  This routine deliberately does not call the single-phase HB/TD
    solver, because doing so would erase the three-phase neutral coupling that
    changes the gain/current characteristics.
    """
    fs=electrical.switching_frequency_hz; n=max(256,int(samples_per_cycle)); time=np.arange(n,dtype=float)/(fs*n); theta=2*math.pi*fs*time
    total_irect=np.zeros(n); total_bus=np.zeros(n); phase_irect=[]; phase_ir=[]
    for p in electrical.phases:
        ang=math.radians(p.resonant_current_angle_deg)
        ir=math.sqrt(2.0)*p.resonant_current_rms_a*np.cos(theta+ang)
        # The 3P electrical model reports each phase's useful DC output power.
        # Use a full-wave rectified sinusoidal current shape, normalized to the
        # exact phase DC current.  Summing the three 120-deg phases produces the
        # characteristic six-pulse output-ripple family without pretending to
        # be a nonlinear switched rectifier solution.
        # Rectifier conduction follows the reflected-load branch, not the
        # total resonant current (which also contains magnetizing current).
        load_ang=math.radians(p.reflected_load_current_angle_deg)
        shape=np.abs(np.cos(theta+load_ang)); shape/=max(float(np.mean(shape)),1e-30)
        irect=(p.output_power_w/max(spec.vout_v,1e-12))*shape
        source_ang=math.radians(-p.phase_offset_deg)
        vpole=0.5*vbus_v*np.where(np.cos(theta+source_ang)>=0.0,1.0,-1.0)
        total_irect+=irect; total_bus+=vpole*ir/max(vbus_v,1e-12); phase_irect.append(irect); phase_ir.append(ir)
    target_power=spec.pout_w*load_fraction; rload=spec.vout_v**2/max(target_power,1e-12)
    vo,_,ico=_output_network(total_irect,fs,spec.vout_v,rload,spec.output_capacitance_f,spec.output_cap_esr_ohm)
    bus_ripple=total_bus-float(np.mean(total_bus))
    raw={
        'i_rectified_total':('总整流输出电流','A',total_irect,'interleaved','3P Y-LLC FHA 相电流整流叠加',6*fs),
        'i_output_cap_total':('总输出电容电流','A',ico,'interleaved','系统 Irect-Iload',6*fs),
        'v_output_system':('系统输出电压','V',vo,'interleaved','3P 输出电容网络周期纹波',6*fs),
        'i_bus_total':('总母线输入电流','A',total_bus,'interleaved','三桥臂瞬时输入功率/Vbus',3*fs),
        'i_bus_ripple':('母线输入纹波电流','A',bus_ripple,'interleaved','去除平均值后的总母线电流',3*fs),
    }
    for idx,p in enumerate(electrical.phases):
        raw[f'i_rectified_p{idx+1}']=(f'Phase {idx+1} 整流电流','A',phase_irect[idx],'interleaved',f'Y-phase {p.phase_offset_deg:g}°',2*fs)
        raw[f'i_resonant_p{idx+1}']=(f'Phase {idx+1} 谐振电流','A',phase_ir[idx],'interleaved',f'Y-phase {p.phase_offset_deg:g}°',fs)
    signals={key:WaveformSignal(key,label,unit,values,signal_statistics(values,samples_per_period=n,switching_frequency_hz=fs,fundamental_frequency_hz=fund),group,desc) for key,(label,unit,values,group,desc,fund) in raw.items()}
    warnings=tuple(electrical.warnings)+("3P waveform is the topology-specific floating-neutral FHA reconstruction; single-phase HB/TD fan-out is intentionally disabled until the dedicated 3P nonlinear solver is used.",)
    bundle=WaveformBundle(time,fs,'V8.3 3-phase Y-connected 120deg LLC',signals,warnings,{'phase_count':3,'phase_offsets_deg':'0,120,240','samples_per_cycle':n,'common_frequency_hz':fs,'electrical_model':'three_phase_star_floating_neutral_fha'})
    return bundle,float(np.sqrt(np.mean(ico**2))),float(np.ptp(vo)),float(np.sqrt(np.mean(bus_ripple**2)))


def solve_interleaved_llc(
    spec: LLCDesignSpec,
    phase_count: int,
    *,
    vbus_v: float|None=None,
    load_fraction: float=1.0,
    fidelity: FidelityLevel|None=FidelityLevel.HARMONIC_BALANCE,
    phase_spec_overrides: Mapping[int,Mapping[str,object]]|None=None,
    samples_per_cycle: int=1024,
) -> InterleavedLLCResult:
    fidelity = fidelity or FidelityLevel.HARMONIC_BALANCE
    offsets=fixed_phase_offsets_deg(phase_count); vbus=float(spec.vbus_nom_v if vbus_v is None else vbus_v)
    electrical=solve_interleaved_electrical_point(spec,phase_count,vbus_v=vbus,load_fraction=load_fraction,phase_spec_overrides=phase_spec_overrides)
    gain_curve=solve_system_gain_curve(spec,phase_count,vbus_v=vbus,load_fraction=load_fraction,phase_spec_overrides=phase_spec_overrides,points=100)

    # True 3P 120-deg Y-connected electrical model.  Do not fan-out a 1P HB/TD
    # waveform: the floating neutral couples the phases and changes the gain.
    if electrical.topology == MultiphaseTopology.THREE_PHASE_STAR_120:
        bundle,cout_rms,vo_pp,bus_rms=_synthesize_three_phase_star_waveforms(spec,electrical,vbus_v=vbus,load_fraction=load_fraction,samples_per_cycle=samples_per_cycle)
        powers=[p.output_power_w for p in electrical.phases]; avg=sum(powers)/3.0; imbalance=(max(powers)-min(powers))/max(avg,1e-12)*100.0
        warnings=list(bundle.warnings)
        if fidelity != FidelityLevel.FHA:
            warnings.append(f'Requested {fidelity.value}; 3P system gain/current sharing remains topology-specific FHA in V8.3 rather than an invalid single-phase {fidelity.value} fan-out.')
        phases=tuple(InterleavedPhaseResult(p.index,p.phase_offset_deg,None,p.output_power_w,p.share_percent,p.q_effective,p.input_phase_deg) for p in electrical.phases)
        # Replace bundle warnings with the full set.
        bundle=WaveformBundle(bundle.time_s,bundle.switching_frequency_hz,bundle.model_name,bundle.signals,tuple(warnings),bundle.metadata)
        return InterleavedLLCResult(3,offsets,phases,bundle,cout_rms,vo_pp,bus_rms,float(imbalance),float(sum(powers)),float(electrical.switching_frequency_hz),electrical,gain_curve,tuple(warnings))

    # 2P/90 independent-cell topology: each phase keeps its own nonlinear model,
    # but all phases are constrained to the common Fsw and solved power share.
    base_phase=spec.clone(pout_w=spec.pout_w/phase_count); overrides=phase_spec_overrides or {}
    phase_specs=tuple(base_phase.clone(**dict(overrides.get(i,{}))) if i in overrides else base_phase for i in range(phase_count))
    models=[]
    for p,ps in zip(electrical.phases,phase_specs):
        models.append(_solve_phase_waveform(ps,common_fs=electrical.switching_frequency_hz,power_w=p.output_power_w,vbus_v=vbus,fidelity=fidelity,samples_per_cycle=samples_per_cycle))
    fs=electrical.switching_frequency_hz; n=min(len(_one_cycle(r,'i_rectified')) for r in models)
    total_irect=np.zeros(n); total_bus=np.zeros(n); phase_irect=[]; phase_ir=[]
    for p,r in zip(electrical.phases,models):
        frac=p.phase_offset_deg/360.0
        irect0=_one_cycle(r,'i_rectified')[:n]
        target_avg=p.output_power_w/max(spec.vout_v,1e-12); actual_avg=float(np.mean(irect0)); scale=target_avg/max(actual_avg,1e-12)
        irect=_shift_periodic(irect0*scale,frac)
        ir=_shift_periodic(_one_cycle(r,'i_resonant')[:n],frac); vb=_shift_periodic(_one_cycle(r,'v_bridge')[:n],frac)
        total_irect+=irect; total_bus+=vb*ir/max(vbus,1e-9); phase_irect.append(irect); phase_ir.append(ir)
    target_power=spec.pout_w*load_fraction; rload=spec.vout_v**2/max(target_power,1e-12); vo,_,ico=_output_network(total_irect,fs,spec.vout_v,rload,spec.output_capacitance_f,spec.output_cap_esr_ohm)
    time=np.arange(n)/(fs*n); bus_ripple=total_bus-float(np.mean(total_bus))
    raw={
        'i_rectified_total':('总整流输出电流','A',total_irect,'interleaved','2P 自洽相功率 + 固定 90° 后整流电流叠加',2*fs),
        'i_output_cap_total':('总输出电容电流','A',ico,'interleaved','系统 Irect-Iload',2*fs),
        'v_output_system':('系统输出电压','V',vo,'interleaved','总输出电容网络的周期纹波',2*fs),
        'i_bus_total':('总母线输入电流','A',total_bus,'interleaved','各相瞬时输入功率/Vbus',fs),
        'i_bus_ripple':('母线输入纹波电流','A',bus_ripple,'interleaved','去除平均值后的总母线电流',fs),
    }
    for idx,p in enumerate(electrical.phases):
        raw[f'i_rectified_p{idx+1}']=(f'Phase {idx+1} 整流电流','A',phase_irect[idx],'interleaved',f'固定相位 {p.phase_offset_deg:g}°',2*fs)
        raw[f'i_resonant_p{idx+1}']=(f'Phase {idx+1} 谐振电流','A',phase_ir[idx],'interleaved',f'固定相位 {p.phase_offset_deg:g}°',fs)
    signals={key:WaveformSignal(key,label,unit,values,signal_statistics(values,samples_per_period=n,switching_frequency_hz=fs,fundamental_frequency_hz=fund),group,desc) for key,(label,unit,values,group,desc,fund) in raw.items()}
    powers=[p.output_power_w for p in electrical.phases]; avg=sum(powers)/phase_count; imbalance=(max(powers)-min(powers))/max(avg,1e-12)*100
    warnings=list(electrical.warnings); model_vo=[m.metrics.output_voltage_v for m in models]; spread=max(model_vo)-min(model_vo)
    if spread>0.02*spec.vout_v: warnings.append(f'{fidelity.value} fixed-frequency phase models disagree by {spread:.3f} V; FHA common-node load sharing remains the 2P system constraint in this release.')
    bundle=WaveformBundle(time,fs,'V8.3 2-phase 90deg parallel LLC',signals,tuple(warnings),{'phase_count':phase_count,'phase_offsets_deg':','.join(str(v) for v in offsets),'samples_per_cycle':n,'common_frequency_hz':fs,'electrical_model':'two_phase_parallel_common_output'})
    phases=tuple(InterleavedPhaseResult(p.index,p.phase_offset_deg,models[i],p.output_power_w,p.share_percent,p.q_effective,p.input_phase_deg) for i,p in enumerate(electrical.phases))
    return InterleavedLLCResult(phase_count,offsets,phases,bundle,float(np.sqrt(np.mean(ico**2))),float(np.ptp(vo)),float(np.sqrt(np.mean(bus_ripple**2))),float(imbalance),float(sum(powers)),float(fs),electrical,gain_curve,tuple(warnings))
