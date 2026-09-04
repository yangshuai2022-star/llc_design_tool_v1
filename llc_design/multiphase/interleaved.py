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
    PhaseElectricalPoint, InterleavedElectricalPoint, SystemGainCurve, MultiphaseTopology, fixed_phase_offsets_deg,
    solve_interleaved_electrical_point, solve_system_gain_curve, three_phase_design_context,
)
from .star_time_domain import (
    ThreePhaseTDResult, solve_three_phase_td_regulated, solve_three_phase_td_gain_curve,
)
from .star_harmonic_balance import (
    ThreePhaseHBConfig, ThreePhaseHBResult, solve_three_phase_hb_regulated, solve_three_phase_hb_gain_curve,
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


def _three_phase_fundamental_impedance_angle(v: NDArray[np.float64], i: NDArray[np.float64]) -> float:
    if len(v) < 8 or len(i) != len(v):
        return 0.0
    vf=np.fft.rfft(v); inf=np.fft.rfft(i)
    if len(vf)<2 or abs(inf[1])<1e-18:
        return 0.0
    return float(np.degrees(np.angle(vf[1]/inf[1])))


def _three_phase_electrical_from_model(
    spec: LLCDesignSpec,
    *,
    vbus_v: float,
    model: ThreePhaseTDResult | ThreePhaseHBResult,
    tank,
    rac_ohm: float,
    topology_spec: LLCDesignSpec,
) -> InterleavedElectricalPoint:
    ptotal=float(model.output_power_w); pphase=ptotal/3.0; q=tank.zr_ohm/rac_ohm
    source=model.source_norm*((2.0/3.0)*vbus_v)
    angle=_three_phase_fundamental_impedance_angle(source,model.i_resonant_a)
    src_rms=float(np.sqrt(np.mean(source*source)))
    iload_primary=model.i_secondary_phase_a/max(topology_spec.turns_ratio,1e-30)
    iload_rms=float(np.sqrt(np.mean(iload_primary*iload_primary)))
    required_m=topology_spec.turns_ratio*(spec.vout_v+spec.rectifier_equivalent_drop_v)/vbus_v
    phases=tuple(
        PhaseElectricalPoint(
            idx+1,off,pphase,100.0/3.0,q,model.normalized_gain,required_m,angle,
            model.resonant_current_rms_a,tank,off-angle,src_rms,topology_spec.turns_ratio,
            topology_spec.primary_turns,topology_spec.secondary_turns,iload_rms,off,
        )
        for idx,off in enumerate(fixed_phase_offsets_deg(3))
    )
    warnings=list(model.warnings)
    warnings.append("3P work point is solved with the shared Y/Y six-pulse-rectifier topology; phase powers are balanced by topology assumption, not forced independent-cell loads.")
    return InterleavedElectricalPoint(
        3,fixed_phase_offsets_deg(3),model.switching_frequency_hz,model.output_voltage_v,
        ptotal,phases,tuple(warnings),bool(model.converged),float(model.residual_norm),
        MultiphaseTopology.THREE_PHASE_STAR_120,
    )


def _three_phase_gain_curve(
    spec: LLCDesignSpec,
    *,
    topology_spec: LLCDesignSpec,
    tank,
    fidelity: FidelityLevel,
    vbus_v: float,
    load_fraction: float,
    requested_points: int,
) -> SystemGainCurve:
    pnom=spec.pout_w*load_fraction; rload=spec.vout_v**2/max(pnom,1e-30)
    if fidelity == FidelityLevel.FHA:
        return solve_system_gain_curve(spec,3,vbus_v=vbus_v,load_fraction=load_fraction,points=requested_points)
    if fidelity == FidelityLevel.SWITCHED_TIME_DOMAIN:
        f,m,dc,vo=solve_three_phase_td_gain_curve(
            topology_spec,tank,vbus_v=vbus_v,load_fraction=1.0,
            points=min(int(requested_points),72),
        )
    else:
        # Topology-specific HB is materially more expensive than the analytical
        # piecewise TD solver. Keep the interactive curve sparse; callers can
        # request a denser dedicated HB curve explicitly from the public API.
        f,m,dc,vo=solve_three_phase_hb_gain_curve(
            topology_spec,tank,vbus_v=vbus_v,load_fraction=1.0,
            points=min(int(requested_points),24),
            config=ThreePhaseHBConfig(max_harmonic=11,collocation_points=72,samples_per_cycle=258,max_nfev=500),
        )
    ptotal=vo*vo/rload
    pphase=tuple((ptotal/3.0).copy() for _ in range(3))
    return SystemGainCurve(f,dc,vo,pphase,pnom,MultiphaseTopology.THREE_PHASE_STAR_120)


def _three_phase_waveform_bundle(
    spec: LLCDesignSpec,
    model: ThreePhaseTDResult | ThreePhaseHBResult,
    *,
    vbus_v: float,
) -> tuple[WaveformBundle,float,float,float]:
    fs=float(model.switching_frequency_hz); n=len(model.time_s)
    # Dedicated 3P solvers guarantee n%6==0 so 120° shifts are exact samples.
    shift=n//3
    def sh(x,k): return np.roll(np.asarray(x,dtype=float),k*shift)
    ir=[sh(model.i_resonant_a,k) for k in range(3)]
    im=[sh(model.i_magnetizing_a,k) for k in range(3)]
    isec=[sh(model.i_secondary_phase_a,k) for k in range(3)]
    vcr=[sh(model.v_cr_v,k) for k in range(3)]
    vlm=[sh(model.v_lm_v,k) for k in range(3)]
    vphase=[sh(model.source_norm*((2.0/3.0)*vbus_v),k) for k in range(3)]
    irect=np.asarray(model.i_output_rectified_a,dtype=float)
    rload=spec.vout_v**2/max(model.output_power_w,1e-30)
    vo,_,ico=_output_network(irect,fs,model.output_voltage_v,rload,spec.output_capacitance_f,spec.output_cap_esr_ohm)
    pin=np.zeros(n)
    for k in range(3): pin += vphase[k]*ir[k]
    ibus=pin/max(vbus_v,1e-30); bus_ripple=ibus-float(np.mean(ibus))
    model_tag='three_phase_y_shared_bridge_td' if isinstance(model,ThreePhaseTDResult) else 'three_phase_y_shared_bridge_hb'
    raw={
        'i_rectified_total':('总整流输出电流','A',irect,'interleaved','3P 共用六脉波整流桥输出电流',6*fs),
        'i_output_cap_total':('总输出电容电流','A',ico,'interleaved','系统 Irect-Iload',6*fs),
        'v_output_system':('系统输出电压','V',vo,'interleaved','共享输出电容网络周期纹波',6*fs),
        'i_bus_total':('总母线输入电流','A',ibus,'interleaved','三相瞬时输入功率/Vbus',3*fs),
        'i_bus_ripple':('母线输入纹波电流','A',bus_ripple,'interleaved','去除平均值后的总母线输入电流',3*fs),
    }
    for k,off in enumerate(fixed_phase_offsets_deg(3)):
        raw[f'i_resonant_p{k+1}']=(f'Phase {k+1} 谐振电流','A',ir[k],'interleaved',f'3P dedicated model, {off:g}°',fs)
        raw[f'i_magnetizing_p{k+1}']=(f'Phase {k+1} 励磁电流','A',im[k],'interleaved',f'3P dedicated model, {off:g}°',fs)
        raw[f'i_secondary_p{k+1}']=(f'Phase {k+1} 次级绕组电流','A',isec[k],'interleaved',f'3P dedicated model, {off:g}°',fs)
        raw[f'v_cr_p{k+1}']=(f'Phase {k+1} Cr 电压','V',vcr[k],'interleaved',f'3P dedicated model, {off:g}°',fs)
        raw[f'v_lm_p{k+1}']=(f'Phase {k+1} Lm 电压','V',vlm[k],'interleaved',f'3P dedicated model, {off:g}°',fs)
        raw[f'v_tank_input_p{k+1}']=(f'Phase {k+1} Tank 输入电压','V',vphase[k],'interleaved',f'3P 1/3-2/3 switching states, {off:g}°',fs)
    signals={key:WaveformSignal(key,label,unit,values,signal_statistics(values,samples_per_period=n,switching_frequency_hz=fs,fundamental_frequency_hz=fund),group,desc) for key,(label,unit,values,group,desc,fund) in raw.items()}
    warnings=tuple(model.warnings)
    bundle=WaveformBundle(
        np.asarray(model.time_s,dtype=float),fs,'V8.3 dedicated 3-phase 120deg LLC',signals,warnings,
        {'phase_count':3,'phase_offsets_deg':'0,120,240','samples_per_cycle':n,'common_frequency_hz':fs,
         'electrical_model':model_tag,'three_phase_mode':model.mode.value,'normalized_gain_m':model.normalized_gain,
         'solver_residual':model.residual_norm},
    )
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

    if phase_count == 3:
        if phase_spec_overrides:
            raise NotImplementedError("3P/120 shared-bridge MVP currently supports balanced phase parameters only; asymmetric phase parameters require the later complementarity model.")
        ctx=three_phase_design_context(spec,total_power_w=spec.pout_w*load_fraction)
        # V8.3 MVP policy: the topology-specific 3P FHA is the authoritative
        # gain/work-point model for first release. Dedicated TD/HB are allowed
        # to enrich waveforms, but a difficult nonlinear branch must never
        # block the user from validating a 3P design.
        electrical=solve_interleaved_electrical_point(spec,3,vbus_v=vbus,load_fraction=load_fraction)
        extra=[]
        if fidelity == FidelityLevel.HARMONIC_BALANCE:
            hb=solve_three_phase_hb_regulated(ctx.spec,ctx.tank,vbus_v=vbus,load_fraction=1.0,config=ThreePhaseHBConfig(max_harmonic=7,collocation_points=48,samples_per_cycle=max(258,samples_per_cycle),max_nfev=260))
            if hb.converged:
                model=hb
                electrical=_three_phase_electrical_from_model(spec,vbus_v=vbus,model=model,tank=ctx.tank,rac_ohm=ctx.rac_ohm,topology_spec=ctx.spec)
            else:
                model=solve_three_phase_td_regulated(ctx.spec,ctx.tank,vbus_v=vbus,load_fraction=1.0,samples_per_cycle=max(258,samples_per_cycle))
                extra.append(f"3P HB did not converge (residual={hb.residual_norm:.3e}); V8.3 MVP keeps the dedicated 3P FHA gain/work point and shows the topology-specific TD best-fit waveform instead.")
        else:
            model=solve_three_phase_td_regulated(ctx.spec,ctx.tank,vbus_v=vbus,load_fraction=1.0,samples_per_cycle=max(258,samples_per_cycle))
            if fidelity == FidelityLevel.FHA:
                extra.append("3P FHA selected: gain/work point is the dedicated Y-connected/shared-six-pulse-rectifier FHA model; TD is used only to provide a topology-specific waveform preview.")

        if isinstance(model,ThreePhaseTDResult) and model.converged and fidelity == FidelityLevel.SWITCHED_TIME_DOMAIN:
            electrical=_three_phase_electrical_from_model(spec,vbus_v=vbus,model=model,tank=ctx.tank,rac_ohm=ctx.rac_ohm,topology_spec=ctx.spec)
        elif isinstance(model,ThreePhaseTDResult) and not model.converged:
            extra.append(f"3P TD preferred-mode solve is best-fit only (residual={model.residual_norm:.3e}); dedicated 3P FHA remains authoritative for gain and regulated operating point in this MVP release.")

        # Fast-launch policy: expose one stable, topology-correct 3P gain curve
        # now. HB/TD gain curves are research-grade extensions and do not block
        # V8.3 user validation.
        gain_curve=solve_system_gain_curve(spec,3,vbus_v=vbus,load_fraction=load_fraction,points=100)
        bundle,cout_rms,vo_pp,bus_rms=_three_phase_waveform_bundle(spec,model,vbus_v=vbus)
        warnings=tuple(bundle.warnings)+tuple(extra)
        if warnings != bundle.warnings:
            md=dict(bundle.metadata); md['gain_reference_model']='three_phase_shared_bridge_fha'; md['waveform_converged']=bool(model.converged)
            bundle=WaveformBundle(bundle.time_s,bundle.switching_frequency_hz,bundle.model_name,bundle.signals,warnings,md)
        phases=tuple(InterleavedPhaseResult(p.index,p.phase_offset_deg,None,p.output_power_w,p.share_percent,p.q_effective,p.input_phase_deg) for p in electrical.phases)
        return InterleavedLLCResult(3,offsets,phases,bundle,cout_rms,vo_pp,bus_rms,0.0,float(electrical.total_output_power_w),float(electrical.switching_frequency_hz),electrical,gain_curve,warnings)

    electrical=solve_interleaved_electrical_point(spec,phase_count,vbus_v=vbus,load_fraction=load_fraction,phase_spec_overrides=phase_spec_overrides)
    gain_curve=solve_system_gain_curve(spec,phase_count,vbus_v=vbus,load_fraction=load_fraction,phase_spec_overrides=phase_spec_overrides,points=100)

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
