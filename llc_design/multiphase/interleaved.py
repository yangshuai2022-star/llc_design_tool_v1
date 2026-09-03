"""Two- and three-phase interleaved LLC analysis.

V8 intentionally fixes the phase definitions requested by the project:
  * 2 phase: 0 / 90 electrical degrees of the LLC switching period
  * 3 phase: 0 / 120 / 240 degrees
No custom phase-offset API is exposed.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping
import numpy as np
from numpy.typing import NDArray

from ..analysis import FidelityLevel, GoldenSolverConfig, LLCAnalysisRequest, LLCGoldenSolver, LLCModelResult
from ..analysis.harmonic_balance import HarmonicBalanceConfig
from ..analysis.time_domain import TimeDomainConfig
from ..core.spec import LLCDesignSpec
from ..dynamics.waveforms import WaveformBundle, WaveformSignal, signal_statistics


@dataclass(frozen=True)
class InterleavedPhaseResult:
    index: int
    phase_offset_deg: float
    electrical: LLCModelResult
    output_power_w: float
    share_percent: float


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
    warnings: tuple[str,...]=()


def fixed_phase_offsets_deg(phase_count: int) -> tuple[float,...]:
    if phase_count==2: return (0.0,90.0)
    if phase_count==3: return (0.0,120.0,240.0)
    raise ValueError("V8 interleaved LLC supports only fixed 2-phase or 3-phase systems")


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


def _select_model(analysis, fidelity: FidelityLevel|None):
    if fidelity is None: return analysis.reference
    if fidelity in analysis.results: return analysis.results[fidelity]
    return analysis.reference


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
    offsets=fixed_phase_offsets_deg(phase_count); vbus=float(spec.vbus_nom_v if vbus_v is None else vbus_v)
    overrides=phase_spec_overrides or {}; results=[]
    # The physical tank of each cell is designed for 1/N of system rated power.
    base_phase=spec.clone(pout_w=spec.pout_w/phase_count)
    golden=LLCGoldenSolver(GoldenSolverConfig(
        include_fha=True,include_harmonic_balance=True,include_time_domain=(fidelity==FidelityLevel.SWITCHED_TIME_DOMAIN),
        harmonic_balance=HarmonicBalanceConfig(samples_per_cycle=max(256,samples_per_cycle),output_cycles=1,max_harmonic=7),
        time_domain=TimeDomainConfig(samples_per_cycle=max(256,min(samples_per_cycle,1024)),output_cycles=1),strict=False,
    ))
    symmetric_model = None
    if not overrides:
        req=LLCAnalysisRequest(base_phase,vbus_v=vbus,load_fraction=load_fraction,waveform_cycles=1,samples_per_cycle=samples_per_cycle)
        symmetric_model=_select_model(golden.solve(req),fidelity)
    for idx,offset in enumerate(offsets):
        if symmetric_model is not None:
            model=symmetric_model
        else:
            pspec=base_phase.clone(**dict(overrides.get(idx,{}))) if idx in overrides else base_phase
            req=LLCAnalysisRequest(pspec,vbus_v=vbus,load_fraction=load_fraction,waveform_cycles=1,samples_per_cycle=samples_per_cycle)
            model=_select_model(golden.solve(req),fidelity)
        results.append((offset,model))
    # Use a common system timebase; modest frequency mismatch is preserved as a warning,
    # while waveform superposition assumes the commanded common switching frequency.
    fs_values=[r.metrics.switching_frequency_hz for _,r in results]; fs=float(np.mean(fs_values)); n=min(len(_one_cycle(r,'i_rectified')) for _,r in results)
    total_irect=np.zeros(n); total_bus=np.zeros(n); phase_irect=[]; phase_ir=[]; phase_p=[]
    for offset,r in results:
        frac=offset/360.0
        irect=_shift_periodic(_one_cycle(r,'i_rectified')[:n],frac); ir=_shift_periodic(_one_cycle(r,'i_resonant')[:n],frac); vb=_shift_periodic(_one_cycle(r,'v_bridge')[:n],frac)
        total_irect+=irect; total_bus+=vb*ir/max(vbus,1e-9); phase_irect.append(irect); phase_ir.append(ir); phase_p.append(spec.vout_v*float(np.mean(irect)))
    total_power=sum(phase_p); shares=[100*p/max(total_power,1e-12) for p in phase_p]; avg=100/phase_count; imbalance=(max(shares)-min(shares))/avg*100 if shares else 0.0
    system_power=spec.pout_w*load_fraction; rload=spec.vout_v**2/max(system_power,1e-12); vo,iload,ico=_output_network(total_irect,fs,spec.vout_v,rload,spec.output_capacitance_f,spec.output_cap_esr_ohm)
    time=np.arange(n)/(fs*n); bus_ripple=total_bus-float(np.mean(total_bus))
    raw={
        'i_rectified_total':('总整流输出电流','A',total_irect,'interleaved','各相固定交错后的整流电流叠加',2*fs),
        'i_output_cap_total':('总输出电容电流','A',ico,'interleaved','系统 Irect-Iload',2*fs),
        'v_output_system':('系统输出电压','V',vo,'interleaved','总输出电容网络的周期纹波',2*fs),
        'i_bus_total':('总母线输入电流','A',total_bus,'interleaved','各相瞬时输入功率/Vbus',fs),
        'i_bus_ripple':('母线输入纹波电流','A',bus_ripple,'interleaved','去除平均值后的总母线电流',fs),
    }
    for idx,(offset,r) in enumerate(results):
        raw[f'i_rectified_p{idx+1}']=(f'Phase {idx+1} 整流电流','A',phase_irect[idx],'interleaved',f'固定相位 {offset:g}°',2*fs)
        raw[f'i_resonant_p{idx+1}']=(f'Phase {idx+1} 谐振电流','A',phase_ir[idx],'interleaved',f'固定相位 {offset:g}°',fs)
    signals={}
    for key,(label,unit,values,group,desc,fund) in raw.items():
        signals[key]=WaveformSignal(key,label,unit,values,signal_statistics(values,samples_per_period=n,switching_frequency_hz=fs,fundamental_frequency_hz=fund),group,desc)
    warnings=[]
    freq_spread=(max(fs_values)-min(fs_values))/max(fs,1e-9)*100
    if freq_spread>0.1: warnings.append(f'Per-phase natural regulated frequencies differ by {freq_spread:.3f}%; a real interleaved controller must enforce a common Fsw/current-sharing trim.')
    bundle=WaveformBundle(time,fs,f'V8 {phase_count}-phase interleaved LLC',signals,tuple(warnings),{'phase_count':phase_count,'phase_offsets_deg':','.join(str(v) for v in offsets),'samples_per_cycle':n})
    phases=tuple(InterleavedPhaseResult(i+1,results[i][0],results[i][1],phase_p[i],shares[i]) for i in range(phase_count))
    return InterleavedLLCResult(phase_count,offsets,phases,bundle,float(np.sqrt(np.mean(ico**2))),float(np.ptp(vo)),float(np.sqrt(np.mean(bus_ripple**2))),float(imbalance),float(total_power),tuple(warnings))
