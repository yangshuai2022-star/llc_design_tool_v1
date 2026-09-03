"""Hybrid complementarity rectifier solver for LLC CCM/DCM operation.

The secondary rectifier is represented by three mutually exclusive states:
``CLAMP_POS``, ``OPEN`` and ``CLAMP_NEG``.  In a clamp state the secondary
bridge enforces +/-n*(Vo+Vdrop).  In OPEN, secondary current is exactly zero,
therefore Ir == Im and Lr/Lm carry the same current.  A new clamp state is
entered only when the open-circuit transformer voltage violates the output
clamp.  This is the LLC rectifier complementarity condition and removes the
artificial ``tanh(I/Ieps)`` conduction used by the legacy smooth TD model.

The numerical event timing is limited by the selected samples/cycle; topology
constraints themselves are enforced exactly after every step.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math

import numpy as np
from numpy.typing import NDArray

from .plant import DynamicPhasorModel, DynamicPhasorSteadyState, LLCPlantInputs
from .waveforms import WaveformBundle, WaveformSignal, signal_statistics
from ..analysis.waveform import build_standard_waveform_bundle


class RectifierState(IntEnum):
    CLAMP_NEG = -1
    OPEN = 0
    CLAMP_POS = 1


@dataclass(frozen=True)
class ComplementaritySimulationConfig:
    samples_per_cycle: int = 1024
    output_cycles: int = 2
    minimum_settling_cycles: int = 20
    maximum_settling_cycles: int = 400
    convergence_tolerance: float = 2e-7
    voltage_hysteresis_v: float = 1e-5

    def validate(self) -> None:
        if self.samples_per_cycle < 256:
            raise ValueError("complementarity solver requires at least 256 samples/cycle")
        if self.output_cycles < 1:
            raise ValueError("output_cycles must be >=1")
        if self.maximum_settling_cycles < self.minimum_settling_cycles:
            raise ValueError("maximum_settling_cycles must exceed minimum")
        if self.convergence_tolerance <= 0:
            raise ValueError("convergence tolerance must be positive")


@dataclass(frozen=True)
class _Alg:
    bridge_v: float
    vout_v: float
    vp_v: float
    vs_v: float
    isec_a: float
    irect_a: float
    ico_a: float
    mode: RectifierState


def _bridge_value(phase: float, level: float, deadtime_fraction: float) -> float:
    wrapped=phase%(2*math.pi)
    da=min(max(deadtime_fraction,0.0),0.45)*2*math.pi
    if da>0:
        for edge in (0.0,math.pi,2*math.pi):
            d=abs((wrapped-edge+math.pi)%(2*math.pi)-math.pi)
            if d<=da/2: return 0.0
    return level if wrapped<math.pi else -level


def _output_voltage(p, vco: float, irect: float, disturbance: float) -> float:
    r=p.load_resistance_ohm; esr=p.output_cap_esr_ohm
    return (r*vco+r*esr*(irect-disturbance))/(r+esr)


def _open_candidate(model: DynamicPhasorModel, inputs: LLCPlantInputs, phase: float, state: NDArray[np.float64]) -> tuple[float,float,float,float]:
    p=model.p
    ir,vcr,im,vco=map(float,state)
    i=0.5*(ir+im)
    vout=_output_voltage(p,vco,0.0,inputs.load_current_disturbance_a)
    bridge=_bridge_value(phase,p.bridge_gain*inputs.bus_voltage_v,p.primary_deadtime_s*inputs.switching_frequency_hz)
    di=(bridge-vcr-(p.series_resistance_ohm+p.magnetizing_series_resistance_ohm)*i)/(p.lr_h+p.lm_h)
    vp=p.magnetizing_series_resistance_ohm*i+p.lm_h*di
    clamp=p.turns_ratio*max(vout+p.rectifier_equivalent_drop_v,0.0)
    return vp,clamp,vout,bridge


def _select_mode(model: DynamicPhasorModel, inputs: LLCPlantInputs, phase: float, state: NDArray[np.float64], mode: RectifierState, hysteresis_v: float) -> RectifierState:
    p=model.p
    ir,_,im,_=map(float,state)
    ip=ir-im
    # Complementarity: a conducting branch cannot support reverse current.
    if mode==RectifierState.CLAMP_POS and ip<=0.0:
        mode=RectifierState.OPEN
    elif mode==RectifierState.CLAMP_NEG and ip>=0.0:
        mode=RectifierState.OPEN
    if mode==RectifierState.OPEN:
        vp,clamp,_,_=_open_candidate(model,inputs,phase,state)
        if vp>clamp+hysteresis_v:
            return RectifierState.CLAMP_POS
        if vp<-clamp-hysteresis_v:
            return RectifierState.CLAMP_NEG
    return mode


def _project_open(state: NDArray[np.float64]) -> NDArray[np.float64]:
    x=np.asarray(state,dtype=float).copy()
    # At a valid current-zero commutation Ir=Im.  Projection removes only the
    # tiny fixed-step event error and preserves the average inductor current.
    i=0.5*(x[0]+x[2]); x[0]=i; x[2]=i
    return x


def _alg_rhs(model: DynamicPhasorModel, inputs: LLCPlantInputs, phase: float, state: NDArray[np.float64], mode: RectifierState) -> tuple[_Alg,NDArray[np.float64]]:
    p=model.p
    ir,vcr,im,vco=map(float,state)
    bridge=_bridge_value(phase,p.bridge_gain*inputs.bus_voltage_v,p.primary_deadtime_s*inputs.switching_frequency_hz)
    if mode==RectifierState.OPEN:
        i=0.5*(ir+im)
        vout=_output_voltage(p,vco,0.0,inputs.load_current_disturbance_a)
        di=(bridge-vcr-(p.series_resistance_ohm+p.magnetizing_series_resistance_ohm)*i)/(p.lr_h+p.lm_h)
        vp=p.magnetizing_series_resistance_ohm*i+p.lm_h*di
        ico=-vout/p.load_resistance_ohm-inputs.load_current_disturbance_a
        rhs=np.asarray([di,i/p.cr_f,di,ico/p.output_capacitance_f],dtype=float)
        return _Alg(bridge,vout,vp,vp/p.turns_ratio,0.0,0.0,ico,mode),rhs
    sign=1.0 if mode==RectifierState.CLAMP_POS else -1.0
    isec=p.turns_ratio*(ir-im)
    irect=max(sign*isec,0.0)
    vout=_output_voltage(p,vco,irect,inputs.load_current_disturbance_a)
    vp=sign*p.turns_ratio*max(vout+p.rectifier_equivalent_drop_v,0.0)
    ico=irect-vout/p.load_resistance_ohm-inputs.load_current_disturbance_a
    rhs=np.asarray([
        (bridge-vcr-vp-p.series_resistance_ohm*ir)/p.lr_h,
        ir/p.cr_f,
        (vp-p.magnetizing_series_resistance_ohm*im)/p.lm_h,
        ico/p.output_capacitance_f,
    ],dtype=float)
    return _Alg(bridge,vout,vp,vp/p.turns_ratio,isec,irect,ico,mode),rhs


def _step(model,inputs,phase,state,mode,dt,hyst):
    mode=_select_mode(model,inputs,phase,state,mode,hyst)
    if mode==RectifierState.OPEN:
        state=_project_open(state)
    omega=2*math.pi*inputs.switching_frequency_hz
    # RK4 within the current topology; complementarity is re-evaluated at the
    # end of the step.  With >=1024 samples/cycle event uncertainty is <10 ns at 100 kHz.
    _,k1=_alg_rhs(model,inputs,phase,state,mode)
    _,k2=_alg_rhs(model,inputs,phase+omega*dt/2,state+dt*k1/2,mode)
    _,k3=_alg_rhs(model,inputs,phase+omega*dt/2,state+dt*k2/2,mode)
    _,k4=_alg_rhs(model,inputs,phase+omega*dt,state+dt*k3,mode)
    x=state+dt*(k1+2*k2+2*k3+k4)/6
    new_mode=_select_mode(model,inputs,phase+omega*dt,x,mode,hyst)
    if new_mode==RectifierState.OPEN:
        x=_project_open(x)
    return x,new_mode


def _advance_cycle(model,inputs,state,mode,dt,n,hyst):
    phase=0.0
    omega=2*math.pi*inputs.switching_frequency_hz
    x=np.asarray(state,dtype=float).copy(); m=mode
    for _ in range(n):
        x,m=_step(model,inputs,phase,x,m,dt,hyst); phase+=omega*dt
    return x,m


def simulate_complementarity_steady_state(model: DynamicPhasorModel, steady_state: DynamicPhasorSteadyState, config: ComplementaritySimulationConfig|None=None) -> WaveformBundle:
    cfg=config or ComplementaritySimulationConfig(); cfg.validate()
    inputs=steady_state.inputs; fs=inputs.switching_frequency_hz; period=1/fs; dt=period/cfg.samples_per_cycle
    state=np.asarray([steady_state.states[0],steady_state.states[2],steady_state.states[4],steady_state.states[6]],dtype=float)
    mode=RectifierState.OPEN
    scales=np.asarray([max(steady_state.resonant_current_peak_a,1.0),max(inputs.bus_voltage_v,10.0),max(steady_state.magnetizing_current_peak_a,1.0),max(steady_state.output_voltage_v,1.0)])
    mismatch=math.inf; cycles=0
    for c in range(cfg.maximum_settling_cycles):
        start=state.copy()
        state,mode=_advance_cycle(model,inputs,state,mode,dt,cfg.samples_per_cycle,cfg.voltage_hysteresis_v)
        mismatch=float(np.linalg.norm((state-start)/scales)); cycles=c+1
        if cycles>=cfg.minimum_settling_cycles and mismatch<=cfg.convergence_tolerance: break
    converged=mismatch<=cfg.convergence_tolerance
    count=cfg.output_cycles*cfg.samples_per_cycle
    ir=np.zeros(count); vcr=np.zeros(count); im=np.zeros(count); vco=np.zeros(count)
    bridge=np.zeros(count); vp=np.zeros(count); vout=np.zeros(count); mode_arr=np.zeros(count); isec=np.zeros(count); irect=np.zeros(count)
    phase=0.0; omega=2*math.pi*fs; m=mode
    for k in range(count):
        m=_select_mode(model,inputs,phase,state,m,cfg.voltage_hysteresis_v)
        if m==RectifierState.OPEN: state=_project_open(state)
        alg,_=_alg_rhs(model,inputs,phase,state,m)
        ir[k],vcr[k],im[k],vco[k]=state
        bridge[k]=alg.bridge_v; vp[k]=alg.vp_v; vout[k]=alg.vout_v; mode_arr[k]=int(m); isec[k]=alg.isec_a; irect[k]=alg.irect_a
        state,m=_step(model,inputs,phase,state,m,dt,cfg.voltage_hysteresis_v); phase+=omega*dt
    time=np.arange(count)*dt
    bundle=build_standard_waveform_bundle(
        time_s=time,switching_frequency_hz=fs,model_name="V8 Complementarity TD",
        bus_voltage_v=inputs.bus_voltage_v,primary_topology=model.p.primary_topology,
        primary_deadtime_s=model.p.primary_deadtime_s,turns_ratio=model.p.turns_ratio,
        load_resistance_ohm=model.p.load_resistance_ohm,output_capacitance_f=model.p.output_capacitance_f,
        output_cap_esr_ohm=model.p.output_cap_esr_ohm,series_resistance_ohm=model.p.series_resistance_ohm,
        resonant_inductance_h=model.p.lr_h,output_voltage_mean_v=float(np.mean(vout)),bridge_voltage_v=bridge,
        resonant_current_a=ir,resonant_capacitor_voltage_v=vcr,magnetizing_current_a=im,transformer_primary_voltage_v=vp,
        warnings=(() if converged else ("Complementarity orbit did not fully settle.",)),
        metadata={"converged":str(converged),"periodic_mismatch":mismatch,"settled_cycles":cycles,"samples_per_cycle":cfg.samples_per_cycle,"output_cycles":cfg.output_cycles,"rectifier_model":"complementarity"},
    )
    # Replace derived secondary traces with the exact complementarity currents
    # (OPEN state must be identically zero) and expose rectifier state.
    signals=dict(bundle.signals)
    for key,label,unit,values,group,desc,fund in [
        ("i_transformer_secondary","变压器次级电流 Is","A",isec,"secondary","互补整流模型：OPEN 状态严格为零",fs),
        ("i_rectified","SR 整流输出电流 Irect","A",irect,"secondary","互补整流后的输出电流",2*fs),
        ("rectifier_state","整流互补状态","state",mode_arr,"secondary","+1=正钳位, 0=DCM OPEN, -1=负钳位",fs),
    ]:
        signals[key]=WaveformSignal(key,label,unit,values,signal_statistics(values,samples_per_period=cfg.samples_per_cycle,switching_frequency_hz=fs,fundamental_frequency_hz=fund),group,desc)
    return WaveformBundle(bundle.time_s,bundle.switching_frequency_hz,bundle.model_name,signals,bundle.warnings,bundle.metadata)
