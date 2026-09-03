"""V8 adapter for the exact rectifier-complementarity time-domain model."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .metrics import metrics_from_waveform
from .types import FidelityLevel, LLCAnalysisRequest, LLCModelResult, SolverConvergence
from ..core.operating_point import solve_operating_point
from ..core.tank import TankDesign, design_tank
from ..dynamics.plant import DynamicPhasorModel, LLCPlantInputs, LLCPlantParameters
from ..dynamics.complementarity import ComplementaritySimulationConfig, simulate_complementarity_steady_state


@dataclass(frozen=True)
class ComplementarityTimeDomainConfig:
    samples_per_cycle: int = 1024
    output_cycles: int = 2
    minimum_settling_cycles: int = 20
    maximum_settling_cycles: int = 400
    convergence_tolerance: float = 2e-7
    series_resistance_ohm: float | None = None


def solve_complementarity_time_domain(request: LLCAnalysisRequest, config: ComplementarityTimeDomainConfig|None=None, *, tank: TankDesign|None=None) -> LLCModelResult:
    request.validate(); cfg=config or ComplementarityTimeDomainConfig()
    spec=request.spec; td=tank or design_tank(spec)
    op=solve_operating_point(spec,td,request.bus_voltage_v,request.load_fraction)
    series=spec.resonant_cap_esr_ohm if cfg.series_resistance_ohm is None else cfg.series_resistance_ohm
    p=LLCPlantParameters.from_design(spec,td,op,None,series_resistance_ohm=series)
    model=DynamicPhasorModel(p)
    fs=float(request.frequency_hz or op.switching_frequency_hz)
    steady=model.solve_steady_state(LLCPlantInputs(fs,request.bus_voltage_v,0.0),operating_point=op)
    bundle=simulate_complementarity_steady_state(model,steady,ComplementaritySimulationConfig(
        samples_per_cycle=cfg.samples_per_cycle,output_cycles=cfg.output_cycles,
        minimum_settling_cycles=cfg.minimum_settling_cycles,maximum_settling_cycles=cfg.maximum_settling_cycles,
        convergence_tolerance=cfg.convergence_tolerance,
    ))
    bridge=bundle.signal('v_bridge').values; ir=bundle.signal('i_resonant').values
    n=cfg.samples_per_cycle; cycles=max(1,len(ir)//n); usable=cycles*n
    vb=np.fft.rfft(bridge[:usable])/usable; ii=np.fft.rfft(ir[:usable])/usable; k=cycles
    phase=0.0
    if k<len(vb) and abs(ii[k])>1e-15:
        z=vb[k]/ii[k]; phase=math.degrees(math.atan2(z.imag,z.real))
    series_loss=series*float(np.mean(ir**2))
    rect_loss=spec.rectifier_equivalent_drop_v*bundle.signal('i_rectified').statistics.average
    vavg=bundle.signal('v_output').statistics.average
    gain=spec.turns_ratio*(vavg+spec.rectifier_equivalent_drop_v)/(spec.bridge_gain*request.bus_voltage_v)
    metrics=metrics_from_waveform(request,bundle,input_phase_deg=phase,normalized_gain=gain,modeled_series_loss_w=series_loss,modeled_rectifier_drop_loss_w=rect_loss)
    mismatch=float(bundle.metadata.get('periodic_mismatch',math.inf)); converged=bundle.metadata.get('converged')=='True'
    open_fraction=float(np.mean(np.abs(bundle.signal('rectifier_state').values)<0.5))
    return LLCModelResult(
        fidelity=FidelityLevel.SWITCHED_TIME_DOMAIN,request=request,waveform=bundle,metrics=metrics,
        convergence=SolverConvergence(converged,mismatch,int(bundle.metadata.get('settled_cycles',0)),'hybrid_rectifier_complementarity','hybrid ON+/OPEN/ON- rectifier state solution'),
        warnings=bundle.warnings,diagnostics={'rectifier_model':'complementarity','dcm_open_fraction':open_fraction,'series_loss_w':series_loss,'rectifier_drop_loss_w':rect_loss},native_result=None,
    )
