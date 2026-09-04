from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from scipy import signal
from .models import AnalogTransferFunction, DigitalTransferFunction, StabilityClass


@dataclass(frozen=True)
class ControlResponseAnalysis:
    frequency_hz: NDArray[np.float64]
    magnitude_db: NDArray[np.float64]
    phase_deg: NDArray[np.float64]
    analog_magnitude_db: NDArray[np.float64] | None
    analog_phase_deg: NDArray[np.float64] | None
    group_delay_s: NDArray[np.float64]
    impulse_time_s: NDArray[np.float64]
    impulse: NDArray[np.float64]
    step_time_s: NDArray[np.float64]
    step: NDArray[np.float64]
    poles: NDArray[np.complex128]
    zeros: NDArray[np.complex128]
    stable: bool
    stability_class: StabilityClass
    max_pole_radius: float
    dc_gain: float
    step_final_value: float
    step_peak: float
    step_overshoot_percent: float
    settling_time_s: float


def _settling_time(t: np.ndarray, y: np.ndarray, final: float, tol: float=0.02) -> float:
    scale=max(abs(final),1e-12); bad=np.flatnonzero(np.abs(y-final)>tol*scale)
    if bad.size == 0: return 0.0
    idx=int(bad[-1]+1)
    return float(t[idx]) if idx < len(t) else float(t[-1])


def analyze_digital_filter(digital: DigitalTransferFunction, *, analog: AnalogTransferFunction|None=None, points: int=1200, response_samples: int=256) -> ControlResponseAnalysis:
    d=digital.normalized(); fs=d.sample_rate_hz
    fmin=max(1e-3,fs*1e-5); fmax=fs*0.499
    f=np.geomspace(fmin,fmax,int(points)); w=2*np.pi*f/fs
    _,h=signal.freqz(np.asarray(d.b),np.asarray(d.a),worN=w)
    mag=20*np.log10(np.maximum(np.abs(h),1e-300)); phase=np.unwrap(np.angle(h))*180/np.pi
    amag=aph=None
    if analog is not None:
        b,a=analog.arrays(); _,ha=signal.freqs(b,a,worN=2*np.pi*f)
        amag=20*np.log10(np.maximum(np.abs(ha),1e-300)); aph=np.unwrap(np.angle(ha))*180/np.pi
    # group_delay returns samples; avoid exact singular points by using same grid
    with np.errstate(all='ignore'):
        _,gd=signal.group_delay((np.asarray(d.b),np.asarray(d.a)),w=w,fs=2*np.pi)
    gd=np.nan_to_num(gd,nan=0.0,posinf=0.0,neginf=0.0)/fs
    system=(np.asarray(d.b),np.asarray(d.a),1.0/fs)
    ti,yi=signal.dimpulse(system,n=int(response_samples)); ts,ys=signal.dstep(system,n=int(response_samples))
    ti=np.asarray(ti,dtype=float).reshape(-1); imp=np.asarray(yi[0],dtype=float).reshape(-1)
    ts=np.asarray(ts,dtype=float).reshape(-1); step=np.asarray(ys[0],dtype=float).reshape(-1)
    final=float(step[-1]); peak=float(np.max(step)) if final>=0 else float(np.min(step)); over=max(0.0,(peak-final)/max(abs(final),1e-12)*100.0) if final>=0 else max(0.0,(final-peak)/max(abs(final),1e-12)*100.0)
    dc=float(np.sum(d.b)/np.sum(d.a)) if abs(np.sum(d.a))>1e-15 else float('inf')
    return ControlResponseAnalysis(f,mag,phase,amag,aph,gd,ti,imp,ts,step,d.poles,d.zeros,d.stable,d.stability_class,d.max_pole_radius,dc,final,peak,float(over),_settling_time(ts,step,final))
