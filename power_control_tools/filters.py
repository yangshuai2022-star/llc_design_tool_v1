from __future__ import annotations
import numpy as np
from scipy import signal
from .models import DigitalTransferFunction, FilterDesignResult, FilterResponse, IIRFamily


def _cutoff(response: FilterResponse, f1_hz: float, f2_hz: float | None) -> float | tuple[float,float]:
    if response in (FilterResponse.BANDPASS, FilterResponse.BANDSTOP):
        if f2_hz is None or f2_hz <= f1_hz: raise ValueError("band filter requires f2 > f1")
        return (float(f1_hz), float(f2_hz))
    return float(f1_hz)


def design_iir_filter(
    response: FilterResponse | str,
    family: IIRFamily | str,
    *, sample_rate_hz: float,
    order: int = 2,
    f1_hz: float = 1000.0,
    f2_hz: float | None = None,
    q: float = 10.0,
    passband_ripple_db: float = 1.0,
    stopband_atten_db: float = 40.0,
) -> FilterDesignResult:
    response=FilterResponse(response); family=IIRFamily(family); fs=float(sample_rate_hz)
    if fs <= 0 or order < 1: raise ValueError("invalid sample rate/order")
    if response == FilterResponse.NOTCH:
        if not (0 < f1_hz < fs/2) or q <= 0: raise ValueError("invalid notch frequency/Q")
        b,a=signal.iirnotch(float(f1_hz),float(q),fs=fs)
        return FilterDesignResult(DigitalTransferFunction(tuple(b),tuple(a),fs,"Notch","iirnotch").normalized(),"notch","notch",2,f"Notch {f1_hz:g} Hz, Q={q:g}")
    wn=_cutoff(response,f1_hz,f2_hz)
    vals=(wn,) if isinstance(wn,float) else wn
    if any(v <= 0 or v >= fs/2 for v in vals): raise ValueError("cutoff frequencies must be within 0..Nyquist")
    btype={FilterResponse.LOWPASS:"lowpass",FilterResponse.HIGHPASS:"highpass",FilterResponse.BANDPASS:"bandpass",FilterResponse.BANDSTOP:"bandstop"}[response]
    kwargs=dict(N=int(order),Wn=wn,btype=btype,fs=fs,output="ba")
    if family == IIRFamily.BUTTERWORTH: b,a=signal.butter(**kwargs)
    elif family == IIRFamily.BESSEL: b,a=signal.bessel(norm="phase",**kwargs)
    elif family == IIRFamily.CHEBYSHEV1: b,a=signal.cheby1(rp=float(passband_ripple_db),**kwargs)
    elif family == IIRFamily.CHEBYSHEV2: b,a=signal.cheby2(rs=float(stopband_atten_db),**kwargs)
    elif family == IIRFamily.ELLIPTIC: b,a=signal.ellip(rp=float(passband_ripple_db),rs=float(stopband_atten_db),**kwargs)
    else: raise ValueError(f"unsupported family {family}")
    d=DigitalTransferFunction(tuple(b),tuple(a),fs,f"{family.value} {response.value}",family.value).normalized()
    return FilterDesignResult(d,family.value,response.value,int(order),f"{family.value} order {order} {response.value}")


def design_fir_filter(response: FilterResponse | str, *, sample_rate_hz: float, num_taps: int=31, f1_hz: float=1000.0, f2_hz: float|None=None, window: str="hamming") -> FilterDesignResult:
    response=FilterResponse(response); fs=float(sample_rate_hz)
    if num_taps < 2: raise ValueError("num_taps must be >=2")
    cutoff=_cutoff(response,f1_hz,f2_hz)
    pass_zero={FilterResponse.LOWPASS:"lowpass",FilterResponse.HIGHPASS:"highpass",FilterResponse.BANDPASS:"bandpass",FilterResponse.BANDSTOP:"bandstop"}.get(response)
    if pass_zero is None: raise ValueError("FIR notch should use bandstop")
    b=signal.firwin(int(num_taps),cutoff,window=window,pass_zero=pass_zero,fs=fs)
    a=np.array([1.0])
    return FilterDesignResult(DigitalTransferFunction(tuple(b),tuple(a),fs,f"FIR {response.value}",window),"fir",response.value,num_taps-1,f"Windowed FIR ({window}), {num_taps} taps")


def design_moving_average(*, sample_rate_hz: float, length: int) -> FilterDesignResult:
    if length < 1: raise ValueError("length must be >=1")
    b=np.ones(int(length),dtype=float)/float(length)
    return FilterDesignResult(DigitalTransferFunction(tuple(b),(1.0,),float(sample_rate_hz),"Moving Average","moving_average"),"fir","moving_average",length-1,f"{length}-point moving average")


def design_dc_blocker(*, sample_rate_hz: float, pole_radius: float=0.995) -> FilterDesignResult:
    if not (0.0 < pole_radius < 1.0): raise ValueError("pole radius must be within 0..1")
    b=(1.0,-1.0); a=(1.0,-float(pole_radius))
    return FilterDesignResult(DigitalTransferFunction(b,a,float(sample_rate_hz),"DC Blocker","dc_blocker"),"iir","dc_blocker",1,f"DC blocker r={pole_radius:g}")
