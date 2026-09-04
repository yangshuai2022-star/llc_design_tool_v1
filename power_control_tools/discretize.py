from __future__ import annotations
import math
import numpy as np
from numpy.polynomial import polynomial as P
from scipy import signal
from .models import AnalogTransferFunction, DigitalTransferFunction, DiscretizationMethod


def _trim_leading(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    nz = np.flatnonzero(np.abs(x) > 1e-18)
    return x[nz[0]:] if nz.size else np.array([0.0])


def _mapped_polynomial(coeff_desc: np.ndarray, k: float, common_degree: int) -> np.ndarray:
    """Return q=z^-1 ascending coefficients after s=K(1-q)/(1+q)."""
    coeff_desc = _trim_leading(coeff_desc)
    degree = len(coeff_desc) - 1
    out = np.zeros(common_degree + 1, dtype=float)
    for idx, c in enumerate(coeff_desc):
        pwr = degree - idx
        minus = np.array([math.comb(pwr, j) * ((-1.0) ** j) for j in range(pwr + 1)], dtype=float)
        plus_order = common_degree - pwr
        plus = np.array([math.comb(plus_order, j) for j in range(plus_order + 1)], dtype=float)
        term = P.polymul(minus, plus) * float(c) * (k ** pwr)
        out[:len(term)] += term
    return out


def discretize_transfer_function(
    analog: AnalogTransferFunction,
    sample_rate_hz: float,
    method: DiscretizationMethod | str = DiscretizationMethod.TUSTIN,
    *,
    prewarp_frequency_hz: float | None = None,
) -> DigitalTransferFunction:
    analog.validate()
    fs = float(sample_rate_hz)
    if fs <= 0: raise ValueError("sample_rate_hz must be positive")
    method = DiscretizationMethod(method)
    b, a = analog.arrays()
    if method == DiscretizationMethod.BACKWARD_EULER:
        numd, dend, _ = signal.cont2discrete((b, a), 1.0/fs, method="backward_diff")[:3]
        bd = np.asarray(numd).reshape(-1)
        ad = np.asarray(dend).reshape(-1)
    else:
        if method == DiscretizationMethod.PREWARP_TUSTIN:
            if prewarp_frequency_hz is None or prewarp_frequency_hz <= 0 or prewarp_frequency_hz >= fs/2:
                raise ValueError("prewarp frequency must be within 0..Nyquist")
            wp = 2.0 * math.pi * float(prewarp_frequency_hz)
            k = wp / math.tan(wp/(2.0*fs))
        else:
            k = 2.0 * fs
        common = max(len(_trim_leading(b))-1, len(_trim_leading(a))-1)
        bd = _mapped_polynomial(b, k, common)
        ad = _mapped_polynomial(a, k, common)
    if abs(ad[0]) < 1e-30: raise ValueError("discretization produced a0=0")
    bd = bd/ad[0]; ad = ad/ad[0]
    return DigitalTransferFunction(tuple(float(v) for v in bd), tuple(float(v) for v in ad), fs, analog.name, method.value).normalized()
