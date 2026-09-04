from __future__ import annotations
import math
import numpy as np
from .models import AnalogTransferFunction, ControllerKind


def _w(f_hz: float) -> float:
    if f_hz <= 0:
        raise ValueError("frequency must be positive")
    return 2.0 * math.pi * float(f_hz)


def _poly_from_real_roots_hz(freqs_hz: list[float] | tuple[float, ...]) -> np.ndarray:
    roots = [-_w(f) for f in freqs_hz]
    return np.poly(roots).astype(float)


def design_controller(kind: ControllerKind | str, **p: float) -> AnalogTransferFunction:
    kind = ControllerKind(kind)
    k = float(p.get("gain", 1.0))
    if kind == ControllerKind.INTEGRATOR:
        return AnalogTransferFunction((k,), (1.0, 0.0), "Integrator")
    if kind == ControllerKind.PI:
        kp = float(p.get("kp", k)); ki = float(p.get("ki", 1.0))
        return AnalogTransferFunction((kp, ki), (1.0, 0.0), "PI")
    if kind == ControllerKind.MODIFIED_PI:
        # H(s)=K(1+sTz)/(sTz(1+sTp)).  fp may be given directly.
        tz = float(p.get("tz_s", 1.0 / _w(float(p.get("fz_hz", 100.0)))))
        tp = float(p.get("tp_s", 1.0 / _w(float(p.get("fp_hz", 10_000.0)))))
        if tz <= 0 or tp <= 0: raise ValueError("time constants must be positive")
        # K(1+sTz)/(s*Tz*(1+sTp))
        return AnalogTransferFunction((k * tz, k), (tz * tp, tz, 0.0), "Modified PI")
    if kind in (ControllerKind.LEAD, ControllerKind.LAG, ControllerKind.ONE_P_ONE_Z):
        fz = float(p.get("fz_hz", 1_000.0)); fp = float(p.get("fp_hz", 10_000.0))
        wz, wp = _w(fz), _w(fp)
        # K*(1+s/wz)/(1+s/wp) = K*wp/wz*(s+wz)/(s+wp)
        num = k * wp / wz * np.array([1.0, wz])
        den = np.array([1.0, wp])
        name = {ControllerKind.LEAD:"Lead", ControllerKind.LAG:"Lag", ControllerKind.ONE_P_ONE_Z:"1P1Z"}[kind]
        return AnalogTransferFunction(tuple(num), tuple(den), name)
    if kind in (ControllerKind.TWO_P_TWO_Z, ControllerKind.THREE_P_THREE_Z):
        count = 2 if kind == ControllerKind.TWO_P_TWO_Z else 3
        zeros = [float(p.get(f"fz{i+1}_hz", 100.0 * (10**i))) for i in range(count)]
        poles = [float(p.get(f"fp{i+1}_hz", 1_000.0 * (10**i))) for i in range(count)]
        num = _poly_from_real_roots_hz(zeros)
        den = _poly_from_real_roots_hz(poles)
        # normalize transfer to requested DC gain when finite
        dc0 = num[-1] / den[-1]
        num = num * (k / dc0)
        return AnalogTransferFunction(tuple(num), tuple(den), "2P2Z" if count == 2 else "3P3Z")
    if kind == ControllerKind.GENERAL:
        num = p.get("numerator")
        den = p.get("denominator")
        if num is None or den is None:
            raise ValueError("GENERAL requires numerator and denominator")
        return AnalogTransferFunction(tuple(float(v) for v in num), tuple(float(v) for v in den), "General")
    raise ValueError(f"unsupported controller type {kind}")
