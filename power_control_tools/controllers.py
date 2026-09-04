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


def _series_tf(num1, den1, num2, den2):
    return np.polymul(np.asarray(num1, dtype=float), np.asarray(num2, dtype=float)), np.polymul(np.asarray(den1, dtype=float), np.asarray(den2, dtype=float))


def _type2_from_pz(fp0_hz: float, fz_hz: float, fp_hz: float) -> AnalogTransferFunction:
    """Canonical inverting Type-II compensator.

    G(s) = -wp0 * (1+s/wz) / [s * (1+s/wp)]
    The minus sign matches the common inverting error-amplifier network.  In a
    closed-loop block diagram the feedback sign can be accounted for outside
    this compensator; the displayed transfer remains faithful to the analog
    circuit convention.
    """
    wp0, wz, wp = _w(fp0_hz), _w(fz_hz), _w(fp_hz)
    # -wp0*wp/wz * (s+wz) / [s(s+wp)]
    return AnalogTransferFunction(tuple((-wp0 * wp / wz) * np.asarray([1.0, wz])), (1.0, wp, 0.0), "Type-II")


def _type3_from_pz(fp0_hz: float, fz1_hz: float, fz2_hz: float, fp1_hz: float, fp2_hz: float) -> AnalogTransferFunction:
    """Canonical inverting Type-III compensator.

    G(s) = -wp0 (1+s/wz1)(1+s/wz2) /
           [s (1+s/wp1)(1+s/wp2)]
    """
    wp0, wz1, wz2, wp1, wp2 = map(_w, (fp0_hz, fz1_hz, fz2_hz, fp1_hz, fp2_hz))
    num = -wp0 * wp1 * wp2 / (wz1 * wz2) * np.poly([-wz1, -wz2])
    den = np.polymul([1.0, 0.0], np.poly([-wp1, -wp2]))
    return AnalogTransferFunction(tuple(num), tuple(den), "Type-III")


def _type2_from_rc(r1_ohm: float, r2_ohm: float, c1_f: float, c2_f: float) -> AnalogTransferFunction:
    """Common op-amp Type-II network from R1/R2/C1/C2.

    Feedback network: C2 || (R2 + C1 series), input resistor R1.
      wp0 = 1/[R1(C1+C2)]
      wz  = 1/(R2 C1)
      wp  = (C1+C2)/(R2 C1 C2)
    """
    if min(r1_ohm, r2_ohm, c1_f, c2_f) <= 0:
        raise ValueError("Type-II R/C values must be positive")
    wp0 = 1.0 / (r1_ohm * (c1_f + c2_f))
    wz = 1.0 / (r2_ohm * c1_f)
    wp = (c1_f + c2_f) / (r2_ohm * c1_f * c2_f)
    return _type2_from_pz(wp0/(2*math.pi), wz/(2*math.pi), wp/(2*math.pi))


def _type3_from_rc(r1_ohm: float, r2_ohm: float, r3_ohm: float, c1_f: float, c2_f: float, c3_f: float) -> AnalogTransferFunction:
    """Common op-amp Type-III network from R1/R2/R3/C1/C2/C3.

    The feedback branch is C2 || (R2 + C1 series); the input branch is
    R1 || (R3 + C3 series). This gives:
      wp0 = 1/[R1(C1+C2)]
      wz1 = 1/(R2 C1)
      wz2 = 1/[C3(R1+R3)]
      wp1 = (C1+C2)/(R2 C1 C2)
      wp2 = 1/(R3 C3)
    """
    if min(r1_ohm, r2_ohm, r3_ohm, c1_f, c2_f, c3_f) <= 0:
        raise ValueError("Type-III R/C values must be positive")
    wp0 = 1.0 / (r1_ohm * (c1_f + c2_f))
    wz1 = 1.0 / (r2_ohm * c1_f)
    wz2 = 1.0 / (c3_f * (r1_ohm + r3_ohm))
    wp1 = (c1_f + c2_f) / (r2_ohm * c1_f * c2_f)
    wp2 = 1.0 / (r3_ohm * c3_f)
    return _type3_from_pz(wp0/(2*math.pi), wz1/(2*math.pi), wz2/(2*math.pi), wp1/(2*math.pi), wp2/(2*math.pi))


def design_controller(kind: ControllerKind | str, **p: float) -> AnalogTransferFunction:
    kind = ControllerKind(kind)
    k = float(p.get("gain", 1.0))
    if kind == ControllerKind.INTEGRATOR:
        return AnalogTransferFunction((k,), (1.0, 0.0), "Integrator")
    if kind == ControllerKind.PI:
        # User convention: Kp * (1 + 1/(Ti*s))
        kp = float(p.get("kp", k))
        ti = float(p["ti_s"]) if "ti_s" in p else (kp / float(p["ki"]) if "ki" in p and float(p["ki"]) != 0.0 else 0.01)
        if ti <= 0: raise ValueError("Ti must be positive")
        return AnalogTransferFunction((kp, kp/ti), (1.0, 0.0), "PI")
    if kind == ControllerKind.PIF:
        # PI followed by a first-order low-pass pole.
        kp = float(p.get("kp", k)); ti = float(p.get("ti_s", 0.01)); wp = _w(float(p.get("lpf_pole_hz", p.get("fp_hz", 10_000.0))))
        if ti <= 0: raise ValueError("Ti must be positive")
        num, den = _series_tf((kp, kp/ti), (1.0, 0.0), (wp,), (1.0, wp))
        return AnalogTransferFunction(tuple(num), tuple(den), "PIF")
    if kind == ControllerKind.PID:
        # Kp * (1 + 1/(Ti*s) + Td*s)
        kp = float(p.get("kp", k)); ti = float(p.get("ti_s", 0.01)); td = float(p.get("td_s", 1e-4))
        if ti <= 0 or td < 0: raise ValueError("Ti must be >0 and Td must be >=0")
        return AnalogTransferFunction((kp*td, kp, kp/ti), (1.0, 0.0), "PID")
    if kind == ControllerKind.PIDF:
        # Project convention: ideal PID followed by a first-order LPF.
        kp = float(p.get("kp", k)); ti = float(p.get("ti_s", 0.01)); td = float(p.get("td_s", 1e-4)); wp = _w(float(p.get("lpf_pole_hz", p.get("fp_hz", 10_000.0))))
        if ti <= 0 or td < 0: raise ValueError("Ti must be >0 and Td must be >=0")
        num, den = _series_tf((kp*td, kp, kp/ti), (1.0, 0.0), (wp,), (1.0, wp))
        return AnalogTransferFunction(tuple(num), tuple(den), "PIDF")
    if kind == ControllerKind.TYPE_II:
        if str(p.get("type_input_mode", "pz")).lower() == "rc":
            return _type2_from_rc(float(p["r1_ohm"]), float(p["r2_ohm"]), float(p["c1_f"]), float(p["c2_f"]))
        return _type2_from_pz(float(p.get("fp0_hz", 100.0)), float(p.get("fz1_hz", p.get("fz_hz", 500.0))), float(p.get("fp1_hz", p.get("fp_hz", 10_000.0))))
    if kind == ControllerKind.TYPE_III:
        if str(p.get("type_input_mode", "pz")).lower() == "rc":
            return _type3_from_rc(float(p["r1_ohm"]), float(p["r2_ohm"]), float(p["r3_ohm"]), float(p["c1_f"]), float(p["c2_f"]), float(p["c3_f"]))
        return _type3_from_pz(float(p.get("fp0_hz", 100.0)), float(p.get("fz1_hz", 500.0)), float(p.get("fz2_hz", 1_000.0)), float(p.get("fp1_hz", 10_000.0)), float(p.get("fp2_hz", 20_000.0)))
    if kind == ControllerKind.MODIFIED_PI:
        tz = float(p.get("tz_s", 1.0 / _w(float(p.get("fz_hz", 100.0)))))
        tp = float(p.get("tp_s", 1.0 / _w(float(p.get("fp_hz", 10_000.0)))))
        if tz <= 0 or tp <= 0: raise ValueError("time constants must be positive")
        return AnalogTransferFunction((k * tz, k), (tz * tp, tz, 0.0), "Modified PI")
    if kind in (ControllerKind.LEAD, ControllerKind.LAG, ControllerKind.ONE_P_ONE_Z):
        fz = float(p.get("fz_hz", 1_000.0)); fp = float(p.get("fp_hz", 10_000.0))
        wz, wp = _w(fz), _w(fp)
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
        dc0 = num[-1] / den[-1]
        num = num * (k / dc0)
        return AnalogTransferFunction(tuple(num), tuple(den), "2P2Z" if count == 2 else "3P3Z")
    if kind == ControllerKind.GENERAL:
        num = p.get("numerator"); den = p.get("denominator")
        if num is None or den is None: raise ValueError("GENERAL requires numerator and denominator")
        return AnalogTransferFunction(tuple(float(v) for v in num), tuple(float(v) for v in den), "General")
    raise ValueError(f"unsupported controller type {kind}")
