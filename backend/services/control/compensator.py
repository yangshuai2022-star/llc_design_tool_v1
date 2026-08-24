"""Compensation designer for the LLC frequency-modulated voltage loop.

The design loop is driven entirely by the kernel's own stability margins:
candidate zero/pole placement is adjusted until ``build_digital_loop_analysis``
reports the target crossover and phase margin.  No hand-rolled delay or plant
model is used anywhere - the kernel is the single source of truth.

Controller families map to the kernel's existing discrete forms:
  - pi       -> PIControllerConfig      (kernel firmware-exact trapezoidal PI)
  - type_ii  -> 2P2Z: zero + integrator + one HF pole
  - type_iii -> 2P2Z: two zeros + integrator + one HF pole (max phase boost)
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from llc_design.control.digital_loop import (
    PIControllerConfig,
    TwoP2ZControllerConfig,
    build_digital_loop_analysis,
)

from backend.schemas.control_schema import CompensatorRequest, PlantContext
from backend.services.control.plant import build_plant, fm_gain_hz_per_pu

_MAX_ITERATIONS = 16


def _margins_dict(margins: Any) -> dict[str, Any]:
    crossover = getattr(margins, "critical_gain_crossover_hz", None)
    pm = getattr(margins, "phase_margin_deg", None)
    gm = getattr(margins, "gain_margin_db", None)
    dm = getattr(margins, "delay_margin_s", None)
    return {
        "crossover_hz": None if crossover is None or not math.isfinite(crossover) else float(crossover),
        "phase_margin_deg": None if pm is None or not math.isfinite(pm) else float(pm),
        "gain_margin_db": None if gm is None or not math.isfinite(gm) else float(gm),
        "delay_margin_s": None if dm is None or not math.isfinite(dm) else float(dm),
        "stable": bool(pm is not None and math.isfinite(pm) and pm > 0.0),
    }


def _build_config(ctype: str, kc: float, fz1: float, fz2: float, fp: float, ts: float) -> Any:
    if ctype == "pi":
        if fz1 <= 0.0:
            raise ValueError("PI zero frequency must be positive")
        return PIControllerConfig(kp=kc, ti_s=1.0 / (2.0 * math.pi * fz1), sample_time_s=ts)
    nyquist = 0.5 / ts
    if ctype == "type_ii":
        fz_far = min(fp * 50.0, 0.49 * nyquist)
        zeros = [fz1, fz_far]
        w_far = 2.0 * math.pi * fz_far
    elif ctype == "type_iii":
        zeros = [fz1, fz2]
        w_far = 2.0 * math.pi * fz2
    else:
        raise ValueError(f"unsupported compensator type: {ctype}")
    poles = [0.0, fp]
    # The kernel's gain multiplies the UN-normalized polynomial
    # (s+wz1)(s+wz2)/(s(s+wp)); convert the normalized loop-shaping gain
    # C(s) = Kc*(1+s/wz1)(1+s/wz2)/(s(1+s/wp)) into that convention:
    #   kernel_gain = Kc * wp / (wz1 * wz2)
    kernel_gain = kc * (2.0 * math.pi * fp) / ((2.0 * math.pi * fz1) * w_far)
    return TwoP2ZControllerConfig.from_analog_poles_zeros(
        gain=kernel_gain, zeros_hz=zeros, poles_hz=poles, sample_time_s=ts,
    )


def _analog_parameters(ctype: str, kc: float, fz1: float, fz2: float, fp: float) -> dict[str, Any]:
    """Report the continuous (analog) form the digital controller realizes."""
    if ctype == "pi":
        ki = kc * 2.0 * math.pi * fz1
        return {"Kp": kc, "Ki": ki, "zeros_hz": [fz1], "poles_hz": [0.0]}
    if ctype == "type_ii":
        return {"Kc": kc, "Ki": kc * 2.0 * math.pi * fz1, "zeros_hz": [fz1], "poles_hz": [0.0, fp]}
    return {"Kc": kc, "Ki": kc * 2.0 * math.pi * fz1, "zeros_hz": [fz1, fz2], "poles_hz": [0.0, fp]}


def design_compensator(request: CompensatorRequest) -> dict[str, Any]:
    ctx = request.plant_context
    spec, small = build_plant(ctx)
    ts = ctx.sample_time_s
    fsw = small.operating_point.switching_frequency_hz
    nyquist = 0.5 / ts

    fc_target = float(request.target_bandwidth_hz)
    fc_target = min(fc_target, 0.30 * min(fsw, nyquist))
    pm_target = float(request.target_phase_margin_deg)

    # Initial placement: zeros ~ fc/4 (type_ii/iii), HF pole ~ 6*fc (clamped).
    fz1 = fc_target / 4.0
    fz2 = fc_target / 4.0
    fp = min(fc_target * 6.0, 0.45 * nyquist, 0.30 * fsw)
    if fp <= fz1:
        fp = max(2.0 * fz1, 0.20 * nyquist)

    # Analytic starting gain so the first kernel evaluation is already close.
    response = small.continuous_transfer.frequency_response(np.asarray([fc_target]))
    kfm = fm_gain_hz_per_pu(small)
    gtot = kfm * complex(response[0])
    s = 2j * math.pi * fc_target
    wz1 = 2.0 * math.pi * fz1
    if request.type == "pi":
        n0 = (1.0 + s / wz1) / s
    elif request.type == "type_ii":
        wfar = 2.0 * math.pi * min(fp * 50.0, 0.49 * nyquist)
        wp = 2.0 * math.pi * fp
        n0 = (1.0 + s / wz1) * (1.0 + s / wfar) / (s * (1.0 + s / wp))
    else:
        wz2 = 2.0 * math.pi * fz2
        wp = 2.0 * math.pi * fp
        n0 = (1.0 + s / wz1) * (1.0 + s / wz2) / (s * (1.0 + s / wp))
    magnitude = abs(gtot * n0)
    kc = 1.0 / magnitude if magnitude > 0.0 else 1.0

    best: dict[str, Any] | None = None
    for iteration in range(_MAX_ITERATIONS):
        config = _build_config(request.type, kc, fz1, fz2, fp, ts)
        loop = build_digital_loop_analysis(small, controller_config=config)
        margins = loop.margins_nominal_delay
        measured = _margins_dict(margins)
        fc_meas = measured["crossover_hz"]
        pm_meas = measured["phase_margin_deg"]
        if fc_meas is None or pm_meas is None:
            kc *= 0.5
            continue
        error_fc = abs(math.log(fc_meas / fc_target)) if fc_meas > 0.0 else 9.0
        error_pm = abs(pm_meas - pm_target)
        score = error_fc + 0.01 * error_pm
        if best is None or score < best["score"]:
            best = {
                "score": score, "kc": kc, "fz1": fz1, "fz2": fz2, "fp": fp,
                "measured": measured, "iteration": iteration,
            }
        if error_fc < 0.30 and error_pm <= 5.0:
            break
        # Adjust zero placement for phase margin.
        if pm_meas < pm_target - 2.0:
            fz1 = max(fz1 / 1.4, fc_target / 60.0)
            fz2 = max(fz2 / 1.4, fc_target / 60.0)
        elif pm_meas > pm_target + 8.0:
            fz1 = min(fz1 * 1.35, fc_target / 1.2)
            fz2 = min(fz2 * 1.35, fc_target / 1.2)
        # Adjust gain for crossover (integrator-dominated: fc ~ K).
        if fc_meas > 0.0:
            kc *= min(max(fc_target / fc_meas, 0.25), 4.0)
        if fp <= 0.0:
            fp = 0.25 * nyquist

    if best is None:
        raise ValueError("compensator design failed to converge")
    kc, fz1, fz2, fp = best["kc"], best["fz1"], best["fz2"], best["fp"]
    config = _build_config(request.type, kc, fz1, fz2, fp, ts)
    loop = build_digital_loop_analysis(small, controller_config=config)
    achieved = _margins_dict(loop.margins_nominal_delay)

    discrete = config.transfer_function()
    return {
        "type": request.type,
        "continuous": _analog_parameters(request.type, kc, fz1, fz2, fp),
        "discrete": {
            "b0": float(discrete.numerator[0]),
            "b1": float(discrete.numerator[1]) if len(discrete.numerator) > 1 else 0.0,
            "b2": float(discrete.numerator[2]) if len(discrete.numerator) > 2 else 0.0,
            "a1": float(discrete.denominator[1]) if len(discrete.denominator) > 1 else 0.0,
            "a2": float(discrete.denominator[2]) if len(discrete.denominator) > 2 else 0.0,
        },
        "sample_time_s": ts,
        "achieved": achieved,
        "iterations": iteration + 1,
        "c_code": None,
        "warnings": list(loop.warnings),
    }
