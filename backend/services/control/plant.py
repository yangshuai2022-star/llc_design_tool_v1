"""Control Loop Designer - LLC plant service.

Thin adapter over the existing llc_design small-signal kernel.  The kernel's
dynamic-phasor model is linearized by ``build_small_signal_analysis``; this
module only translates web payloads into kernel calls and back.  No control
math is re-implemented here.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from llc_design.control.analysis import build_small_signal_analysis
from llc_design.control.digital_loop import (
    FrequencyModulatorLUT,
    evaluate_fm_operating_point,
)
from llc_design.core.spec import LLCDesignSpec, PrimaryTopology

from backend.schemas.control_schema import ManualPlantParams, PlantContext

# Reuse the exact parameter validation the LLC design page uses so the control
# plant and the power-stage design always agree on spec interpretation.
from webapp.service import spec_from_payload  # noqa: E402


def _complex_pairs(values: Any) -> list[tuple[float, float]]:
    return [(float(v.real), float(v.imag)) for v in values]


def _spec_from_manual(params: ManualPlantParams) -> LLCDesignSpec:
    """Back-compute fr/Ln/Q so that design_tank reproduces the given tank.

    design_tank is closed-form:
        rac = (8/pi^2) * n^2 * Vout^2 / Pout
        zr  = Q * rac ;  Lr = zr / (2*pi*fr) ;  Cr = 1 / (2*pi*fr*zr)
    Solving for the spec fields that yield the requested Lr/Cr/Lm is exact.
    """
    lr, cr, lm = params.lr_h, params.cr_f, params.lm_h
    if lr <= 0.0 or cr <= 0.0 or lm <= 0.0:
        raise ValueError("Lr, Cr and Lm must be positive")
    if lm <= lr:
        raise ValueError("Lm must exceed Lr (Ln > 1)")
    fr = 1.0 / (2.0 * math.pi * math.sqrt(lr * cr))
    ln = lm / lr
    rac = (8.0 / math.pi**2) * params.turns_ratio**2 * params.vout_v**2 / params.pout_w
    zr = math.sqrt(lr / cr)
    q = zr / rac

    secondary_turns = params.secondary_turns or 4
    primary_turns = params.primary_turns or max(1, int(round(params.turns_ratio * secondary_turns)))

    spec = LLCDesignSpec().clone(
        vbus_nom_v=params.vbus_v,
        vbus_min_normal_v=0.85 * params.vbus_v,
        vbus_max_v=params.vbus_v,
        vbus_hold_end_v=0.70 * params.vbus_v,
        vout_v=params.vout_v,
        pout_w=params.pout_w,
        primary_topology=PrimaryTopology(str(params.primary_topology)),
        resonant_frequency_hz=fr,
        ln_ratio=ln,
        q_full_load=q,
        primary_turns=primary_turns,
        secondary_turns=secondary_turns,
    )
    spec.validate()
    return spec


def build_spec(ctx: PlantContext) -> LLCDesignSpec:
    if ctx.plant is not None:
        return _spec_from_manual(ctx.plant)
    return spec_from_payload(dict(ctx.spec or {}))


def build_plant(ctx: PlantContext) -> tuple[LLCDesignSpec, Any]:
    """Build the linearized small-signal LLC plant at one work point."""
    spec = build_spec(ctx)
    vbus = spec.vbus_nom_v if ctx.vbus_v is None else float(ctx.vbus_v)
    small = build_small_signal_analysis(
        spec,
        vbus_v=vbus,
        load_fraction=ctx.load_fraction,
        sample_time_s=ctx.sample_time_s,
    )
    return spec, small


def fm_gain_hz_per_pu(small: Any) -> float:
    """Frequency-modulator local gain (Hz per pu command) at the work point."""
    fsw = small.operating_point.switching_frequency_hz
    lut = FrequencyModulatorLUT.firmware_default()
    return float(evaluate_fm_operating_point(lut, switching_frequency_hz=fsw).gain_hz_per_pu)


def to_plant_response(ctx: PlantContext, small: Any) -> dict[str, Any]:
    continuous = small.continuous_transfer
    discrete = small.discrete_plant
    op = small.operating_point
    return {
        "mode": "manual" if ctx.plant is not None else "design",
        "control_input": continuous.input_name,
        "control_output": continuous.output_name,
        "model_name": small.continuous_plant.model_name,
        "operating_point": {
            "label": f"{op.vbus_v:.0f}V_{op.load_fraction * 100:.0f}pct",
            "vbus_v": op.vbus_v,
            "load_fraction": op.load_fraction,
            "pout_w": op.pout_w,
            "switching_frequency_hz": op.switching_frequency_hz,
            "required_gain": op.required_gain,
            "achieved_gain": op.achieved_gain,
            "input_phase_deg": op.input_phase_deg,
        },
        "continuous": {
            "input_name": continuous.input_name,
            "input_unit": continuous.input_unit,
            "output_name": continuous.output_name,
            "output_unit": continuous.output_unit,
            "dc_gain": float(continuous.dc_gain),
            "numerator": [float(v) for v in continuous.numerator],
            "denominator": [float(v) for v in continuous.denominator],
            "poles": _complex_pairs(continuous.poles),
            "zeros": _complex_pairs(continuous.zeros),
        },
        "discrete_numerator": [float(v) for v in discrete.numerator],
        "discrete_denominator": [float(v) for v in discrete.denominator],
        "sample_time_s": ctx.sample_time_s,
        "fm_gain_hz_per_pu": fm_gain_hz_per_pu(small),
        "warnings": plant_warnings(small),
    }


def plant_response(ctx: PlantContext) -> dict[str, Any]:
    _, small = build_plant(ctx)
    return to_plant_response(ctx, small)


def plant_warnings(small: Any) -> list[str]:
    warnings: list[str] = []
    if not small.continuous_plant.stable:
        warnings.append("Linearized plant has right-half-plane dynamics; control design must compensate.")
    return warnings
