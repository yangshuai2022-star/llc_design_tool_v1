"""Loop-gain analysis and Bode data extraction.

Both are thin adapters over the kernel's ``build_digital_loop_analysis``:
the kernel computes the full digital signal chain (power stage x FM gain x
controller x analog sense x ADC sampling x delay envelopes) and the stability
margins.  This module only selects which response to expose and converts
complex arrays into magnitude/phase JSON.
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

from backend.schemas.control_schema import CompensatorRequest, ControllerConfigSchema, PlantContext
from backend.services.control.compensator import design_compensator
from backend.services.control.plant import build_plant

_RESPONSE_NAMES = (
    "power_stage",
    "fm_power_stage",
    "controller",
    "sense_total",
    "delay_nominal",
    "open_loop_nominal",
    "closed_loop_nominal",
    "sensitivity_nominal",
    "closed_loop_output_impedance",
)


def controller_from_schema(controller: ControllerConfigSchema, ctx: PlantContext) -> Any:
    """Translate a schema controller into a kernel controller config."""
    ts = ctx.sample_time_s
    if controller.kind == "pi":
        if controller.kp is None or controller.ti_s is None:
            raise ValueError("pi controller requires kp and ti_s")
        return PIControllerConfig(
            kp=controller.kp, ti_s=controller.ti_s, sample_time_s=ts,
            output_min=controller.output_min, output_max=controller.output_max,
        )
    if controller.kind == "2p2z":
        if None in (controller.b0, controller.b1, controller.b2, controller.a1, controller.a2):
            raise ValueError("2p2z controller requires b0 b1 b2 a1 a2")
        return TwoP2ZControllerConfig(
            b0=controller.b0, b1=controller.b1, b2=controller.b2,
            a1=controller.a1, a2=controller.a2, sample_time_s=ts,
            output_min=controller.output_min, output_max=controller.output_max,
        )
    if controller.kind == "designed":
        if controller.type not in ("pi", "type_ii", "type_iii"):
            raise ValueError("designed controller requires type pi|type_ii|type_iii")
        if controller.target_bandwidth_hz is None or controller.target_phase_margin_deg is None:
            raise ValueError("designed controller requires target_bandwidth_hz and target_phase_margin_deg")
        design = design_compensator(CompensatorRequest(
            plant_context=ctx,
            type=controller.type,
            target_bandwidth_hz=controller.target_bandwidth_hz,
            target_phase_margin_deg=controller.target_phase_margin_deg,
        ))
        coefficients = design["discrete"]
        return TwoP2ZControllerConfig(
            b0=coefficients["b0"], b1=coefficients["b1"], b2=coefficients["b2"],
            a1=coefficients["a1"], a2=coefficients["a2"], sample_time_s=ts,
            output_min=controller.output_min, output_max=controller.output_max,
        )
    raise ValueError(f"unsupported controller kind: {controller.kind}")


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


def build_loop(ctx: PlantContext, controller: ControllerConfigSchema, frequencies_hz: list[float] | None):
    spec, small = build_plant(ctx)
    config = controller_from_schema(controller, ctx)
    loop = build_digital_loop_analysis(
        small,
        controller_config=config,
        frequencies_hz=frequencies_hz,
    )
    return spec, small, config, loop


def loop_gain_response(ctx: PlantContext, controller: ControllerConfigSchema) -> dict[str, Any]:
    _, _, config, loop = build_loop(ctx, controller, None)
    return {
        "margins": _margins_dict(loop.margins_nominal_delay),
        "margins_min_delay": _margins_dict(loop.margins_minimum_delay),
        "margins_max_delay": _margins_dict(loop.margins_maximum_delay),
        "controller_kind": str(type(config).__name__),
        "warnings": list(loop.warnings),
    }


def bode_response(ctx: PlantContext, controller: ControllerConfigSchema) -> dict[str, Any]:
    _, _, _, loop = build_loop(ctx, controller, None)
    frequencies = loop.frequencies_hz
    responses: dict[str, dict[str, list[float]]] = {}
    for name in _RESPONSE_NAMES:
        complex_response = loop.responses.get(name)
        if complex_response is None:
            continue
        complex_response = np.asarray(complex_response, dtype=complex)
        responses[name] = {
            "magnitude_db": [20.0 * math.log10(max(abs(v), 1e-300)) for v in complex_response],
            "phase_deg": [math.degrees(math.atan2(v.imag, v.real)) for v in complex_response],
        }
    margins = _margins_dict(loop.margins_nominal_delay)
    annotations: dict[str, float | None] = {
        "crossover_hz": margins["crossover_hz"],
        "phase_margin_deg": margins["phase_margin_deg"],
        "gain_margin_db": margins["gain_margin_db"],
    }
    return {
        "frequencies_hz": [float(f) for f in frequencies],
        "responses": responses,
        "annotations": annotations,
    }
