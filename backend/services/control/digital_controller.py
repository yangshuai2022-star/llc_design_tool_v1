"""Digital controller implementation and protection supervisor.

The discrete coefficients and the C implementation come from the existing
kernel: ``controller.transfer_function()`` for coefficients, the kernel
``export_controller_c99`` generator for the firmware file, and the kernel
difference-equation formatter for the y[k] expression.

The protection supervisor is an engineering state-machine design template
whose thresholds are derived from the power-stage specification; it is not a
kernel-verified model and is labelled as such.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from llc_design.control.digital_loop import export_controller_c99

from backend.schemas.control_schema import ControllerConfigSchema, PlantContext
from backend.services.control.loop_gain import controller_from_schema
from backend.services.control.plant import build_spec


def digital_controller_response(ctx: PlantContext, controller: ControllerConfigSchema) -> dict[str, Any]:
    config = controller_from_schema(controller, ctx)
    tf = config.transfer_function()
    numerator = [float(v) for v in tf.numerator]
    denominator = [float(v) for v in tf.denominator]
    coefficients = {
        "b0": numerator[0] if len(numerator) > 0 else 0.0,
        "b1": numerator[1] if len(numerator) > 1 else 0.0,
        "b2": numerator[2] if len(numerator) > 2 else 0.0,
        "a1": -float(denominator[1]) if len(denominator) > 1 else 0.0,
        "a2": -float(denominator[2]) if len(denominator) > 2 else 0.0,
    }
    # NOTE: the kernel stores the difference equation as y[k] = -a1*y[k-1] - a2*y[k-2] + ...
    # (its difference_equation() negates the denominator signs internally).
    equation = tf.difference_equation()

    with tempfile.TemporaryDirectory() as directory:
        path = export_controller_c99(
            tf, Path(directory) / "llc_controller.c",
            function_name="llc_voltage_controller_run",
            output_min=controller.output_min,
            output_max=controller.output_max,
        )
        c_code = path.read_text(encoding="utf-8")

    return {
        "kind": controller.kind,
        "coefficients": coefficients,
        "difference_equation": equation,
        "sample_time_s": ctx.sample_time_s,
        "c_code": c_code,
        "warnings": [],
    }


def protection_supervisor(ctx: PlantContext) -> dict[str, Any]:
    """State-machine design template with spec-derived thresholds."""
    spec = build_spec(ctx)
    full_load_a = spec.pout_w / spec.vout_v
    fmin = spec.minimum_frequency_hz
    fmax = spec.maximum_frequency_hz
    thresholds = {
        "ovp_v": round(1.10 * spec.vout_v, 3),
        "uvp_v": round(0.90 * spec.vout_v, 3),
        "ocp_a": round(1.20 * full_load_a, 3),
        "freq_min_hz": fmin,
        "freq_max_hz": fmax,
        "soft_start_start_hz": fmax,
        "soft_start_steps": 16,
        "light_load_fraction": 0.20,
        "burst_fraction": 0.05,
    }
    states = [
        {
            "name": "OFF",
            "description": "Converter disabled; PWM outputs held safe.",
            "actions": ["PWM disabled", "Soft-start timer reset", "Fault latches cleared on enable edge"],
        },
        {
            "name": "SOFT_START",
            "description": "Frequency ramps from fmax (minimum gain) down to the regulated point.",
            "actions": ["Command forced to minimum gain", "Frequency clamp [fmin, fmax] enforced", "OVP/OCP armed after output reaches 90%"],
        },
        {
            "name": "NORMAL",
            "description": "Closed-loop frequency-modulated regulation.",
            "actions": ["Voltage loop active", "Frequency clamp [fmin, fmax]", "OVP/UVP/OCP supervision active"],
        },
        {
            "name": "LIGHT_LOAD",
            "description": "Load below light-load threshold; loop stays active with reduced performance target.",
            "actions": ["Optional frequency fold-back toward fmax", "Loop remains closed"],
        },
        {
            "name": "BURST",
            "description": "Below burst threshold: output regulated by skipping bursts of switching.",
            "actions": ["Hysteretic Vout window control", "Burst ON/OFF timer", "Frequency clamp inside burst"],
        },
        {
            "name": "FAULT_LATCH",
            "description": "OVP / OCP / UVP latch; PWM disabled until re-enable.",
            "actions": ["PWM disabled", "Fault reason recorded", "Recovery only via explicit enable"],
        },
    ]
    transitions = [
        {"source": "OFF", "target": "SOFT_START", "condition": "Enable asserted and no latched fault"},
        {"source": "SOFT_START", "target": "NORMAL", "condition": "Soft-start timer complete and Vout within window"},
        {"source": "NORMAL", "target": "LIGHT_LOAD", "condition": f"Load < {thresholds['light_load_fraction']:.0%}"},
        {"source": "LIGHT_LOAD", "target": "NORMAL", "condition": "Load returns above light-load threshold"},
        {"source": "LIGHT_LOAD", "target": "BURST", "condition": f"Load < {thresholds['burst_fraction']:.0%}"},
        {"source": "BURST", "target": "LIGHT_LOAD", "condition": "Load returns above burst threshold"},
        {"source": "SOFT_START", "target": "FAULT_LATCH", "condition": f"Vout > {thresholds['ovp_v']} V or Iout > {thresholds['ocp_a']} A"},
        {"source": "NORMAL", "target": "FAULT_LATCH", "condition": f"Vout > {thresholds['ovp_v']} V, Vout < {thresholds['uvp_v']} V or Iout > {thresholds['ocp_a']} A"},
        {"source": "LIGHT_LOAD", "target": "FAULT_LATCH", "condition": "Same OVP/OCP/UVP thresholds as NORMAL"},
        {"source": "BURST", "target": "FAULT_LATCH", "condition": "OVP/OCP/UVP trip during burst ON window"},
        {"source": "FAULT_LATCH", "target": "OFF", "condition": "Explicit re-enable edge"},
    ]
    return {
        "thresholds": thresholds,
        "states": states,
        "transitions": transitions,
        "note": (
            "Engineering state-machine design template. Thresholds are derived from the "
            "power-stage specification (OVP 110%, UVP 90%, OCP 120% of full load) and MUST "
            "be re-tuned against the actual hardware, sensing gains and silicon before release. "
            "This block is not a kernel-verified model."
        ),
    }
