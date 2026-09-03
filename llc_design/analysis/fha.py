"""V8 adapter around the established first-harmonic LLC model."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize_scalar, root_scalar

from .metrics import metrics_from_waveform
from .types import (
    FidelityLevel,
    LLCAnalysisRequest,
    LLCModelResult,
    SolverConvergence,
)
from .waveform import build_standard_waveform_bundle
from ..core.operating_point import LLCOperatingPoint, solve_operating_point
from ..core.tank import (
    TankDesign,
    design_tank,
    equivalent_ac_load_ohm,
    gain,
    tank_state,
)
from ..dynamics.waveforms import deadtime_bridge_square


FOUR_OVER_PI = 4.0 / math.pi


def _load_rac(turns_ratio: float, output_voltage_v: float, drop_v: float,
              load_resistance_ohm: float) -> float:
    """Return FHA Rac for a resistive DC load including a constant SR drop."""

    voltage = max(output_voltage_v, 1e-9)
    pout = voltage * voltage / load_resistance_ohm
    transferred = pout * (1.0 + drop_v / voltage)
    return equivalent_ac_load_ohm(
        turns_ratio, voltage + drop_v, max(transferred, 1e-12))


def _fixed_frequency_output(
    request: LLCAnalysisRequest,
    tank: TankDesign,
    frequency_hz: float,
) -> tuple[float, float]:
    """Solve the FHA output voltage at fixed frequency and fixed DC load."""

    spec = request.spec
    level = spec.bridge_gain * request.bus_voltage_v
    rload = request.load_resistance_ohm
    drop = spec.rectifier_equivalent_drop_v
    target = request.output_voltage_target_v
    upper = max(4.0 * target, 2.5 * level / spec.turns_ratio, 10.0)

    def residual(vout: float) -> float:
        rac = _load_rac(spec.turns_ratio, vout, drop, rload)
        predicted = gain(tank, frequency_hz, rac) * level / spec.turns_ratio - drop
        return predicted - vout

    grid = np.linspace(max(1e-3, 0.001 * target), upper, 181)
    values = [residual(float(value)) for value in grid]
    roots: list[float] = []
    for v0, v1, e0, e1 in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
        if abs(e0) < 1e-10:
            roots.append(float(v0))
        if e0 * e1 < 0.0:
            result = root_scalar(residual, bracket=(float(v0), float(v1)), method="brentq")
            if result.converged:
                roots.append(float(result.root))
    if roots:
        vout = min(roots, key=lambda value: abs(value - target))
    else:
        result = minimize_scalar(
            lambda value: abs(residual(float(value))),
            bounds=(float(grid[0]), upper),
            method="bounded",
        )
        vout = float(result.x)
    return vout, _load_rac(spec.turns_ratio, vout, drop, rload)


def _phasor_waveforms(
    request: LLCAnalysisRequest,
    tank: TankDesign,
    frequency_hz: float,
    output_voltage_v: float,
    rac_ohm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct standard FHA resonant-state and clamp waveforms."""

    spec = request.spec
    total_samples = request.waveform_cycles * request.samples_per_cycle
    time_s = np.arange(total_samples, dtype=float) / (
        frequency_hz * request.samples_per_cycle)
    phase = 2.0 * math.pi * frequency_hz * time_s
    level = spec.bridge_gain * request.bus_voltage_v
    bridge = deadtime_bridge_square(
        phase, level, spec.primary_deadtime_s * frequency_hz)

    omega = 2.0 * math.pi * frequency_hz
    state = tank_state(tank, frequency_hz, rac_ohm)
    # Re{X exp(j theta)} convention.  A positive sine is represented by -jA.
    bridge_fundamental_peak = complex(0.0, -FOUR_OVER_PI * level)
    ir_phasor = bridge_fundamental_peak / state.z_input_ohm
    vp_phasor = ir_phasor * state.z_parallel_ohm
    im_phasor = vp_phasor / (1j * omega * tank.lm_h)
    vcr_phasor = ir_phasor / (1j * omega * tank.cr_f)

    carrier = np.exp(1j * phase)
    ir = np.real(ir_phasor * carrier)
    im = np.real(im_phasor * carrier)
    vcr = np.real(vcr_phasor * carrier)
    primary_load = ir - im
    polarity = np.where(primary_load >= 0.0, 1.0, -1.0)
    vp = spec.turns_ratio * (
        output_voltage_v + spec.rectifier_equivalent_drop_v
    ) * polarity
    return time_s, bridge, ir, vcr, im, vp


def solve_fha(request: LLCAnalysisRequest) -> LLCModelResult:
    """Solve one work point with the existing FHA design model."""

    request.validate()
    spec = request.spec
    tank = design_tank(spec)
    op: LLCOperatingPoint | None = None
    if request.regulate_output and request.frequency_hz is None:
        op = solve_operating_point(
            spec, tank, request.bus_voltage_v, request.load_fraction)
        frequency_hz = op.switching_frequency_hz
        output_voltage_v = request.output_voltage_target_v
        rac_ohm = op.rac_ohm
        input_phase_deg = op.input_phase_deg
    else:
        frequency_hz = (
            request.frequency_hz
            if request.frequency_hz is not None
            else spec.resonant_frequency_hz
        )
        output_voltage_v, rac_ohm = _fixed_frequency_output(
            request, tank, frequency_hz)
        input_phase_deg = tank_state(tank, frequency_hz, rac_ohm).input_phase_deg

    time_s, bridge, ir, vcr, im, vp = _phasor_waveforms(
        request, tank, frequency_hz, output_voltage_v, rac_ohm)
    bundle = build_standard_waveform_bundle(
        time_s=time_s,
        switching_frequency_hz=frequency_hz,
        model_name="fha_v8_fast",
        bus_voltage_v=request.bus_voltage_v,
        primary_topology=spec.primary_topology.value,
        primary_deadtime_s=spec.primary_deadtime_s,
        turns_ratio=spec.turns_ratio,
        load_resistance_ohm=request.load_resistance_ohm,
        output_capacitance_f=spec.output_capacitance_f,
        output_cap_esr_ohm=spec.output_cap_esr_ohm,
        series_resistance_ohm=0.0,
        resonant_inductance_h=tank.lr_h,
        output_voltage_mean_v=output_voltage_v,
        bridge_voltage_v=bridge,
        resonant_current_a=ir,
        resonant_capacitor_voltage_v=vcr,
        magnetizing_current_a=im,
        transformer_primary_voltage_v=vp,
        warnings=(
            "FHA retains only the fundamental resonant-state components.",
            "Deadtime is displayed in bridge/gate traces but is not included in the FHA phasor solution.",
        ),
        metadata={
            "rac_ohm": rac_ohm,
            "regulated": str(request.regulate_output and request.frequency_hz is None),
            "harmonic_order": 1,
        },
    )
    normalized_gain = (
        spec.turns_ratio * (output_voltage_v + spec.rectifier_equivalent_drop_v)
        / (spec.bridge_gain * request.bus_voltage_v)
    )
    metrics = metrics_from_waveform(
        request,
        bundle,
        input_phase_deg=input_phase_deg,
        normalized_gain=normalized_gain,
    )
    return LLCModelResult(
        fidelity=FidelityLevel.FHA,
        request=request,
        waveform=bundle,
        metrics=metrics,
        convergence=SolverConvergence(
            converged=True,
            residual_norm=0.0,
            iterations=1,
            method="closed_form_fha",
            message="FHA analytical solution",
        ),
        harmonic_orders=(1,),
        warnings=bundle.warnings,
        diagnostics={
            "rac_ohm": rac_ohm,
            "normalized_frequency": frequency_hz / tank.fr_hz,
        },
        native_result=op,
    )
