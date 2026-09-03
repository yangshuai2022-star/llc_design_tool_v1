"""V8 adapter for the nonlinear piecewise switched LLC steady-state solver."""

from __future__ import annotations

from dataclasses import dataclass
import math

from scipy.optimize import minimize_scalar, root_scalar

import numpy as np

from .metrics import metrics_from_waveform
from .types import (
    FidelityLevel,
    LLCAnalysisRequest,
    LLCModelResult,
    SolverConvergence,
)
from ..core.operating_point import LLCOperatingPoint, solve_operating_point
from ..core.tank import TankDesign, design_tank
from ..dynamics.plant import (
    DynamicPhasorModel,
    DynamicPhasorSteadyState,
    LLCPlantInputs,
    LLCPlantParameters,
)
from ..dynamics.switched import SwitchedSimulationConfig, simulate_switched_steady_state
from ..dynamics.waveforms import WaveformBundle
from ..models.system import SystemAnalysis


@dataclass(frozen=True)
class TimeDomainConfig:
    """Configuration for the V8 ideal switched reference model."""

    samples_per_cycle: int = 512
    output_cycles: int = 2
    minimum_settling_cycles: int = 12
    maximum_settling_cycles: int = 500
    convergence_tolerance: float = 1e-8
    rectifier_smoothing_current_a: float = 0.02
    shooting_max_evaluations: int = 180
    use_periodic_shooting: bool = True
    series_resistance_ohm: float | None = None
    trim_frequency_to_output: bool = True
    output_voltage_tolerance_v: float = 0.05
    frequency_scan_points: int = 9
    frequency_tolerance_hz: float = 0.2
    strict_convergence: bool = False
    retry_samples_per_cycle: int = 512

    def validate(self) -> None:
        if self.samples_per_cycle < 128:
            raise ValueError("time-domain solver requires at least 128 samples per cycle")
        if self.output_cycles < 1:
            raise ValueError("output_cycles must be >= 1")
        if self.minimum_settling_cycles < 1:
            raise ValueError("minimum_settling_cycles must be >= 1")
        if self.maximum_settling_cycles < self.minimum_settling_cycles:
            raise ValueError("maximum_settling_cycles must exceed minimum_settling_cycles")
        if self.shooting_max_evaluations < 10:
            raise ValueError("shooting_max_evaluations must be >= 10")
        if self.output_voltage_tolerance_v <= 0.0:
            raise ValueError("output_voltage_tolerance_v must be positive")
        if self.frequency_scan_points < 5:
            raise ValueError("frequency_scan_points must be >= 5")
        if self.frequency_tolerance_hz <= 0.0:
            raise ValueError("frequency_tolerance_hz must be positive")
        if self.retry_samples_per_cycle < 128:
            raise ValueError("retry_samples_per_cycle must be >= 128")


@dataclass(frozen=True)
class TimeDomainNativeSolution:
    operating_point: LLCOperatingPoint
    parameters: LLCPlantParameters
    dynamic_phasor_steady_state: DynamicPhasorSteadyState
    waveform: WaveformBundle
    input_phase_deg: float
    series_loss_w: float
    rectifier_drop_loss_w: float


def _fundamental_input_phase(bundle: WaveformBundle) -> float:
    samples_per_cycle = int(bundle.metadata.get("samples_per_cycle", 0))
    if samples_per_cycle <= 0:
        samples_per_cycle = max(1, len(bundle.time_s) // int(bundle.metadata.get("output_cycles", 1)))
    cycles = max(1, len(bundle.time_s) // samples_per_cycle)
    usable = cycles * samples_per_cycle
    bridge = bundle.signal("v_bridge").values[:usable]
    current = bundle.signal("i_resonant").values[:usable]
    vb = np.fft.rfft(bridge) / usable
    ir = np.fft.rfft(current) / usable
    fundamental_index = cycles
    if fundamental_index >= len(vb) or abs(ir[fundamental_index]) < 1e-15:
        return 0.0
    impedance = vb[fundamental_index] / ir[fundamental_index]
    return float(math.degrees(math.atan2(impedance.imag, impedance.real)))


def _simulate(
    model: DynamicPhasorModel,
    steady: DynamicPhasorSteadyState,
    config: TimeDomainConfig,
    samples_per_cycle: int,
) -> WaveformBundle:
    return simulate_switched_steady_state(
        model,
        steady,
        SwitchedSimulationConfig(
            samples_per_cycle=samples_per_cycle,
            output_cycles=config.output_cycles,
            minimum_settling_cycles=config.minimum_settling_cycles,
            maximum_settling_cycles=config.maximum_settling_cycles,
            convergence_tolerance=config.convergence_tolerance,
            rectifier_smoothing_current_a=config.rectifier_smoothing_current_a,
            use_periodic_shooting=config.use_periodic_shooting,
            shooting_max_evaluations=config.shooting_max_evaluations,
        ),
    )


def solve_time_domain(
    request: LLCAnalysisRequest,
    config: TimeDomainConfig | None = None,
    *,
    system_analysis: SystemAnalysis | None = None,
    tank: TankDesign | None = None,
) -> LLCModelResult:
    """Solve the highest-fidelity ideal-switching model currently in V8.1.

    This is the Golden *ideal topology* reference.  Device Coss/Qoss and
    transformer parasitics are intentionally not hidden in this result; those
    belong to the later V8 parasitic-aware layer.
    """

    request.validate()
    cfg = config or TimeDomainConfig(
        samples_per_cycle=max(128, min(request.samples_per_cycle, 512)),
        output_cycles=request.waveform_cycles,
    )
    cfg.validate()
    spec = request.spec
    tank_design = tank or design_tank(spec)
    operating_point = solve_operating_point(
        spec, tank_design, request.bus_voltage_v, request.load_fraction)
    # By default HB and switched-TD use the same explicit Cr-ESR damping.
    # A caller may pass ``system_analysis`` to collapse measured/estimated
    # current-squared losses into the TD model, or explicitly override Rseries.
    if cfg.series_resistance_ohm is not None:
        series_resistance = cfg.series_resistance_ohm
    elif system_analysis is not None:
        series_resistance = None
    else:
        series_resistance = spec.resonant_cap_esr_ohm
    parameters = LLCPlantParameters.from_design(
        spec,
        tank_design,
        operating_point,
        system_analysis,
        series_resistance_ohm=series_resistance,
    )
    model = DynamicPhasorModel(parameters)

    def fixed_frequency_steady(frequency_hz: float) -> DynamicPhasorSteadyState:
        return model.solve_steady_state(
            LLCPlantInputs(frequency_hz, request.bus_voltage_v, 0.0),
            operating_point=operating_point,
        )

    def simulate_steady(
        steady_state: DynamicPhasorSteadyState,
    ) -> WaveformBundle:
        candidate = _simulate(model, steady_state, cfg, cfg.samples_per_cycle)
        if (
            candidate.metadata.get("converged") != "True"
            and cfg.retry_samples_per_cycle != cfg.samples_per_cycle
        ):
            retry = _simulate(
                model, steady_state, cfg, cfg.retry_samples_per_cycle)
            if retry.metadata.get("converged") == "True":
                candidate = retry
        return candidate

    frequency_evaluations = 0
    if request.frequency_hz is not None:
        steady = fixed_frequency_steady(request.frequency_hz)
        bundle = simulate_steady(steady)
        frequency_evaluations = 1
    elif request.regulate_output:
        edf_seed = model.solve_regulated_steady_state(
            bus_voltage_v=request.bus_voltage_v,
            target_output_voltage_v=request.output_voltage_target_v,
            frequency_guess_hz=operating_point.switching_frequency_hz,
            minimum_frequency_hz=spec.minimum_frequency_hz,
            maximum_frequency_hz=spec.maximum_frequency_hz,
            operating_point=operating_point,
        )
        cache: dict[float, tuple[DynamicPhasorSteadyState, WaveformBundle]] = {}

        def evaluate_frequency(
            frequency_hz: float,
        ) -> tuple[DynamicPhasorSteadyState, WaveformBundle]:
            nonlocal frequency_evaluations
            frequency_hz = float(frequency_hz)
            for existing, result in cache.items():
                if abs(existing - frequency_hz) <= max(1e-6, 1e-10 * frequency_hz):
                    return result
            steady_state = (
                edf_seed
                if abs(frequency_hz - edf_seed.inputs.switching_frequency_hz)
                <= max(1e-6, 1e-10 * frequency_hz)
                else fixed_frequency_steady(frequency_hz)
            )
            result = (steady_state, simulate_steady(steady_state))
            cache[frequency_hz] = result
            frequency_evaluations += 1
            return result

        def output_error(frequency_hz: float) -> float:
            _, candidate = evaluate_frequency(frequency_hz)
            return (
                candidate.signal("v_output").statistics.average
                - request.output_voltage_target_v
            )

        seed_frequency = edf_seed.inputs.switching_frequency_hz
        seed_error = output_error(seed_frequency)
        selected_frequency = seed_frequency
        if (
            cfg.trim_frequency_to_output
            and abs(seed_error) > cfg.output_voltage_tolerance_v
        ):
            grid = np.geomspace(
                spec.minimum_frequency_hz,
                spec.maximum_frequency_hz,
                cfg.frequency_scan_points,
            )
            grid = np.unique(np.append(grid, seed_frequency))
            values: list[tuple[float, float]] = []
            for frequency in grid:
                try:
                    values.append((float(frequency), float(output_error(float(frequency)))))
                except (RuntimeError, ValueError):
                    continue
            brackets = [
                (f0, f1)
                for (f0, e0), (f1, e1) in zip(values[:-1], values[1:])
                if e0 * e1 < 0.0
            ]
            if brackets:
                bracket = min(
                    brackets,
                    key=lambda pair: abs(0.5 * (pair[0] + pair[1]) - seed_frequency),
                )
                root = root_scalar(
                    output_error,
                    bracket=bracket,
                    method="brentq",
                    xtol=cfg.frequency_tolerance_hz,
                )
                if root.converged:
                    selected_frequency = float(root.root)
            elif values:
                minimum = min(value[0] for value in values)
                maximum = max(value[0] for value in values)
                optimized = minimize_scalar(
                    lambda frequency: abs(output_error(float(frequency))),
                    bounds=(minimum, maximum),
                    method="bounded",
                    options={"xatol": cfg.frequency_tolerance_hz},
                )
                selected_frequency = float(optimized.x)
        steady, bundle = evaluate_frequency(selected_frequency)
    else:
        steady = fixed_frequency_steady(operating_point.switching_frequency_hz)
        bundle = simulate_steady(steady)
        frequency_evaluations = 1
    converged = bundle.metadata.get("converged") == "True"
    mismatch = float(bundle.metadata.get("periodic_mismatch", math.inf))
    if cfg.strict_convergence and not converged:
        raise RuntimeError(
            "switched periodic solution did not converge: "
            f"mismatch={mismatch:.3e}"
        )

    input_phase = _fundamental_input_phase(bundle)
    ir = bundle.signal("i_resonant").values
    series_loss = parameters.series_resistance_ohm * float(np.mean(ir**2))
    rectifier_drop_loss = (
        spec.rectifier_equivalent_drop_v
        * bundle.signal("i_rectified").statistics.average
    )
    output_average = bundle.signal("v_output").statistics.average
    normalized_gain = (
        spec.turns_ratio
        * (output_average + spec.rectifier_equivalent_drop_v)
        / (spec.bridge_gain * request.bus_voltage_v)
    )
    metrics = metrics_from_waveform(
        request,
        bundle,
        input_phase_deg=input_phase,
        normalized_gain=normalized_gain,
        modeled_series_loss_w=series_loss,
        modeled_rectifier_drop_loss_w=rectifier_drop_loss,
    )
    convergence = SolverConvergence(
        converged=converged,
        residual_norm=mismatch,
        iterations=int(bundle.metadata.get("shooting_evaluations", 0)),
        method="periodic_shooting_rk4",
        message=(
            "periodic shooting converged"
            if converged else "periodic shooting returned a usable but non-converged orbit"
        ),
    )
    native = TimeDomainNativeSolution(
        operating_point=operating_point,
        parameters=parameters,
        dynamic_phasor_steady_state=steady,
        waveform=bundle,
        input_phase_deg=input_phase,
        series_loss_w=series_loss,
        rectifier_drop_loss_w=rectifier_drop_loss,
    )
    warnings = list(bundle.warnings)
    warnings.append(
        "This is the V8.1 ideal switched reference; nonlinear Coss and magnetic parasitics are not included yet."
    )
    return LLCModelResult(
        fidelity=FidelityLevel.SWITCHED_TIME_DOMAIN,
        request=request,
        waveform=bundle,
        metrics=metrics,
        convergence=convergence,
        harmonic_orders=(),
        warnings=tuple(warnings),
        diagnostics={
            "dynamic_phasor_frequency_hz": steady.inputs.switching_frequency_hz,
            "dynamic_phasor_output_voltage_v": steady.output_voltage_v,
            "periodic_mismatch": mismatch,
            "series_resistance_ohm": parameters.series_resistance_ohm,
            "series_loss_w": series_loss,
            "rectifier_drop_loss_w": rectifier_drop_loss,
            "frequency_evaluations": frequency_evaluations,
            "switched_output_error_v": (
                output_average - request.output_voltage_target_v
            ),
        },
        native_result=native,
    )
