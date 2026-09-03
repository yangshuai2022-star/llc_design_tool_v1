"""Self-consistent multi-harmonic harmonic-balance model for LLC converters.

The solver does not linearly superpose independent harmonic FHA circuits.  It
iterates/solves the nonlinear full-wave rectifier clamp together with all
selected harmonic currents.  For each odd harmonic h::

    Ir_h = (Vbridge_h - Vprimary_h) / Zseries(h*ws)
    Im_h = Vprimary_h / Zm(h*ws)
    Iload_h = Ir_h - Im_h

``Vprimary(t)`` is generated from the polarity of the reconstructed load
current, so the harmonic coefficients and rectifier commutation instants are
self-consistent.  The DC output equation enforces average rectified current
balance for the specified resistive load.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares, minimize_scalar, root_scalar

from .metrics import metrics_from_waveform
from .types import (
    FidelityLevel,
    LLCAnalysisRequest,
    LLCModelResult,
    SolverConvergence,
)
from .waveform import build_standard_waveform_bundle
from ..core.operating_point import solve_operating_point
from ..core.tank import GainNotReachableError, TankDesign, design_tank, target_gain
from ..dynamics.waveforms import deadtime_bridge_square


class HarmonicBalanceConvergenceError(RuntimeError):
    """Raised when no physically acceptable harmonic-balance solution exists."""


@dataclass(frozen=True)
class HarmonicBalanceConfig:
    """Numerical and physical options for the V8.1 multi-harmonic solver."""

    max_harmonic: int = 7
    adaptive_harmonics: bool = True
    minimum_harmonic: int = 5
    harmonic_convergence_tolerance: float = 2.5e-4
    samples_per_cycle: int = 2048
    output_cycles: int = 2
    series_resistance_ohm: float | None = None
    magnetizing_series_resistance_ohm: float = 0.0
    rectifier_smoothing_current_a: float = 0.02
    rectifier_smoothing_fraction: float = 1.0e-3
    use_exact_rectifier_projection: bool = True
    allow_regularized_fallback: bool = True
    zero_crossing_samples: int = 2048
    residual_tolerance: float = 2.0e-6
    maximum_function_evaluations: int = 500
    output_voltage_tolerance_v: float = 0.02
    frequency_scan_points: int = 17
    frequency_tolerance_hz: float = 0.05
    include_primary_deadtime: bool = True
    strict_convergence: bool = True

    def validate(self) -> None:
        if self.max_harmonic < 1 or self.max_harmonic % 2 == 0:
            raise ValueError("max_harmonic must be a positive odd integer")
        if self.minimum_harmonic < 1 or self.minimum_harmonic % 2 == 0:
            raise ValueError("minimum_harmonic must be a positive odd integer")
        if self.minimum_harmonic > self.max_harmonic:
            raise ValueError("minimum_harmonic cannot exceed max_harmonic")
        if self.samples_per_cycle < 256:
            raise ValueError("harmonic-balance solver requires at least 256 samples per cycle")
        if self.output_cycles < 1:
            raise ValueError("output_cycles must be >= 1")
        if self.zero_crossing_samples < 256:
            raise ValueError("zero_crossing_samples must be >= 256")
        if self.residual_tolerance <= 0.0:
            raise ValueError("residual_tolerance must be positive")
        if self.maximum_function_evaluations < 20:
            raise ValueError("maximum_function_evaluations must be >= 20")
        if self.frequency_scan_points < 9:
            raise ValueError("frequency_scan_points must be >= 9")
        if self.series_resistance_ohm is not None and self.series_resistance_ohm < 0.0:
            raise ValueError("series resistance cannot be negative")
        if self.magnetizing_series_resistance_ohm < 0.0:
            raise ValueError("magnetizing series resistance cannot be negative")


@dataclass(frozen=True)
class HarmonicConvergenceStep:
    maximum_harmonic: int
    output_voltage_v: float
    resonant_current_rms_a: float
    relative_change: float
    residual_norm: float
    converged: bool


@dataclass(frozen=True)
class HarmonicBalanceNativeSolution:
    """Native phasor solution retained for validation and future SR/magnetics."""

    frequency_hz: float
    output_voltage_v: float
    load_resistance_ohm: float
    harmonics: tuple[int, ...]
    bridge_phasors_v: NDArray[np.complex128]
    primary_voltage_phasors_v: NDArray[np.complex128]
    resonant_current_phasors_a: NDArray[np.complex128]
    magnetizing_current_phasors_a: NDArray[np.complex128]
    primary_load_current_phasors_a: NDArray[np.complex128]
    resonant_capacitor_voltage_phasors_v: NDArray[np.complex128]
    convergence: SolverConvergence
    convergence_history: tuple[HarmonicConvergenceStep, ...]
    input_phase_deg: float
    input_power_w: float
    output_power_w: float
    series_loss_w: float
    magnetizing_loss_w: float
    rectifier_drop_loss_w: float
    power_balance_error_w: float
    rectifier_current_average_a: float
    smoothing_current_a: float
    exact_sign_residual_norm: float
    solver_message: str = ""
    diagnostics: dict[str, float | int | str | bool] = field(default_factory=dict)

    @property
    def resonant_current_rms_a(self) -> float:
        return float(math.sqrt(np.sum(np.abs(self.resonant_current_phasors_a) ** 2) / 2.0))

    @property
    def magnetizing_current_rms_a(self) -> float:
        return float(math.sqrt(np.sum(np.abs(self.magnetizing_current_phasors_a) ** 2) / 2.0))


def odd_harmonics(max_harmonic: int) -> tuple[int, ...]:
    if max_harmonic < 1 or max_harmonic % 2 == 0:
        raise ValueError("maximum harmonic must be a positive odd integer")
    return tuple(range(1, max_harmonic + 1, 2))


def synthesize_real_waveform(
    phasors: NDArray[np.complex128] | Iterable[complex],
    harmonics: Iterable[int],
    phase_rad: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Synthesize ``x(theta)=Re(sum(X_h exp(j*h*theta)))``."""

    h = np.asarray(tuple(harmonics), dtype=float)
    x = np.asarray(tuple(phasors), dtype=np.complex128)
    phase = np.asarray(phase_rad, dtype=float)
    if len(h) != len(x):
        raise ValueError("phasor and harmonic arrays must have equal length")
    if len(h) == 0:
        return np.zeros_like(phase)
    basis = np.exp(1j * np.outer(h, phase))
    return np.real(x[:, None] * basis).sum(axis=0).astype(float)


def extract_peak_phasors(
    waveform: NDArray[np.float64] | Iterable[float],
    harmonics: Iterable[int],
) -> NDArray[np.complex128]:
    """Extract peak complex phasors using the synthesis convention above."""

    values = np.asarray(tuple(waveform) if not isinstance(waveform, np.ndarray) else waveform,
                        dtype=float)
    h = tuple(int(value) for value in harmonics)
    if values.ndim != 1 or len(values) < 8:
        raise ValueError("waveform must be a one-dimensional array with at least eight samples")
    if h and max(h) >= len(values) // 2:
        raise ValueError("waveform sample count is insufficient for requested harmonics")
    spectrum = np.fft.rfft(values) / len(values)
    return np.asarray([2.0 * spectrum[index] for index in h], dtype=np.complex128)


def find_trigonometric_zero_crossings(
    phasors: NDArray[np.complex128] | Iterable[complex],
    harmonics: Iterable[int],
    *,
    search_samples: int = 2048,
) -> tuple[float, ...]:
    """Locate all polarity-changing roots of a periodic harmonic waveform.

    Dense sampling is used only to bracket roots.  Each crossing is then
    polished with Brent's method, so rectifier commutation angles move
    continuously with the harmonic phasors instead of being quantized to FFT
    sample locations.  Tangential zeroes that do not change polarity are
    intentionally ignored.
    """

    h_tuple = tuple(int(value) for value in harmonics)
    x = np.asarray(tuple(phasors), dtype=np.complex128)
    if len(h_tuple) != len(x) or not h_tuple:
        raise ValueError("phasors and harmonics must be non-empty and equal length")
    if search_samples < 256:
        raise ValueError("search_samples must be >= 256")
    h = np.asarray(h_tuple, dtype=float)

    def scalar(theta: float) -> float:
        return float(np.real(np.sum(x * np.exp(1j * h * theta))))

    grid = np.linspace(0.0, 2.0 * math.pi, search_samples + 1)
    values = synthesize_real_waveform(x, h_tuple, grid)
    roots: list[float] = []
    zero_tolerance = 1e-11 * max(float(np.max(np.abs(values))), 1.0)
    # Capture a crossing that falls exactly on a search-grid sample.
    for index in range(search_samples):
        if abs(float(values[index])) > zero_tolerance:
            continue
        before = float(values[index - 1 if index > 0 else search_samples - 1])
        after = float(values[index + 1])
        if before * after < 0.0:
            roots.append(float(grid[index]) % (2.0 * math.pi))
    for index in range(search_samples):
        a = float(grid[index])
        b = float(grid[index + 1])
        fa = float(values[index])
        fb = float(values[index + 1])
        if abs(fa) <= zero_tolerance or abs(fb) <= zero_tolerance:
            continue
        if fa * fb < 0.0:
            root = root_scalar(
                scalar, bracket=(a, b), method="brentq", xtol=1e-13)
            if root.converged:
                roots.append(float(root.root) % (2.0 * math.pi))

    roots.sort()
    unique: list[float] = []
    tolerance = 1e-9
    for root in roots:
        if not unique or abs(root - unique[-1]) > tolerance:
            unique.append(root)
    if len(unique) > 1 and (2.0 * math.pi - unique[-1] + unique[0]) <= tolerance:
        unique.pop()
    if len(unique) < 2:
        raise HarmonicBalanceConvergenceError(
            "rectifier projection did not find two polarity-changing current zeroes"
        )
    return tuple(unique)


def exact_rectifier_projection(
    load_current_phasors_a: NDArray[np.complex128] | Iterable[complex],
    harmonics: Iterable[int],
    *,
    zero_crossing_samples: int = 2048,
) -> tuple[NDArray[np.complex128], float, tuple[float, ...]]:
    """Return exact sign-wave phasors and mean absolute load current.

    The current is a finite trigonometric polynomial.  Once its zero-crossing
    angles are known, both the Fourier coefficients of ``sign(i)`` and
    ``mean(abs(i))`` are integrated analytically over each constant-polarity
    interval.  This removes the regularization bias of a tanh rectifier model
    from the final harmonic-balance solution.
    """

    h_tuple = tuple(int(value) for value in harmonics)
    h = np.asarray(h_tuple, dtype=float)
    phasors = np.asarray(tuple(load_current_phasors_a), dtype=np.complex128)
    roots = find_trigonometric_zero_crossings(
        phasors, h_tuple, search_samples=zero_crossing_samples)
    boundaries = list(roots) + [roots[0] + 2.0 * math.pi]
    polarity_phasors = np.zeros(len(h_tuple), dtype=np.complex128)
    absolute_integral = 0.0

    def current(theta: float) -> float:
        return float(np.real(np.sum(phasors * np.exp(1j * h * theta))))

    def antiderivative(theta: float) -> float:
        # Re{X exp(jhθ)} = Re(X) cos(hθ) - Im(X) sin(hθ).
        return float(np.sum(
            phasors.real / h * np.sin(h * theta)
            + phasors.imag / h * np.cos(h * theta)
        ))

    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        polarity = 1.0 if current(0.5 * (start + stop)) >= 0.0 else -1.0
        polarity_phasors += polarity * (
            np.exp(-1j * h * start) - np.exp(-1j * h * stop)
        ) / (1j * h)
        absolute_integral += polarity * (
            antiderivative(stop) - antiderivative(start)
        )

    polarity_phasors /= math.pi
    mean_absolute_current = absolute_integral / (2.0 * math.pi)
    return (
        np.asarray(polarity_phasors, dtype=np.complex128),
        float(max(mean_absolute_current, 0.0)),
        roots,
    )


class MultiHarmonicLLCSolver:
    """Nonlinear odd-harmonic steady-state solver for one LLC work point."""

    def __init__(
        self,
        request: LLCAnalysisRequest,
        config: HarmonicBalanceConfig | None = None,
        *,
        tank: TankDesign | None = None,
    ):
        request.validate()
        self.request = request
        self.config = config or HarmonicBalanceConfig(
            samples_per_cycle=max(256, request.samples_per_cycle),
            output_cycles=request.waveform_cycles,
        )
        self.config.validate()
        self.tank = tank or design_tank(request.spec)
        self.series_resistance_ohm = (
            request.spec.resonant_cap_esr_ohm
            if self.config.series_resistance_ohm is None
            else float(self.config.series_resistance_ohm)
        )
        self._theta = 2.0 * math.pi * np.arange(
            self.config.samples_per_cycle, dtype=float
        ) / self.config.samples_per_cycle

    def _bridge_waveform(self, frequency_hz: float) -> NDArray[np.float64]:
        spec = self.request.spec
        deadtime_fraction = (
            spec.primary_deadtime_s * frequency_hz
            if self.config.include_primary_deadtime else 0.0
        )
        return deadtime_bridge_square(
            self._theta,
            spec.bridge_gain * self.request.bus_voltage_v,
            deadtime_fraction,
        )

    def _initial_state(
        self,
        frequency_hz: float,
        harmonics: tuple[int, ...],
        bridge_phasors: NDArray[np.complex128],
        previous: HarmonicBalanceNativeSolution | None,
    ) -> NDArray[np.float64]:
        spec = self.request.spec
        count = len(harmonics)
        currents = np.zeros(count, dtype=np.complex128)
        if previous is not None:
            previous_map = {
                harmonic: value
                for harmonic, value in zip(
                    previous.harmonics, previous.primary_load_current_phasors_a)
            }
            for index, harmonic in enumerate(harmonics):
                currents[index] = previous_map.get(harmonic, 0.0j)
            output_guess = previous.output_voltage_v
        else:
            output_guess = self.request.output_voltage_target_v
            rload = self.request.load_resistance_ohm
            drop = spec.rectifier_equivalent_drop_v
            rac = (8.0 / math.pi**2) * spec.turns_ratio**2 * rload * (
                1.0 + drop / max(output_guess, 1e-9)
            )
            omega = 2.0 * math.pi * frequency_hz
            z_series = (
                self.series_resistance_ohm
                + 1j * omega * self.tank.lr_h
                + 1.0 / (1j * omega * self.tank.cr_f)
            )
            z_lm = (
                self.config.magnetizing_series_resistance_ohm
                + 1j * omega * self.tank.lm_h
            )
            z_parallel = 1.0 / (1.0 / rac + 1.0 / z_lm)
            ir = bridge_phasors[0] / (z_series + z_parallel)
            vp = ir * z_parallel
            im = vp / z_lm
            currents[0] = ir - im

        return np.concatenate((currents.real, currents.imag, [output_guess]))

    @staticmethod
    def _unpack_state(
        values: NDArray[np.float64], count: int
    ) -> tuple[NDArray[np.complex128], float]:
        phasors = values[:count] + 1j * values[count:2 * count]
        return np.asarray(phasors, dtype=np.complex128), float(values[-1])

    def _solve_order(
        self,
        frequency_hz: float,
        maximum_harmonic: int,
        previous: HarmonicBalanceNativeSolution | None = None,
    ) -> HarmonicBalanceNativeSolution:
        if frequency_hz <= 0.0:
            raise ValueError("switching frequency must be positive")
        harmonics = odd_harmonics(maximum_harmonic)
        h = np.asarray(harmonics, dtype=float)
        count = len(harmonics)
        spec = self.request.spec
        omega = 2.0 * math.pi * frequency_hz
        basis = np.exp(1j * np.outer(h, self._theta))
        bridge_waveform = self._bridge_waveform(frequency_hz)
        bridge_phasors = extract_peak_phasors(bridge_waveform, harmonics)
        z_series = (
            self.series_resistance_ohm
            + 1j * h * omega * self.tank.lr_h
            + 1.0 / (1j * h * omega * self.tank.cr_f)
        )
        z_magnetizing = (
            self.config.magnetizing_series_resistance_ohm
            + 1j * h * omega * self.tank.lm_h
        )

        expected_primary_peak = (
            math.pi * self.request.requested_output_power_w
            / (
                2.0
                * spec.turns_ratio
                * self.request.output_voltage_target_v
            )
        )
        current_scale = max(expected_primary_peak, 0.25)
        output_current_scale = max(
            self.request.requested_output_power_w
            / self.request.output_voltage_target_v,
            0.25,
        )
        smoothing_current = max(
            self.config.rectifier_smoothing_current_a,
            self.config.rectifier_smoothing_fraction * current_scale,
        )
        rload = self.request.load_resistance_ohm

        def evaluate(
            unknowns: NDArray[np.float64],
            *,
            exact_projection: bool,
        ) -> tuple[
            NDArray[np.float64],
            NDArray[np.complex128],
            NDArray[np.complex128],
            NDArray[np.complex128],
            NDArray[np.complex128],
            NDArray[np.float64],
            float,
            tuple[float, ...],
        ]:
            load_phasors, output_voltage = self._unpack_state(unknowns, count)
            load_current = np.real(load_phasors[:, None] * basis).sum(axis=0)
            if exact_projection:
                polarity_phasors, mean_absolute_current, zero_crossings = (
                    exact_rectifier_projection(
                        load_phasors,
                        harmonics,
                        zero_crossing_samples=self.config.zero_crossing_samples,
                    )
                )
                primary_voltage_phasors = (
                    spec.turns_ratio
                    * (output_voltage + spec.rectifier_equivalent_drop_v)
                    * polarity_phasors
                )
                rectifier_current_average = (
                    spec.turns_ratio * mean_absolute_current
                )
            else:
                polarity = np.tanh(load_current / smoothing_current)
                primary_voltage_waveform = (
                    spec.turns_ratio
                    * (output_voltage + spec.rectifier_equivalent_drop_v)
                    * polarity
                )
                primary_voltage_phasors = extract_peak_phasors(
                    primary_voltage_waveform, harmonics)
                rectifier_current_average = (
                    spec.turns_ratio * float(np.mean(np.abs(load_current)))
                )
                zero_crossings = ()
            resonant_phasors = (
                bridge_phasors - primary_voltage_phasors
            ) / z_series
            magnetizing_phasors = primary_voltage_phasors / z_magnetizing
            predicted_load_phasors = resonant_phasors - magnetizing_phasors
            complex_residual = (
                predicted_load_phasors - load_phasors
            ) / current_scale
            dc_residual = (
                rectifier_current_average - output_voltage / rload
            ) / output_current_scale
            residual = np.concatenate(
                (complex_residual.real, complex_residual.imag, [dc_residual])
            )
            return (
                residual,
                primary_voltage_phasors,
                resonant_phasors,
                magnetizing_phasors,
                predicted_load_phasors,
                load_current,
                rectifier_current_average,
                zero_crossings,
            )

        initial = self._initial_state(
            frequency_hz, harmonics, bridge_phasors, previous)
        maximum_current = max(100.0, 30.0 * current_scale)
        maximum_output_voltage = max(
            4.0 * self.request.output_voltage_target_v,
            3.0 * spec.bridge_gain * self.request.bus_voltage_v / spec.turns_ratio,
            10.0,
        )
        lower = np.concatenate(
            (np.full(2 * count, -maximum_current), [1e-4])
        )
        upper = np.concatenate(
            (np.full(2 * count, maximum_current), [maximum_output_voltage])
        )

        # Smooth projection is only a robust initializer.  The final solve uses
        # analytically integrated zero-crossing intervals and therefore does
        # not retain tanh regularization error in its voltage harmonics.
        seed = least_squares(
            lambda values: evaluate(values, exact_projection=False)[0],
            initial,
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
            loss="linear",
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
            max_nfev=self.config.maximum_function_evaluations,
        )
        seed_residual = evaluate(
            np.asarray(seed.x, dtype=float), exact_projection=False)[0]

        exact_solution = None
        exact_failure_message = ""
        if self.config.use_exact_rectifier_projection:
            try:
                exact_solution = least_squares(
                    lambda values: evaluate(values, exact_projection=True)[0],
                    np.asarray(seed.x, dtype=float),
                    bounds=(lower, upper),
                    method="trf",
                    x_scale="jac",
                    loss="linear",
                    xtol=1e-11,
                    ftol=1e-11,
                    gtol=1e-11,
                    max_nfev=self.config.maximum_function_evaluations,
                )
                exact_trial_residual = evaluate(
                    np.asarray(exact_solution.x, dtype=float),
                    exact_projection=True,
                )[0]
                exact_acceptable = bool(
                    exact_solution.success
                    and np.linalg.norm(exact_trial_residual)
                    <= self.config.residual_tolerance
                )
            except (HarmonicBalanceConvergenceError, ValueError) as exc:
                exact_acceptable = False
                exact_failure_message = str(exc)
        else:
            exact_acceptable = False

        seed_acceptable = bool(
            seed.success
            and np.linalg.norm(seed_residual) <= self.config.residual_tolerance
        )
        if exact_acceptable and exact_solution is not None:
            solution = exact_solution
            final_exact = True
        elif self.config.allow_regularized_fallback and seed_acceptable:
            solution = seed
            final_exact = False
            if not exact_failure_message:
                exact_failure_message = (
                    "exact zero-crossing projection did not converge on this "
                    "rectifier-conduction branch"
                )
        elif exact_solution is not None:
            solution = exact_solution
            final_exact = True
        else:
            solution = seed
            final_exact = False

        (
            residual,
            primary_voltage_phasors,
            resonant_phasors,
            magnetizing_phasors,
            predicted_load_phasors,
            load_current,
            rectifier_current_average,
            zero_crossings,
        ) = evaluate(
            np.asarray(solution.x, dtype=float),
            exact_projection=final_exact,
        )
        if final_exact:
            exact_residual = residual
        else:
            try:
                exact_residual = evaluate(
                    np.asarray(solution.x, dtype=float),
                    exact_projection=True,
                )[0]
            except HarmonicBalanceConvergenceError:
                exact_residual = np.full_like(residual, math.inf)
        load_phasors, output_voltage = self._unpack_state(solution.x, count)
        resonant_capacitor_phasors = resonant_phasors / (
            1j * h * omega * self.tank.cr_f
        )
        resonant_current = np.real(resonant_phasors[:, None] * basis).sum(axis=0)
        magnetizing_current = np.real(magnetizing_phasors[:, None] * basis).sum(axis=0)
        input_power = float(np.mean(bridge_waveform * resonant_current))
        output_power = output_voltage * output_voltage / rload
        series_loss = self.series_resistance_ohm * float(
            np.mean(resonant_current**2)
        )
        magnetizing_loss = self.config.magnetizing_series_resistance_ohm * float(
            np.mean(magnetizing_current**2)
        )
        drop_loss = (
            spec.rectifier_equivalent_drop_v * rectifier_current_average
        )
        balance_error = (
            input_power - output_power - series_loss - magnetizing_loss - drop_loss
        )
        input_impedance = bridge_phasors[0] / resonant_phasors[0]
        input_phase_deg = math.degrees(math.atan2(
            input_impedance.imag, input_impedance.real))
        residual_norm = float(np.linalg.norm(residual))
        exact_residual_norm = float(np.linalg.norm(exact_residual))
        converged = bool(
            solution.success
            and np.isfinite(residual_norm)
            and residual_norm <= self.config.residual_tolerance
            and output_voltage > 0.0
        )
        convergence = SolverConvergence(
            converged=converged,
            residual_norm=residual_norm,
            iterations=int(seed.nfev + (solution.nfev if final_exact else 0)),
            method=(
                "exact_zero_crossing_multi_harmonic_balance"
                if final_exact
                else "regularized_multi_harmonic_balance"
            ),
            message=(
                f"seed: {seed.message}; final: {solution.message}"
                if final_exact
                else str(solution.message)
            ),
        )
        native = HarmonicBalanceNativeSolution(
            frequency_hz=float(frequency_hz),
            output_voltage_v=float(output_voltage),
            load_resistance_ohm=rload,
            harmonics=harmonics,
            bridge_phasors_v=bridge_phasors,
            primary_voltage_phasors_v=primary_voltage_phasors,
            resonant_current_phasors_a=resonant_phasors,
            magnetizing_current_phasors_a=magnetizing_phasors,
            primary_load_current_phasors_a=load_phasors,
            resonant_capacitor_voltage_phasors_v=resonant_capacitor_phasors,
            convergence=convergence,
            convergence_history=(),
            input_phase_deg=float(input_phase_deg),
            input_power_w=input_power,
            output_power_w=float(output_power),
            series_loss_w=float(series_loss),
            magnetizing_loss_w=float(magnetizing_loss),
            rectifier_drop_loss_w=float(drop_loss),
            power_balance_error_w=float(balance_error),
            rectifier_current_average_a=float(rectifier_current_average),
            smoothing_current_a=float(smoothing_current),
            exact_sign_residual_norm=exact_residual_norm,
            solver_message=(
                f"seed: {seed.message}; final: {solution.message}"
                if final_exact
                else str(solution.message)
            ),
            diagnostics={
                "cost": float(solution.cost),
                "seed_cost": float(seed.cost),
                "seed_residual_norm": float(np.linalg.norm(seed_residual)),
                "exact_rectifier_projection": final_exact,
                "regularized_fallback": not final_exact,
                "exact_projection_message": exact_failure_message,
                "zero_crossing_count": len(zero_crossings),
                "first_zero_crossing_deg": (
                    math.degrees(zero_crossings[0]) if zero_crossings else math.nan
                ),
                "optimality": float(solution.optimality),
                "active_mask_count": int(np.count_nonzero(solution.active_mask)),
                "predicted_load_mismatch": float(
                    np.linalg.norm(predicted_load_phasors - load_phasors)
                ),
            },
        )
        if not converged and self.config.strict_convergence:
            raise HarmonicBalanceConvergenceError(
                f"HB H{maximum_harmonic} failed at {frequency_hz/1e3:.6f} kHz: "
                f"residual={residual_norm:.3e}; {solution.message}"
            )
        return native

    @staticmethod
    def _relative_change(
        current: HarmonicBalanceNativeSolution,
        previous: HarmonicBalanceNativeSolution,
    ) -> float:
        voltage_change = abs(
            current.output_voltage_v - previous.output_voltage_v
        ) / max(abs(current.output_voltage_v), 1e-12)
        current_change = abs(
            current.resonant_current_rms_a - previous.resonant_current_rms_a
        ) / max(abs(current.resonant_current_rms_a), 1e-12)
        return float(max(voltage_change, current_change))

    def solve_at_frequency(
        self,
        frequency_hz: float,
        *,
        previous: HarmonicBalanceNativeSolution | None = None,
        force_maximum_order: bool = False,
    ) -> HarmonicBalanceNativeSolution:
        """Solve one fixed switching frequency with harmonic continuation."""

        if force_maximum_order or not self.config.adaptive_harmonics:
            orders = (self.config.max_harmonic,)
        else:
            orders = odd_harmonics(self.config.max_harmonic)
        history: list[HarmonicConvergenceStep] = []
        current_previous = previous
        final: HarmonicBalanceNativeSolution | None = None
        for maximum_harmonic in orders:
            current = self._solve_order(
                frequency_hz, maximum_harmonic, current_previous)
            relative_change = (
                math.inf if current_previous is None
                else self._relative_change(current, current_previous)
            )
            history.append(HarmonicConvergenceStep(
                maximum_harmonic=maximum_harmonic,
                output_voltage_v=current.output_voltage_v,
                resonant_current_rms_a=current.resonant_current_rms_a,
                relative_change=float(relative_change),
                residual_norm=current.convergence.residual_norm,
                converged=current.convergence.converged,
            ))
            final = current
            if (
                self.config.adaptive_harmonics
                and maximum_harmonic >= self.config.minimum_harmonic
                and relative_change <= self.config.harmonic_convergence_tolerance
            ):
                break
            current_previous = current
        assert final is not None
        return HarmonicBalanceNativeSolution(
            **{
                **final.__dict__,
                "convergence_history": tuple(history),
            }
        )

    def solve_regulated_frequency(self) -> HarmonicBalanceNativeSolution:
        """Find the HB switching frequency that regulates the requested output."""

        spec = self.request.spec
        target_output = self.request.output_voltage_target_v
        try:
            fha = solve_operating_point(
                spec, self.tank, self.request.bus_voltage_v,
                self.request.load_fraction)
            frequency_guess = fha.switching_frequency_hz
        except (GainNotReachableError, ValueError):
            frequency_guess = spec.resonant_frequency_hz

        cache: dict[float, HarmonicBalanceNativeSolution] = {}

        def nearest_previous(frequency_hz: float) -> HarmonicBalanceNativeSolution | None:
            if not cache:
                return None
            return min(cache.items(), key=lambda item: abs(item[0] - frequency_hz))[1]

        def solve_frequency(frequency_hz: float) -> HarmonicBalanceNativeSolution:
            key = float(frequency_hz)
            for existing, result in cache.items():
                if abs(existing - key) <= max(1e-6, 1e-10 * key):
                    return result
            result = self.solve_at_frequency(
                key,
                previous=nearest_previous(key),
                force_maximum_order=True,
            )
            cache[key] = result
            return result

        def error(frequency_hz: float) -> float:
            return solve_frequency(float(frequency_hz)).output_voltage_v - target_output

        frequencies = np.geomspace(
            spec.minimum_frequency_hz,
            spec.maximum_frequency_hz,
            self.config.frequency_scan_points,
        )
        frequencies = np.unique(np.append(frequencies, frequency_guess))
        frequencies.sort()
        values: list[tuple[float, float]] = []
        for frequency in frequencies:
            try:
                values.append((float(frequency), float(error(float(frequency)))))
            except HarmonicBalanceConvergenceError:
                continue
        if not values:
            raise HarmonicBalanceConvergenceError(
                "HB regulated-frequency solve failed at every sampled frequency"
            )

        direct = min(values, key=lambda item: abs(item[1]))
        roots: list[float] = []
        if abs(direct[1]) <= self.config.output_voltage_tolerance_v:
            roots.append(direct[0])
        for (f0, e0), (f1, e1) in zip(values[:-1], values[1:]):
            if e0 * e1 < 0.0:
                try:
                    root = root_scalar(
                        error,
                        bracket=(f0, f1),
                        method="brentq",
                        xtol=self.config.frequency_tolerance_hz,
                    )
                except HarmonicBalanceConvergenceError:
                    continue
                if root.converged:
                    roots.append(float(root.root))

        if roots:
            required = target_gain(spec, self.request.bus_voltage_v)
            if required <= 1.0:
                preferred = [
                    frequency for frequency in roots
                    if frequency >= self.tank.fr_hz * (1.0 - 1e-8)
                ]
            else:
                preferred = [
                    frequency for frequency in roots
                    if frequency <= self.tank.fr_hz * (1.0 + 1e-8)
                ]
            candidates = preferred or roots
            selected_frequency = min(
                candidates, key=lambda value: abs(value - frequency_guess))
        else:
            result = minimize_scalar(
                lambda frequency: abs(error(float(frequency))),
                bounds=(spec.minimum_frequency_hz, spec.maximum_frequency_hz),
                method="bounded",
                options={"xatol": self.config.frequency_tolerance_hz},
            )
            selected_frequency = float(result.x)
            best = solve_frequency(selected_frequency)
            if abs(best.output_voltage_v - target_output) > self.config.output_voltage_tolerance_v:
                available = [solution.output_voltage_v for solution in cache.values()]
                raise GainNotReachableError(
                    "HB cannot regulate the requested output inside the frequency range: "
                    f"best Vo={best.output_voltage_v:.5f} V at "
                    f"{selected_frequency/1e3:.5f} kHz; sampled range "
                    f"{min(available):.4f}..{max(available):.4f} V"
                )

        coarse = solve_frequency(selected_frequency)
        # Re-run harmonic continuation at the final frequency so the returned
        # history records the actual order-convergence evidence.
        final = self.solve_at_frequency(
            selected_frequency,
            previous=coarse,
            force_maximum_order=False,
        )
        return HarmonicBalanceNativeSolution(
            **{
                **final.__dict__,
                "diagnostics": {
                    **final.diagnostics,
                    "frequency_evaluations": len(cache),
                    "frequency_guess_hz": float(frequency_guess),
                    "regulated_output_error_v": float(
                        final.output_voltage_v - target_output),
                },
            }
        )

    def build_waveform(
        self,
        solution: HarmonicBalanceNativeSolution,
    ):
        spec = self.request.spec
        samples_per_cycle = self.config.samples_per_cycle
        total_samples = self.config.output_cycles * samples_per_cycle
        time_s = np.arange(total_samples, dtype=float) / (
            solution.frequency_hz * samples_per_cycle)
        phase = 2.0 * math.pi * solution.frequency_hz * time_s
        bridge = deadtime_bridge_square(
            phase,
            spec.bridge_gain * self.request.bus_voltage_v,
            (
                spec.primary_deadtime_s * solution.frequency_hz
                if self.config.include_primary_deadtime else 0.0
            ),
        )
        ir = synthesize_real_waveform(
            solution.resonant_current_phasors_a, solution.harmonics, phase)
        im = synthesize_real_waveform(
            solution.magnetizing_current_phasors_a, solution.harmonics, phase)
        vcr = synthesize_real_waveform(
            solution.resonant_capacitor_voltage_phasors_v,
            solution.harmonics,
            phase,
        )
        primary_load = ir - im
        polarity = np.where(primary_load >= 0.0, 1.0, -1.0)
        vp = spec.turns_ratio * (
            solution.output_voltage_v + spec.rectifier_equivalent_drop_v
        ) * polarity
        history_text = ",".join(
            f"H{item.maximum_harmonic}:{item.relative_change:.3e}"
            for item in solution.convergence_history
        )
        model_warnings = [
            "Multi-harmonic HB uses an ideal transformer and ideal full-bridge rectifier clamp.",
            "Output-capacitor ripple is reconstructed after the HB DC balance; it is not a retained harmonic state.",
            "Nonlinear MOSFET Coss, winding capacitance and leakage parasitics are reserved for the V8 parasitic layer.",
        ]
        if bool(solution.diagnostics.get("regularized_fallback", False)):
            model_warnings.append(
                "Exact zero-crossing HB did not converge on this conduction branch; "
                "the returned solution uses the regularized rectifier projection "
                "and must be checked against the switched reference."
            )
        return build_standard_waveform_bundle(
            time_s=time_s,
            switching_frequency_hz=solution.frequency_hz,
            model_name="multi_harmonic_hb_v8",
            bus_voltage_v=self.request.bus_voltage_v,
            primary_topology=spec.primary_topology.value,
            primary_deadtime_s=spec.primary_deadtime_s,
            turns_ratio=spec.turns_ratio,
            load_resistance_ohm=self.request.load_resistance_ohm,
            output_capacitance_f=spec.output_capacitance_f,
            output_cap_esr_ohm=spec.output_cap_esr_ohm,
            series_resistance_ohm=self.series_resistance_ohm,
            resonant_inductance_h=self.tank.lr_h,
            output_voltage_mean_v=solution.output_voltage_v,
            bridge_voltage_v=bridge,
            resonant_current_a=ir,
            resonant_capacitor_voltage_v=vcr,
            magnetizing_current_a=im,
            transformer_primary_voltage_v=vp,
            warnings=tuple(model_warnings),
            metadata={
                "maximum_harmonic": solution.harmonics[-1],
                "harmonics": ",".join(str(value) for value in solution.harmonics),
                "residual_norm": solution.convergence.residual_norm,
                "exact_sign_residual_norm": solution.exact_sign_residual_norm,
                "power_balance_error_w": solution.power_balance_error_w,
                "smoothing_current_a": solution.smoothing_current_a,
                "harmonic_convergence": history_text,
                "regulated": str(
                    self.request.regulate_output
                    and self.request.frequency_hz is None
                ),
            },
        )

    def solve(self) -> LLCModelResult:
        if self.request.frequency_hz is not None:
            native = self.solve_at_frequency(self.request.frequency_hz)
        elif self.request.regulate_output:
            native = self.solve_regulated_frequency()
        else:
            native = self.solve_at_frequency(
                self.request.spec.resonant_frequency_hz)
        waveform = self.build_waveform(native)
        normalized_gain = (
            self.request.spec.turns_ratio
            * (
                native.output_voltage_v
                + self.request.spec.rectifier_equivalent_drop_v
            )
            / (
                self.request.spec.bridge_gain
                * self.request.bus_voltage_v
            )
        )
        metrics = metrics_from_waveform(
            self.request,
            waveform,
            input_phase_deg=native.input_phase_deg,
            normalized_gain=normalized_gain,
            input_power_w=native.input_power_w,
            modeled_series_loss_w=(
                native.series_loss_w + native.magnetizing_loss_w
            ),
            modeled_rectifier_drop_loss_w=native.rectifier_drop_loss_w,
        )
        return LLCModelResult(
            fidelity=FidelityLevel.HARMONIC_BALANCE,
            request=self.request,
            waveform=waveform,
            metrics=metrics,
            convergence=native.convergence,
            harmonic_orders=native.harmonics,
            warnings=waveform.warnings,
            diagnostics={
                **native.diagnostics,
                "exact_sign_residual_norm": native.exact_sign_residual_norm,
                "power_balance_error_w": native.power_balance_error_w,
                "input_power_w": native.input_power_w,
                "series_loss_w": native.series_loss_w,
                "rectifier_drop_loss_w": native.rectifier_drop_loss_w,
            },
            native_result=native,
        )


def solve_harmonic_balance(
    request: LLCAnalysisRequest,
    config: HarmonicBalanceConfig | None = None,
) -> LLCModelResult:
    """Convenience entry point for one V8.1 multi-harmonic work point."""

    return MultiHarmonicLLCSolver(request, config).solve()
