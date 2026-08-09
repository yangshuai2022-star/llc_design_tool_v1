"""Nonlinear dynamic-phasor/EDF model of an LLC power stage.

The model retains the sine and cosine components of resonant current,
resonant-capacitor voltage and magnetizing current, plus the output-capacitor
DC state.  The ideal full-wave rectifier is represented by its exact
fundamental describing function for a sinusoidal transformer load current.

State convention
----------------
For every resonant quantity ``q``::

    q(theta, t) = q_c(t) cos(theta) + q_s(t) sin(theta)

where ``d theta / dt = 2*pi*f_s``.  The seven states are::

    [i_r,c, i_r,s, v_Cr,c, v_Cr,s, i_m,c, i_m,s, v_Co]

The model is a design-oriented implementation of the extended describing
function/dynamic-phasor approach used for resonant-converter small-signal
models.  It is intentionally kept independent from any specific controller.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Mapping

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar, root, root_scalar

from ..core.operating_point import LLCOperatingPoint
from ..core.spec import LLCDesignSpec
from ..core.tank import TankDesign

if TYPE_CHECKING:  # pragma: no cover - import only for static typing
    from ..models.system import SystemAnalysis


PI = math.pi
FOUR_OVER_PI = 4.0 / PI
TWO_OVER_PI = 2.0 / PI
SQRT2 = math.sqrt(2.0)


class PlantModelError(RuntimeError):
    """Raised when a nonlinear plant operating point cannot be solved."""


@dataclass(frozen=True)
class LLCPlantInputs:
    """Slow control/disturbance inputs of the nonlinear averaged plant."""

    switching_frequency_hz: float
    bus_voltage_v: float
    load_current_disturbance_a: float = 0.0

    def as_array(self) -> NDArray[np.float64]:
        return np.asarray(
            [self.switching_frequency_hz, self.bus_voltage_v,
             self.load_current_disturbance_a], dtype=float)

    @classmethod
    def from_array(cls, values: NDArray[np.float64]) -> "LLCPlantInputs":
        if len(values) != 3:
            raise ValueError("plant input vector must contain [frequency, bus voltage, load current]")
        return cls(float(values[0]), float(values[1]), float(values[2]))


@dataclass(frozen=True)
class LLCPlantParameters:
    """Electrical parameters used by dynamic-phasor and switched models."""

    lr_h: float
    cr_f: float
    lm_h: float
    output_capacitance_f: float
    output_cap_esr_ohm: float
    load_resistance_ohm: float
    turns_ratio: float
    bridge_gain: float
    series_resistance_ohm: float
    magnetizing_series_resistance_ohm: float = 0.0
    rectifier_equivalent_drop_v: float = 0.0
    primary_deadtime_s: float = 0.0
    primary_topology: str = "FULL_BRIDGE"
    primary_turns: int = 0
    secondary_turns: int = 0
    transformer_core_area_m2: float = 0.0
    transformer_magnetic_path_m: float = 0.0
    resonant_inductor_turns: int = 0
    resonant_inductor_core_area_m2: float = 0.0
    resonant_inductor_magnetic_path_m: float = 0.0

    def validate(self) -> None:
        positive = {
            "Lr": self.lr_h,
            "Cr": self.cr_f,
            "Lm": self.lm_h,
            "Co": self.output_capacitance_f,
            "Rload": self.load_resistance_ohm,
            "turns ratio": self.turns_ratio,
            "bridge gain": self.bridge_gain,
        }
        bad = [name for name, value in positive.items() if value <= 0.0]
        if bad:
            raise ValueError("plant parameters must be positive: " + ", ".join(bad))
        if self.output_cap_esr_ohm < 0.0:
            raise ValueError("output capacitor ESR cannot be negative")
        if self.series_resistance_ohm < 0.0:
            raise ValueError("series resistance cannot be negative")

    @classmethod
    def from_design(
        cls,
        spec: LLCDesignSpec,
        tank: TankDesign,
        operating_point: LLCOperatingPoint,
        analysis: "SystemAnalysis | None" = None,
        *,
        series_resistance_ohm: float | None = None,
    ) -> "LLCPlantParameters":
        """Create plant parameters from the design and one solved work point.

        If a complete :class:`SystemAnalysis` is supplied, conductive losses at
        the selected work point are collapsed into an equivalent primary-side
        damping resistance.  This is deliberately limited to losses that scale
        approximately with current squared; switching and fixed auxiliary
        losses are not folded into the dynamic tank.
        """
        if operating_point.pout_w <= 0.0:
            raise ValueError("operating point output power must be positive")
        r_load = spec.vout_v**2 / operating_point.pout_w
        r_series = (
            estimate_equivalent_series_resistance(analysis, operating_point)
            if series_resistance_ohm is None
            else float(series_resistance_ohm)
        )
        # Without a full loss analysis, retain explicit Cr ESR and a modest
        # reference allowance for bridge/winding/connection damping.
        if analysis is None and series_resistance_ohm is None:
            r_series = spec.resonant_cap_esr_ohm + 0.10
        params = cls(
            lr_h=tank.lr_h,
            cr_f=tank.cr_f,
            lm_h=tank.lm_h,
            output_capacitance_f=spec.output_capacitance_f,
            output_cap_esr_ohm=spec.output_cap_esr_ohm,
            load_resistance_ohm=r_load,
            turns_ratio=spec.turns_ratio,
            bridge_gain=spec.bridge_gain,
            series_resistance_ohm=max(r_series, 1e-6),
            rectifier_equivalent_drop_v=spec.rectifier_equivalent_drop_v,
            primary_deadtime_s=spec.primary_deadtime_s,
            primary_topology=spec.primary_topology.value,
            primary_turns=spec.primary_turns,
            secondary_turns=spec.secondary_turns,
            transformer_core_area_m2=(
                analysis.transformer.core.ae_m2 if analysis is not None else 0.0
            ),
            transformer_magnetic_path_m=(
                analysis.transformer.core.le_m if analysis is not None else 0.0
            ),
            resonant_inductor_turns=(
                analysis.resonant_inductor.turns if analysis is not None else 0
            ),
            resonant_inductor_core_area_m2=(
                analysis.resonant_inductor.core.ae_m2 if analysis is not None else 0.0
            ),
            resonant_inductor_magnetic_path_m=(
                analysis.resonant_inductor.core.le_m if analysis is not None else 0.0
            ),
        )
        params.validate()
        return params


@dataclass(frozen=True)
class DynamicPhasorSteadyState:
    """Solved dynamic-phasor operating point."""

    states: NDArray[np.float64]
    inputs: LLCPlantInputs
    residual_norm: float
    iterations: int
    converged: bool
    output_voltage_v: float
    output_capacitor_voltage_v: float
    rectifier_current_avg_a: float
    resonant_current_peak_a: float
    resonant_current_rms_a: float
    magnetizing_current_peak_a: float
    magnetizing_current_rms_a: float
    primary_load_current_peak_a: float
    secondary_current_rms_a: float
    target_output_voltage_v: float | None = None
    output_voltage_error_v: float = 0.0
    frequency_trimmed: bool = False

    @property
    def state_map(self) -> Mapping[str, float]:
        names = (
            "ir_cos_a", "ir_sin_a", "vcr_cos_v", "vcr_sin_v",
            "im_cos_a", "im_sin_a", "vco_v",
        )
        return {name: float(value) for name, value in zip(names, self.states)}


class DynamicPhasorModel:
    """Seven-state nonlinear LLC dynamic-phasor model."""

    state_names = (
        "ir_cos_a",
        "ir_sin_a",
        "vcr_cos_v",
        "vcr_sin_v",
        "im_cos_a",
        "im_sin_a",
        "vco_v",
    )
    input_names = (
        "switching_frequency_hz",
        "bus_voltage_v",
        "load_current_disturbance_a",
    )
    output_names = (
        "output_voltage_v",
        "resonant_current_rms_a",
        "magnetizing_current_rms_a",
        "secondary_current_rms_a",
        "rectifier_current_avg_a",
        "output_capacitor_voltage_v",
    )

    def __init__(self, parameters: LLCPlantParameters):
        parameters.validate()
        self.p = parameters

    @staticmethod
    def _amplitude(cosine: float, sine: float, epsilon: float = 1e-12) -> float:
        return math.sqrt(cosine * cosine + sine * sine + epsilon)

    def algebraic(self, states: NDArray[np.float64], inputs: LLCPlantInputs) -> dict[str, float]:
        """Return algebraic rectifier and output quantities."""
        ir_c, ir_s, _, _, im_c, im_s, vco = map(float, states)
        load_c = ir_c - im_c
        load_s = ir_s - im_s
        load_peak = self._amplitude(load_c, load_s)
        i_rect_avg = TWO_OVER_PI * self.p.turns_ratio * load_peak

        rload = self.p.load_resistance_ohm
        esr = self.p.output_cap_esr_ohm
        i_dist = inputs.load_current_disturbance_a
        # Co ESR is treated as a series element between the internal capacitor
        # voltage and the output terminal.  Solve the algebraic terminal voltage
        # exactly for a resistive base load plus an injected load disturbance.
        vout = (
            rload * vco + rload * esr * (i_rect_avg - i_dist)
        ) / (rload + esr)

        clamp_secondary_v = max(vout + self.p.rectifier_equivalent_drop_v, 0.0)
        sign_fundamental_gain = FOUR_OVER_PI / load_peak
        vp_c = self.p.turns_ratio * clamp_secondary_v * sign_fundamental_gain * load_c
        vp_s = self.p.turns_ratio * clamp_secondary_v * sign_fundamental_gain * load_s
        return {
            "primary_load_cos_a": load_c,
            "primary_load_sin_a": load_s,
            "primary_load_peak_a": load_peak,
            "rectifier_current_avg_a": i_rect_avg,
            "output_voltage_v": vout,
            "vp_cos_v": vp_c,
            "vp_sin_v": vp_s,
        }

    def rhs(
        self,
        states: NDArray[np.float64],
        inputs: LLCPlantInputs | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evaluate the nonlinear slow-time state derivative."""
        if not isinstance(inputs, LLCPlantInputs):
            inputs = LLCPlantInputs.from_array(np.asarray(inputs, dtype=float))
        if inputs.switching_frequency_hz <= 0.0 or inputs.bus_voltage_v <= 0.0:
            raise ValueError("frequency and bus voltage must be positive")

        ir_c, ir_s, vcr_c, vcr_s, im_c, im_s, vco = map(float, states)
        alg = self.algebraic(states, inputs)
        omega = 2.0 * PI * inputs.switching_frequency_hz
        # Square-wave bridge is referenced to sin(theta).  Its fundamental
        # peak amplitude is 4/pi times the physical square-wave level.
        vb_c = 0.0
        vb_s = FOUR_OVER_PI * self.p.bridge_gain * inputs.bus_voltage_v
        vp_c = alg["vp_cos_v"]
        vp_s = alg["vp_sin_v"]

        f_ir_c = (vb_c - vcr_c - vp_c - self.p.series_resistance_ohm * ir_c) / self.p.lr_h
        f_ir_s = (vb_s - vcr_s - vp_s - self.p.series_resistance_ohm * ir_s) / self.p.lr_h
        d_ir_c = f_ir_c - omega * ir_s
        d_ir_s = f_ir_s + omega * ir_c

        d_vcr_c = ir_c / self.p.cr_f - omega * vcr_s
        d_vcr_s = ir_s / self.p.cr_f + omega * vcr_c

        f_im_c = (vp_c - self.p.magnetizing_series_resistance_ohm * im_c) / self.p.lm_h
        f_im_s = (vp_s - self.p.magnetizing_series_resistance_ohm * im_s) / self.p.lm_h
        d_im_c = f_im_c - omega * im_s
        d_im_s = f_im_s + omega * im_c

        rload = self.p.load_resistance_ohm
        esr = self.p.output_cap_esr_ohm
        i_cap = (
            alg["rectifier_current_avg_a"]
            - alg["output_voltage_v"] / rload
            - inputs.load_current_disturbance_a
        )
        d_vco = i_cap / self.p.output_capacitance_f

        return np.asarray(
            [d_ir_c, d_ir_s, d_vcr_c, d_vcr_s,
             d_im_c, d_im_s, d_vco], dtype=float,
        )

    def outputs(
        self,
        states: NDArray[np.float64],
        inputs: LLCPlantInputs | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return standard nonlinear plant outputs used for linearization."""
        if not isinstance(inputs, LLCPlantInputs):
            inputs = LLCPlantInputs.from_array(np.asarray(inputs, dtype=float))
        ir_c, ir_s, _, _, im_c, im_s, vco = map(float, states)
        alg = self.algebraic(states, inputs)
        ir_peak = self._amplitude(ir_c, ir_s)
        im_peak = self._amplitude(im_c, im_s)
        load_peak = alg["primary_load_peak_a"]
        return np.asarray(
            [
                alg["output_voltage_v"],
                ir_peak / SQRT2,
                im_peak / SQRT2,
                self.p.turns_ratio * load_peak / SQRT2,
                alg["rectifier_current_avg_a"],
                vco,
            ],
            dtype=float,
        )

    def initial_guess(
        self,
        inputs: LLCPlantInputs,
        operating_point: LLCOperatingPoint | None = None,
        output_voltage_guess_v: float | None = None,
    ) -> NDArray[np.float64]:
        """Build a robust phasor initial guess from the FHA circuit."""
        omega = 2.0 * PI * inputs.switching_frequency_hz
        if operating_point is not None:
            rac = operating_point.rac_ohm
            vout = output_voltage_guess_v or operating_point.output_current_a * self.p.load_resistance_ohm
        else:
            vout = output_voltage_guess_v or math.sqrt(
                max(inputs.bus_voltage_v**2 / max(self.p.load_resistance_ohm, 1e-12), 1.0)
            )
            rac = (8.0 / PI**2) * self.p.turns_ratio**2 * self.p.load_resistance_ohm

        # Re{X exp(j theta)} = x_c cos(theta) + x_s sin(theta), therefore
        # X = x_c - j*x_s.  The bridge fundamental is a sine wave.
        vb_peak = FOUR_OVER_PI * self.p.bridge_gain * inputs.bus_voltage_v
        vb = complex(0.0, -vb_peak)
        z_series = (
            self.p.series_resistance_ohm
            + 1j * omega * self.p.lr_h
            + 1.0 / (1j * omega * self.p.cr_f)
        )
        z_lm = 1j * omega * self.p.lm_h
        z_parallel = 1.0 / (1.0 / max(rac, 1e-9) + 1.0 / z_lm)
        ir = vb / (z_series + z_parallel)
        vp = ir * z_parallel
        im = vp / z_lm
        vcr = ir / (1j * omega * self.p.cr_f)

        def coeff(value: complex) -> tuple[float, float]:
            return float(value.real), float(-value.imag)

        ir_c, ir_s = coeff(ir)
        vcr_c, vcr_s = coeff(vcr)
        im_c, im_s = coeff(im)
        return np.asarray(
            [ir_c, ir_s, vcr_c, vcr_s, im_c, im_s, float(vout)],
            dtype=float,
        )

    def solve_steady_state(
        self,
        inputs: LLCPlantInputs,
        *,
        operating_point: LLCOperatingPoint | None = None,
        initial_states: NDArray[np.float64] | None = None,
        tolerance: float = 1e-9,
        max_evaluations: int = 1000,
    ) -> DynamicPhasorSteadyState:
        """Solve ``rhs(x, u)=0`` for the selected LLC work point."""
        guess = (
            np.asarray(initial_states, dtype=float).copy()
            if initial_states is not None
            else self.initial_guess(inputs, operating_point)
        )
        if guess.shape != (7,):
            raise ValueError("dynamic-phasor initial state must contain seven values")

        # Scale residuals by representative state rates so the output-capacitor
        # equation is not numerically hidden behind MHz resonant derivatives.
        current_scale = max(
            operating_point.resonant_current_peak_a if operating_point else 10.0,
            1.0,
        )
        voltage_scale = max(inputs.bus_voltage_v, 10.0)
        output_scale = max(
            operating_point.output_current_a if operating_point else voltage_scale / self.p.load_resistance_ohm,
            1.0,
        )
        omega = 2.0 * PI * inputs.switching_frequency_hz
        scales = np.asarray(
            [current_scale * omega, current_scale * omega,
             voltage_scale * omega, voltage_scale * omega,
             current_scale * omega, current_scale * omega,
             output_scale / self.p.output_capacitance_f],
            dtype=float,
        )

        evaluations = 0

        def residual(x: NDArray[np.float64]) -> NDArray[np.float64]:
            nonlocal evaluations
            evaluations += 1
            return self.rhs(x, inputs) / scales

        solution = root(
            residual,
            guess,
            method="hybr",
            options={"xtol": tolerance, "maxfev": max_evaluations},
        )
        residual_norm = float(np.linalg.norm(self.rhs(solution.x, inputs)))
        # A physically useful acceptance check is expressed in normalized
        # derivatives rather than raw SI units.
        normalized_norm = float(np.linalg.norm(residual(solution.x)))
        converged = bool(solution.success and normalized_norm <= 1e-6)
        if not converged:
            raise PlantModelError(
                "dynamic-phasor steady-state solve failed: "
                f"{solution.message}; normalized residual={normalized_norm:.3e}"
            )

        out = self.outputs(solution.x, inputs)
        alg = self.algebraic(solution.x, inputs)
        ir_peak = out[1] * SQRT2
        im_peak = out[2] * SQRT2
        return DynamicPhasorSteadyState(
            states=np.asarray(solution.x, dtype=float),
            inputs=inputs,
            residual_norm=residual_norm,
            iterations=evaluations,
            converged=converged,
            output_voltage_v=float(out[0]),
            output_capacitor_voltage_v=float(out[5]),
            rectifier_current_avg_a=float(out[4]),
            resonant_current_peak_a=float(ir_peak),
            resonant_current_rms_a=float(out[1]),
            magnetizing_current_peak_a=float(im_peak),
            magnetizing_current_rms_a=float(out[2]),
            primary_load_current_peak_a=float(alg["primary_load_peak_a"]),
            secondary_current_rms_a=float(out[3]),
        )

    def solve_regulated_steady_state(
        self,
        *,
        bus_voltage_v: float,
        target_output_voltage_v: float,
        frequency_guess_hz: float,
        minimum_frequency_hz: float,
        maximum_frequency_hz: float,
        operating_point: LLCOperatingPoint | None = None,
        load_current_disturbance_a: float = 0.0,
        output_tolerance_v: float = 2e-3,
        frequency_samples: int = 31,
    ) -> DynamicPhasorSteadyState:
        """Solve the regulated equilibrium ``Vo(fs)=Vo_target``.

        The ordinary FHA operating-point frequency is an excellent initial
        estimate, but conductive damping and rectifier drops shift the actual
        equilibrium.  This outer scalar solve keeps the small-signal model and
        waveform reconstruction centered on the regulated output voltage.

        Multiple gain roots may exist.  All sign-change roots inside the
        permitted frequency range are evaluated and the root closest to the
        supplied frequency guess is selected.  If no sign change is found, a
        bounded minimum-error solve is used and rejected when the voltage error
        remains larger than ``output_tolerance_v``.
        """
        if bus_voltage_v <= 0.0 or target_output_voltage_v <= 0.0:
            raise ValueError("bus and target output voltage must be positive")
        if not (0.0 < minimum_frequency_hz < maximum_frequency_hz):
            raise ValueError("invalid regulated-frequency bounds")
        if not (minimum_frequency_hz <= frequency_guess_hz <= maximum_frequency_hz):
            frequency_guess_hz = min(
                max(frequency_guess_hz, minimum_frequency_hz), maximum_frequency_hz)
        frequency_samples = max(int(frequency_samples), 9)

        cache: dict[float, DynamicPhasorSteadyState] = {}
        last_states: NDArray[np.float64] | None = None

        def solve_at(frequency_hz: float) -> DynamicPhasorSteadyState:
            nonlocal last_states
            key = float(frequency_hz)
            if key in cache:
                return cache[key]
            inputs = LLCPlantInputs(
                switching_frequency_hz=key,
                bus_voltage_v=bus_voltage_v,
                load_current_disturbance_a=load_current_disturbance_a,
            )
            try:
                steady = self.solve_steady_state(
                    inputs,
                    operating_point=operating_point,
                    initial_states=last_states,
                )
            except PlantModelError:
                steady = self.solve_steady_state(
                    inputs,
                    operating_point=operating_point,
                    initial_states=None,
                )
            last_states = steady.states
            cache[key] = steady
            return steady

        def error(frequency_hz: float) -> float:
            return solve_at(float(frequency_hz)).output_voltage_v - target_output_voltage_v

        # A geometric grid resolves the steep low-frequency region better than
        # a linear grid while still including the original FHA estimate.
        grid = np.geomspace(
            minimum_frequency_hz, maximum_frequency_hz, frequency_samples)
        grid = np.unique(np.append(grid, frequency_guess_hz))
        grid.sort()
        values: list[tuple[float, float]] = []
        for frequency in grid:
            try:
                values.append((float(frequency), float(error(float(frequency)))))
            except PlantModelError:
                continue
        if not values:
            raise PlantModelError("regulated EDF solve failed at every sampled frequency")

        roots: list[float] = []
        for (f0, e0), (f1, e1) in zip(values[:-1], values[1:]):
            if abs(e0) <= output_tolerance_v:
                roots.append(f0)
            if e0 * e1 < 0.0:
                result = root_scalar(error, bracket=(f0, f1), method="brentq", xtol=1e-7)
                if result.converged:
                    roots.append(float(result.root))
        if abs(values[-1][1]) <= output_tolerance_v:
            roots.append(values[-1][0])

        if roots:
            selected_frequency = min(roots, key=lambda f: abs(f - frequency_guess_hz))
            steady = solve_at(selected_frequency)
        else:
            result = minimize_scalar(
                lambda f: abs(error(float(f))),
                bounds=(minimum_frequency_hz, maximum_frequency_hz),
                method="bounded",
                options={"xatol": 1e-6},
            )
            steady = solve_at(float(result.x))
            if abs(steady.output_voltage_v - target_output_voltage_v) > output_tolerance_v:
                raise PlantModelError(
                    "regulated EDF frequency solve cannot reach target output: "
                    f"best Vo={steady.output_voltage_v:.6f} V at "
                    f"{steady.inputs.switching_frequency_hz/1e3:.6f} kHz"
                )

        return DynamicPhasorSteadyState(
            **{
                **steady.__dict__,
                "target_output_voltage_v": float(target_output_voltage_v),
                "output_voltage_error_v": float(
                    steady.output_voltage_v - target_output_voltage_v),
                "frequency_trimmed": True,
            }
        )


def estimate_equivalent_series_resistance(
    analysis: "SystemAnalysis | None",
    operating_point: LLCOperatingPoint,
) -> float:
    """Collapse current-squared losses into an equivalent primary resistance.

    The estimate is used only as damping in the dynamic model.  It excludes
    turn-off, gate-drive, Coss and auxiliary losses because those do not behave
    as a linear tank resistance around the operating point.
    """
    if analysis is None:
        return 0.10
    try:
        point = min(
            analysis.operating_points,
            key=lambda item: (
                abs(item.operating_point.vbus_v - operating_point.vbus_v)
                + 1000.0 * abs(item.operating_point.load_fraction - operating_point.load_fraction)
            ),
        )
        conductive_w = (
            point.primary.conduction_w
            + point.synchronous_rectifier.conduction_w
            + point.transformer.primary_copper_w
            + point.transformer.secondary_copper_w
            + point.resonant_inductor.copper_w
            + point.resonant_capacitor.loss_w
        )
        ir_rms_sq = max(operating_point.resonant_current_rms_a**2, 1e-9)
        # Limit pathological reference records while preserving the loss-based
        # physical estimate for normal designs.
        return min(max(conductive_w / ir_rms_sq, 1e-4), 5.0)
    except (AttributeError, ValueError):
        return 0.10
