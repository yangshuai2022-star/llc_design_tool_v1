"""Numerical linearization of the nonlinear LLC dynamic-phasor model."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy import signal

from ..dynamics.plant import DynamicPhasorModel, DynamicPhasorSteadyState


class ControlInputKind(str, Enum):
    """Physical command variable used by the digital controller."""

    FREQUENCY_HZ = "frequency_hz"
    FREQUENCY_KHZ = "frequency_khz"
    PERIOD_S = "period_s"
    TIMER_COUNTS = "timer_counts"


@dataclass(frozen=True)
class SISOTransferFunction:
    numerator: NDArray[np.float64]
    denominator: NDArray[np.float64]
    input_name: str
    input_unit: str
    output_name: str
    output_unit: str
    poles: NDArray[np.complex128]
    zeros: NDArray[np.complex128]
    dc_gain: float

    @property
    def order(self) -> int:
        return max(0, len(self.denominator) - 1)

    def evaluate(self, s: complex) -> complex:
        return complex(np.polyval(self.numerator, s) / np.polyval(self.denominator, s))

    def frequency_response(
        self, frequencies_hz: NDArray[np.float64],
    ) -> NDArray[np.complex128]:
        omega = 2.0 * math.pi * np.asarray(frequencies_hz, dtype=float)
        _, response = signal.freqresp(
            signal.TransferFunction(self.numerator, self.denominator), w=omega)
        return np.asarray(response, dtype=complex)

    def scaled(
        self,
        factor: float,
        *,
        input_name: str | None = None,
        input_unit: str | None = None,
        output_name: str | None = None,
        output_unit: str | None = None,
    ) -> "SISOTransferFunction":
        """Return a gain-scaled copy without altering poles or zeros.

        This is used for sign conventions such as
        ``Zout=-v_hat/i_load_hat`` and for engineering-unit transforms.
        """
        numerator = np.asarray(self.numerator, dtype=float) * float(factor)
        zeros = (
            np.roots(numerator).astype(complex)
            if len(numerator) > 1 and np.any(np.abs(numerator[:-1]) > 0.0)
            else np.asarray([], dtype=complex)
        )
        return SISOTransferFunction(
            numerator=numerator,
            denominator=self.denominator.copy(),
            input_name=input_name or self.input_name,
            input_unit=input_unit or self.input_unit,
            output_name=output_name or self.output_name,
            output_unit=output_unit or self.output_unit,
            poles=self.poles.copy(),
            zeros=zeros,
            dc_gain=self.dc_gain * float(factor),
        )


@dataclass(frozen=True)
class LinearizedPlant:
    """Continuous-time MIMO linearization around one LLC work point."""

    a: NDArray[np.float64]
    b: NDArray[np.float64]
    c: NDArray[np.float64]
    d: NDArray[np.float64]
    state_names: tuple[str, ...]
    input_names: tuple[str, ...]
    input_units: tuple[str, ...]
    output_names: tuple[str, ...]
    output_units: tuple[str, ...]
    steady_states: NDArray[np.float64]
    steady_inputs: NDArray[np.float64]
    steady_outputs: NDArray[np.float64]
    poles: NDArray[np.complex128]
    model_name: str = "seven_state_dynamic_phasor_edf"
    control_input_kind: ControlInputKind = ControlInputKind.FREQUENCY_HZ
    timer_clock_hz: float | None = None

    @property
    def stable(self) -> bool:
        return bool(np.all(np.real(self.poles) < 0.0))

    def input_index(self, name: str) -> int:
        return self.input_names.index(name)

    def output_index(self, name: str) -> int:
        return self.output_names.index(name)

    def with_control_input(
        self,
        kind: ControlInputKind,
        *,
        timer_clock_hz: float | None = None,
    ) -> "LinearizedPlant":
        """Transform the first input from frequency to a controller variable."""
        f0 = float(self.steady_inputs[0])
        if kind == ControlInputKind.FREQUENCY_HZ:
            factor, name, unit = 1.0, "switching_frequency_hz", "Hz"
        elif kind == ControlInputKind.FREQUENCY_KHZ:
            factor, name, unit = 1000.0, "switching_frequency_khz", "kHz"
        elif kind == ControlInputKind.PERIOD_S:
            factor, name, unit = -f0**2, "switching_period_s", "s"
        elif kind == ControlInputKind.TIMER_COUNTS:
            if timer_clock_hz is None or timer_clock_hz <= 0.0:
                raise ValueError("timer_clock_hz must be positive for timer-count control")
            factor = -f0**2 / timer_clock_hz
            name, unit = "timer_period_counts", "count"
        else:  # pragma: no cover - Enum prevents this branch
            raise ValueError(f"unsupported control input kind: {kind}")
        transformed_b = self.b.copy()
        transformed_d = self.d.copy()
        transformed_b[:, 0] *= factor
        transformed_d[:, 0] *= factor
        input_names = list(self.input_names)
        input_units = list(self.input_units)
        input_names[0] = name
        input_units[0] = unit
        return replace(
            self,
            b=transformed_b,
            d=transformed_d,
            input_names=tuple(input_names),
            input_units=tuple(input_units),
            control_input_kind=kind,
            timer_clock_hz=timer_clock_hz,
        )

    def siso(
        self,
        *,
        input_name: str | None = None,
        output_name: str = "output_voltage_v",
        relative_trim: float = 1e-11,
    ) -> SISOTransferFunction:
        input_idx = 0 if input_name is None else self.input_index(input_name)
        output_idx = self.output_index(output_name)
        num_all, den = signal.ss2tf(
            self.a, self.b, self.c, self.d, input=input_idx)
        numerator = np.asarray(num_all[output_idx], dtype=float)
        denominator = np.asarray(den, dtype=float)
        if denominator[0] == 0.0:
            raise ValueError("invalid transfer-function denominator")
        numerator = numerator / denominator[0]
        denominator = denominator / denominator[0]
        max_num = max(float(np.max(np.abs(numerator))), 1.0)
        first = 0
        while first < len(numerator) - 1 and abs(numerator[first]) <= relative_trim * max_num:
            first += 1
        numerator = numerator[first:]
        zeros = (
            np.roots(numerator).astype(complex)
            if len(numerator) > 1 and np.any(np.abs(numerator[:-1]) > 0.0)
            else np.asarray([], dtype=complex)
        )
        poles = np.roots(denominator).astype(complex)
        dc_gain = float(numerator[-1] / denominator[-1]) if abs(denominator[-1]) > 0.0 else math.inf
        return SISOTransferFunction(
            numerator=numerator,
            denominator=denominator,
            input_name=self.input_names[input_idx],
            input_unit=self.input_units[input_idx],
            output_name=self.output_names[output_idx],
            output_unit=self.output_units[output_idx],
            poles=poles,
            zeros=zeros,
            dc_gain=dc_gain,
        )


def _central_jacobian(
    function: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    point: NDArray[np.float64],
    *,
    relative_step: float,
    absolute_steps: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    point = np.asarray(point, dtype=float)
    value = np.asarray(function(point), dtype=float)
    jac = np.zeros((len(value), len(point)), dtype=float)
    for column in range(len(point)):
        if absolute_steps is not None:
            step = max(float(absolute_steps[column]), relative_step * max(abs(point[column]), 1.0))
        else:
            step = relative_step * max(abs(point[column]), 1.0)
        plus = point.copy()
        minus = point.copy()
        plus[column] += step
        minus[column] -= step
        jac[:, column] = (
            np.asarray(function(plus), dtype=float)
            - np.asarray(function(minus), dtype=float)
        ) / (2.0 * step)
    return jac


def linearize_dynamic_phasor(
    model: DynamicPhasorModel,
    steady_state: DynamicPhasorSteadyState,
    *,
    relative_step: float = 1e-6,
) -> LinearizedPlant:
    """Numerically compute the continuous small-signal state-space model."""
    x0 = np.asarray(steady_state.states, dtype=float)
    u0 = steady_state.inputs.as_array()
    a = _central_jacobian(
        lambda x: model.rhs(x, u0), x0, relative_step=relative_step)
    b = _central_jacobian(
        lambda u: model.rhs(x0, u),
        u0,
        relative_step=relative_step,
        absolute_steps=np.asarray([0.25, 1e-3, 1e-4], dtype=float),
    )
    c = _central_jacobian(
        lambda x: model.outputs(x, u0), x0, relative_step=relative_step)
    d = _central_jacobian(
        lambda u: model.outputs(x0, u),
        u0,
        relative_step=relative_step,
        absolute_steps=np.asarray([0.25, 1e-3, 1e-4], dtype=float),
    )
    poles = np.linalg.eigvals(a).astype(complex)
    outputs = model.outputs(x0, u0)
    return LinearizedPlant(
        a=a,
        b=b,
        c=c,
        d=d,
        state_names=tuple(model.state_names),
        input_names=tuple(model.input_names),
        input_units=("Hz", "V", "A"),
        output_names=tuple(model.output_names),
        output_units=("V", "A", "A", "A", "A", "V"),
        steady_states=x0,
        steady_inputs=u0,
        steady_outputs=np.asarray(outputs, dtype=float),
        poles=poles,
    )
