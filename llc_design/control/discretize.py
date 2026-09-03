"""ZOH discretization and difference-equation generation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray
from scipy import signal

from .linearize import LinearizedPlant


@dataclass(frozen=True)
class DifferenceEquation:
    """Normalized causal difference equation in powers of z^-1."""

    denominator: NDArray[np.float64]
    numerator: NDArray[np.float64]
    input_delay_samples: int
    input_name: str
    output_name: str

    def text(self, precision: int = 10) -> str:
        terms: list[str] = []
        for index, coefficient in enumerate(self.denominator[1:], start=1):
            sign = "+" if -coefficient >= 0.0 else "-"
            terms.append(f" {sign} {abs(-coefficient):.{precision}g}*y[k-{index}]")
        for index, coefficient in enumerate(self.numerator):
            if abs(coefficient) < 1e-18:
                continue
            delay = index + self.input_delay_samples
            suffix = "k" if delay == 0 else f"k-{delay}"
            sign = "+" if coefficient >= 0.0 else "-"
            terms.append(f" {sign} {abs(coefficient):.{precision}g}*u[{suffix}]")
        expression = "".join(terms).lstrip()
        if expression.startswith("+"):
            expression = expression[1:].lstrip()
        return f"y[k] = {expression or '0'}"


@dataclass(frozen=True)
class DiscretePlant:
    ad: NDArray[np.float64]
    bd: NDArray[np.float64]
    cd: NDArray[np.float64]
    dd: NDArray[np.float64]
    sample_time_s: float
    numerator: NDArray[np.float64]
    denominator: NDArray[np.float64]
    poles: NDArray[np.complex128]
    zeros: NDArray[np.complex128]
    input_name: str
    input_unit: str
    output_name: str
    output_unit: str
    input_delay_samples: int = 0

    @property
    def stable(self) -> bool:
        return bool(np.all(np.abs(self.poles) < 1.0))

    @property
    def difference_equation(self) -> DifferenceEquation:
        return DifferenceEquation(
            denominator=self.denominator,
            numerator=self.numerator,
            input_delay_samples=self.input_delay_samples,
            input_name=self.input_name,
            output_name=self.output_name,
        )

    def frequency_response(self, frequencies_hz: NDArray[np.float64]) -> NDArray[np.complex128]:
        frequencies_hz = np.asarray(frequencies_hz, dtype=float)
        z = np.exp(1j * 2.0 * math.pi * frequencies_hz * self.sample_time_s)
        numerator = np.zeros_like(z, dtype=complex)
        denominator = np.zeros_like(z, dtype=complex)
        for index, coefficient in enumerate(self.numerator):
            numerator += coefficient * z ** (-(index + self.input_delay_samples))
        for index, coefficient in enumerate(self.denominator):
            denominator += coefficient * z ** (-index)
        return numerator / denominator


def discretize_zoh(
    plant: LinearizedPlant,
    sample_time_s: float,
    *,
    input_name: str | None = None,
    output_name: str = "output_voltage_v",
    input_delay_samples: int = 0,
    relative_trim: float = 1e-12,
) -> DiscretePlant:
    if sample_time_s <= 0.0:
        raise ValueError("sample time must be positive")
    if input_delay_samples < 0:
        raise ValueError("input delay cannot be negative")
    input_idx = 0 if input_name is None else plant.input_index(input_name)
    output_idx = plant.output_index(output_name)
    b = plant.b[:, [input_idx]]
    c = plant.c[[output_idx], :]
    d = plant.d[[output_idx], [input_idx]].reshape(1, 1)
    ad, bd, cd, dd, _ = signal.cont2discrete(
        (plant.a, b, c, d), sample_time_s, method="zoh")
    num_all, den = signal.ss2tf(ad, bd, cd, dd, input=0)
    numerator = np.asarray(num_all[0], dtype=float)
    denominator = np.asarray(den, dtype=float)
    numerator = numerator / denominator[0]
    denominator = denominator / denominator[0]
    # Unlike continuous transfer functions, leading discrete numerator zeros
    # are physically meaningful sample delays and must not be discarded.  Only
    # suppress numerical noise.
    max_num = max(float(np.max(np.abs(numerator))), 1.0)
    numerator[np.abs(numerator) < relative_trim * max_num] = 0.0
    poles = np.roots(denominator).astype(complex)
    first_nonzero = next((i for i, value in enumerate(numerator) if abs(value) > 0.0), len(numerator) - 1)
    zeros = (
        np.roots(numerator[first_nonzero:]).astype(complex)
        if len(numerator[first_nonzero:]) > 1
        else np.asarray([], dtype=complex)
    )
    return DiscretePlant(
        ad=np.asarray(ad, dtype=float),
        bd=np.asarray(bd, dtype=float),
        cd=np.asarray(cd, dtype=float),
        dd=np.asarray(dd, dtype=float),
        sample_time_s=sample_time_s,
        numerator=numerator,
        denominator=denominator,
        poles=poles,
        zeros=zeros,
        input_name=plant.input_names[input_idx],
        input_unit=plant.input_units[input_idx],
        output_name=plant.output_names[output_idx],
        output_unit=plant.output_units[output_idx],
        input_delay_samples=input_delay_samples,
    )
