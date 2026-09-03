"""Digital voltage-loop modelling for the LLC design tool.

The module connects the already-linearized frequency-to-output LLC plant to the
actual firmware signal chain:

    Vout -> analog divider/filter -> ADC multi-SOC recursive average
         -> PI/PIF/2P2Z -> PCMD -> piecewise FM LUT -> PWM/ZOH -> Gvf(s)

The frequency-domain implementation is intentionally mixed-domain.  Continuous
power-stage and analog blocks are evaluated at ``s=j*w``; digital blocks are
evaluated at ``z=exp(j*w*Ts)``.  An additional all-discrete approximation is
built for z-plane pole checking.  Saturation, burst mode, soft-start, current
limit selection and protection state machines are nonlinear and are therefore
reported as validity conditions rather than included in the linear Bode model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import signal

from .analysis import SmallSignalAnalysis


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


class ControllerKind(str, Enum):
    PI = "pi"
    PIF = "pif"
    TWO_P_TWO_Z = "2p2z"


class FMLUTMode(str, Enum):
    PCMD_TO_TBPRD = "pcmd_to_tbprd"
    PCMD_TO_FREQUENCY = "pcmd_to_frequency"


class PWMCountMode(str, Enum):
    UP = "up"
    UP_DOWN = "up_down"

    @property
    def frequency_divisor(self) -> float:
        return 1.0 if self == PWMCountMode.UP else 2.0


class DelayEnvelope(str, Enum):
    MINIMUM = "minimum"
    NOMINAL = "nominal"
    MAXIMUM = "maximum"


@dataclass(frozen=True)
class DigitalTransferFunction:
    """Transfer function represented in powers of z^-1.

    ``denominator[0]`` is normalized to one.  The corresponding difference
    equation is::

        y[k] = -a1*y[k-1] - ... + b0*x[k] + b1*x[k-1] + ...
    """

    numerator: FloatArray
    denominator: FloatArray
    sample_time_s: float
    name: str = "C(z)"
    input_name: str = "error"
    output_name: str = "command"

    def __post_init__(self) -> None:
        num = np.asarray(self.numerator, dtype=float).reshape(-1)
        den = np.asarray(self.denominator, dtype=float).reshape(-1)
        if len(num) == 0 or len(den) == 0:
            raise ValueError("digital transfer-function polynomials cannot be empty")
        if self.sample_time_s <= 0.0:
            raise ValueError("sample time must be positive")
        if abs(den[0]) < 1e-18:
            raise ValueError("digital transfer-function denominator leading coefficient is zero")
        num = num / den[0]
        den = den / den[0]
        object.__setattr__(self, "numerator", num)
        object.__setattr__(self, "denominator", den)

    @property
    def poles(self) -> ComplexArray:
        if len(self.denominator) <= 1:
            return np.asarray([], dtype=complex)
        return np.roots(self.denominator).astype(complex)

    @property
    def zeros(self) -> ComplexArray:
        nonzero = np.flatnonzero(np.abs(self.numerator) > 1e-16)
        if len(nonzero) == 0:
            return np.asarray([], dtype=complex)
        trimmed = self.numerator[nonzero[0]:]
        if len(trimmed) <= 1:
            return np.asarray([], dtype=complex)
        return np.roots(trimmed).astype(complex)

    @property
    def stable(self) -> bool:
        return bool(np.all(np.abs(self.poles) < 1.0))

    def frequency_response(self, frequencies_hz: Sequence[float] | FloatArray) -> ComplexArray:
        f = np.asarray(frequencies_hz, dtype=float)
        z = np.exp(1j * 2.0 * math.pi * f * self.sample_time_s)
        num = np.zeros_like(z, dtype=complex)
        den = np.zeros_like(z, dtype=complex)
        for index, coefficient in enumerate(self.numerator):
            num += coefficient * z ** (-index)
        for index, coefficient in enumerate(self.denominator):
            den += coefficient * z ** (-index)
        return np.asarray(num / den, dtype=complex)

    def cascade(self, other: "DigitalTransferFunction", *, name: str | None = None) -> "DigitalTransferFunction":
        if not math.isclose(self.sample_time_s, other.sample_time_s, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("cannot cascade digital blocks with different sample times")
        return DigitalTransferFunction(
            np.convolve(self.numerator, other.numerator),
            np.convolve(self.denominator, other.denominator),
            self.sample_time_s,
            name=name or f"{self.name}*{other.name}",
            input_name=self.input_name,
            output_name=other.output_name,
        )

    def scaled(self, gain: float, *, name: str | None = None) -> "DigitalTransferFunction":
        return DigitalTransferFunction(
            self.numerator * float(gain),
            self.denominator.copy(),
            self.sample_time_s,
            name=name or self.name,
            input_name=self.input_name,
            output_name=self.output_name,
        )

    def with_delay(self, samples: int, *, name: str | None = None) -> "DigitalTransferFunction":
        if samples < 0:
            raise ValueError("delay samples cannot be negative")
        if samples == 0:
            return self
        return DigitalTransferFunction(
            np.concatenate((np.zeros(samples, dtype=float), self.numerator)),
            self.denominator.copy(),
            self.sample_time_s,
            name=name or self.name,
            input_name=self.input_name,
            output_name=self.output_name,
        )

    def difference_equation(self, precision: int = 9) -> str:
        terms: list[str] = []
        for index, coefficient in enumerate(self.denominator[1:], start=1):
            value = -float(coefficient)
            sign = "+" if value >= 0.0 else "-"
            terms.append(f" {sign} {abs(value):.{precision}g}*y[k-{index}]")
        for index, coefficient in enumerate(self.numerator):
            if abs(float(coefficient)) < 1e-18:
                continue
            sign = "+" if coefficient >= 0.0 else "-"
            suffix = "k" if index == 0 else f"k-{index}"
            terms.append(f" {sign} {abs(float(coefficient)):.{precision}g}*x[{suffix}]")
        expression = "".join(terms).lstrip()
        if expression.startswith("+"):
            expression = expression[1:].lstrip()
        return f"y[k] = {expression or '0'}"


@dataclass(frozen=True)
class PIControllerConfig:
    kp: float = 0.01
    ti_s: float = 1.0e-3
    sample_time_s: float = 20.0e-6
    output_min: float = 0.0
    output_max: float = 1.0

    def transfer_function(self) -> DigitalTransferFunction:
        if self.kp <= 0.0 or self.ti_s <= 0.0 or self.sample_time_s <= 0.0:
            raise ValueError("PI Kp, Ti and sample time must be positive")
        ki2 = self.sample_time_s / (2.0 * self.ti_s)
        # Exact linear transfer of the firmware implementation:
        # i[k]=i[k-1]+ki2*(e[k]+e[k-1]); u=Kp*(e+i).
        num = self.kp * np.asarray([1.0 + ki2, -1.0 + ki2], dtype=float)
        den = np.asarray([1.0, -1.0], dtype=float)
        return DigitalTransferFunction(num, den, self.sample_time_s, "PI(z)")


@dataclass(frozen=True)
class PIFControllerConfig:
    kp: float = 0.01
    ti_s: float = 1.0e-3
    lpf_cutoff_hz: float = 3500.0
    sample_time_s: float = 20.0e-6
    output_min: float = 0.0
    output_max: float = 1.0

    @property
    def alpha(self) -> float:
        if self.lpf_cutoff_hz <= 0.0:
            return 1.0
        tau = 1.0 / (2.0 * math.pi * self.lpf_cutoff_hz)
        return self.sample_time_s / (self.sample_time_s + tau)

    def transfer_function(self) -> DigitalTransferFunction:
        pi = PIControllerConfig(
            self.kp, self.ti_s, self.sample_time_s,
            self.output_min, self.output_max,
        ).transfer_function()
        alpha = self.alpha
        output_lpf = DigitalTransferFunction(
            np.asarray([alpha], dtype=float),
            np.asarray([1.0, -(1.0 - alpha)], dtype=float),
            self.sample_time_s,
            "PI-output-LPF(z)",
        )
        return pi.cascade(output_lpf, name="PIF(z)")


@dataclass(frozen=True)
class TwoP2ZControllerConfig:
    b0: float = 0.0
    b1: float = 0.0
    b2: float = 0.0
    a1: float = 0.0
    a2: float = 0.0
    sample_time_s: float = 20.0e-6
    output_min: float = 0.0
    output_max: float = 1.0

    def transfer_function(self) -> DigitalTransferFunction:
        return DigitalTransferFunction(
            np.asarray([self.b0, self.b1, self.b2], dtype=float),
            np.asarray([1.0, self.a1, self.a2], dtype=float),
            self.sample_time_s,
            "2P2Z(z)",
        )

    @classmethod
    def from_analog_poles_zeros(
        cls,
        *,
        gain: float,
        zeros_hz: Sequence[float],
        poles_hz: Sequence[float],
        sample_time_s: float,
        output_min: float = 0.0,
        output_max: float = 1.0,
    ) -> "TwoP2ZControllerConfig":
        """Create a 2P2Z by bilinear-transforming a second-order analog form.

        A value of zero in ``poles_hz`` or ``zeros_hz`` creates a root at the
        origin.  The exact digital coefficients remain visible and editable.
        """
        if len(zeros_hz) != 2 or len(poles_hz) != 2:
            raise ValueError("2P2Z analog design requires exactly two poles and two zeros")
        zeros_rad = [-2.0 * math.pi * float(value) for value in zeros_hz]
        poles_rad = [-2.0 * math.pi * float(value) for value in poles_hz]
        num_s = float(gain) * np.poly(zeros_rad)
        den_s = np.poly(poles_rad)
        num_z, den_z, _ = signal.cont2discrete(
            (num_s, den_s), sample_time_s, method="bilinear")[:3]
        num = np.asarray(num_z, dtype=float).reshape(-1)
        den = np.asarray(den_z, dtype=float).reshape(-1)
        num = num / den[0]
        den = den / den[0]
        num = np.pad(num, (0, max(0, 3 - len(num))))[:3]
        den = np.pad(den, (0, max(0, 3 - len(den))))[:3]
        return cls(
            b0=float(num[0]), b1=float(num[1]), b2=float(num[2]),
            a1=float(den[1]), a2=float(den[2]),
            sample_time_s=sample_time_s,
            output_min=output_min, output_max=output_max,
        )


ControllerConfig = PIControllerConfig | PIFControllerConfig | TwoP2ZControllerConfig


def controller_kind(config: ControllerConfig) -> ControllerKind:
    if isinstance(config, PIFControllerConfig):
        return ControllerKind.PIF
    if isinstance(config, PIControllerConfig):
        return ControllerKind.PI
    if isinstance(config, TwoP2ZControllerConfig):
        return ControllerKind.TWO_P_TWO_Z
    raise TypeError(f"unsupported controller configuration: {type(config)!r}")


@dataclass(frozen=True)
class FrequencyModulatorLUT:
    pcmd: FloatArray
    values: FloatArray
    mode: FMLUTMode = FMLUTMode.PCMD_TO_TBPRD
    timer_clock_hz: float = 120.0e6
    count_mode: PWMCountMode = PWMCountMode.UP_DOWN
    name: str = "PCMD-FM-LUT"

    def __post_init__(self) -> None:
        pcmd = np.asarray(self.pcmd, dtype=float).reshape(-1)
        values = np.asarray(self.values, dtype=float).reshape(-1)
        if len(pcmd) < 2 or len(pcmd) != len(values):
            raise ValueError("FM LUT requires two or more PCMD/value pairs")
        if np.any(np.diff(pcmd) <= 0.0):
            raise ValueError("FM LUT PCMD values must be strictly increasing")
        if pcmd[0] > 0.0 or pcmd[-1] < 1.0:
            raise ValueError("FM LUT must cover the complete normalized PCMD range 0..1")
        if np.any(values <= 0.0):
            raise ValueError("FM LUT values must be positive")
        if self.timer_clock_hz <= 0.0:
            raise ValueError("PWM timer clock must be positive")
        object.__setattr__(self, "pcmd", pcmd)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "mode", FMLUTMode(self.mode))
        object.__setattr__(self, "count_mode", PWMCountMode(self.count_mode))

    @classmethod
    def firmware_default(cls) -> "FrequencyModulatorLUT":
        pcmd = np.asarray([
            0.0000, 0.0400, 0.0800, 0.1200, 0.1600,
            0.2000, 0.2500, 0.3000, 0.3550, 0.4100,
            0.4650, 0.5200, 0.5825, 0.6450, 0.7075,
            0.7700, 0.8275, 0.8850, 0.9425, 1.0000,
        ], dtype=float)
        tbprd = np.asarray([
            240, 258, 279, 303, 332,
            367, 424, 500, 533, 571,
            615, 667, 706, 750, 800,
            811, 822, 833, 845, 857,
        ], dtype=float)
        return cls(pcmd, tbprd, FMLUTMode.PCMD_TO_TBPRD, 120e6, PWMCountMode.UP_DOWN,
                   "firmware-20-point-PCMD-TBPRD")

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        mode: FMLUTMode,
        timer_clock_hz: float = 120e6,
        count_mode: PWMCountMode = PWMCountMode.UP_DOWN,
    ) -> "FrequencyModulatorLUT":
        rows: list[tuple[float, float]] = []
        for line_number, raw in enumerate(text.splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [part.strip() for part in line.replace(";", ",").split(",")]
            if len(parts) != 2:
                raise ValueError(f"invalid FM LUT line {line_number}: expected 'pcmd,value'")
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError as exc:
                # Permit one CSV header line.
                if line_number == 1 and any(char.isalpha() for char in line):
                    continue
                raise ValueError(f"invalid numeric FM LUT line {line_number}: {line}") from exc
        if len(rows) < 2:
            raise ValueError("FM LUT text does not contain enough nodes")
        return cls(
            np.asarray([row[0] for row in rows], dtype=float),
            np.asarray([row[1] for row in rows], dtype=float),
            mode, timer_clock_hz, count_mode,
        )

    def to_text(self) -> str:
        header = "pcmd,tbprd" if self.mode == FMLUTMode.PCMD_TO_TBPRD else "pcmd,frequency_hz"
        lines = [header]
        for pcmd, value in zip(self.pcmd, self.values, strict=True):
            lines.append(f"{pcmd:.7g},{value:.10g}")
        return "\n".join(lines)

    def _segment_index(self, command: float, *, side: str = "auto") -> int:
        command = float(np.clip(command, self.pcmd[0], self.pcmd[-1]))
        exact = np.flatnonzero(np.isclose(self.pcmd, command, rtol=0.0, atol=1e-12))
        if len(exact):
            node = int(exact[0])
            if side == "left":
                return max(0, node - 1)
            if side == "right":
                return min(len(self.pcmd) - 2, node)
            # Firmware selects the segment ending at the first upper node.  At
            # an internal node this is the left-hand segment.
            return max(0, min(len(self.pcmd) - 2, node - 1))
        return int(np.clip(np.searchsorted(self.pcmd, command) - 1, 0, len(self.pcmd) - 2))

    def value(self, command: float) -> float:
        p = float(np.clip(command, self.pcmd[0], self.pcmd[-1]))
        return float(np.interp(p, self.pcmd, self.values))

    def tbprd(self, command: float) -> float:
        if self.mode == FMLUTMode.PCMD_TO_TBPRD:
            return self.value(command)
        frequency = self.value(command)
        return self.timer_clock_hz / (self.count_mode.frequency_divisor * frequency)

    def frequency_hz(self, command: float) -> float:
        if self.mode == FMLUTMode.PCMD_TO_FREQUENCY:
            return self.value(command)
        tbprd = self.value(command)
        return self.timer_clock_hz / (self.count_mode.frequency_divisor * tbprd)

    def command_for_frequency(self, frequency_hz: float) -> float:
        target = float(frequency_hz)
        frequencies = np.asarray([self.frequency_hz(value) for value in self.pcmd], dtype=float)
        if np.all(np.diff(frequencies) < 0.0):
            return float(np.interp(target, frequencies[::-1], self.pcmd[::-1]))
        if np.all(np.diff(frequencies) > 0.0):
            return float(np.interp(target, frequencies, self.pcmd))
        # Generic monotonicity-free fallback: locate the nearest crossing.
        index = int(np.argmin(np.abs(frequencies - target)))
        return float(self.pcmd[index])

    def local_gain_hz_per_pu(self, command: float, *, side: str = "auto") -> float:
        index = self._segment_index(command, side=side)
        dp = self.pcmd[index + 1] - self.pcmd[index]
        slope_value = (self.values[index + 1] - self.values[index]) / dp
        if self.mode == FMLUTMode.PCMD_TO_FREQUENCY:
            return float(slope_value)
        tbprd = self.tbprd(command)
        return float(
            -self.timer_clock_hz
            / (self.count_mode.frequency_divisor * tbprd**2)
            * slope_value
        )

    def gain_sides_hz_per_pu(self, command: float) -> tuple[float, float]:
        return (
            self.local_gain_hz_per_pu(command, side="left"),
            self.local_gain_hz_per_pu(command, side="right"),
        )


@dataclass(frozen=True)
class FMOperatingPoint:
    command_pu: float
    frequency_hz: float
    tbprd_counts: float
    gain_hz_per_pu: float
    left_gain_hz_per_pu: float
    right_gain_hz_per_pu: float
    command_headroom_low: float
    command_headroom_high: float


def evaluate_fm_operating_point(
    lut: FrequencyModulatorLUT,
    *,
    switching_frequency_hz: float,
    command_pu: float | None = None,
) -> FMOperatingPoint:
    command = (
        lut.command_for_frequency(switching_frequency_hz)
        if command_pu is None else float(command_pu)
    )
    command = float(np.clip(command, 0.0, 1.0))
    left, right = lut.gain_sides_hz_per_pu(command)
    return FMOperatingPoint(
        command_pu=command,
        frequency_hz=lut.frequency_hz(command),
        tbprd_counts=lut.tbprd(command),
        gain_hz_per_pu=lut.local_gain_hz_per_pu(command),
        left_gain_hz_per_pu=left,
        right_gain_hz_per_pu=right,
        command_headroom_low=command,
        command_headroom_high=1.0 - command,
    )


@dataclass(frozen=True)
class AnalogSenseConfig:
    rup_ohm: float = 117.0e3
    rlow_ohm: float = 1.6e3
    divider_capacitance_f: float = 1.0e-9
    opamp_gain: float = 1.0
    opamp_bandwidth_hz: float = 0.0  # 0 = ideal buffer in the linear model
    adc_series_resistance_ohm: float = 220.0
    adc_shunt_capacitance_f: float = 2.0e-9
    normalize_to_engineering_units: bool = True
    calibration_gain: float | None = None

    def validate(self) -> None:
        if self.rup_ohm <= 0.0 or self.rlow_ohm <= 0.0:
            raise ValueError("sense-divider resistors must be positive")
        if self.divider_capacitance_f < 0.0:
            raise ValueError("divider capacitance cannot be negative")
        if self.opamp_gain <= 0.0:
            raise ValueError("op-amp gain must be positive")
        if self.opamp_bandwidth_hz < 0.0:
            raise ValueError("op-amp bandwidth cannot be negative")
        if self.adc_series_resistance_ohm < 0.0 or self.adc_shunt_capacitance_f < 0.0:
            raise ValueError("ADC RC values cannot be negative")

    @property
    def divider_gain(self) -> float:
        return self.rlow_ohm / (self.rup_ohm + self.rlow_ohm)

    @property
    def divider_thevenin_ohm(self) -> float:
        return self.rup_ohm * self.rlow_ohm / (self.rup_ohm + self.rlow_ohm)

    @property
    def divider_pole_hz(self) -> float:
        if self.divider_capacitance_f <= 0.0:
            return math.inf
        return 1.0 / (2.0 * math.pi * self.divider_thevenin_ohm * self.divider_capacitance_f)

    @property
    def adc_rc_pole_hz(self) -> float:
        if self.adc_series_resistance_ohm <= 0.0 or self.adc_shunt_capacitance_f <= 0.0:
            return math.inf
        return 1.0 / (
            2.0 * math.pi * self.adc_series_resistance_ohm * self.adc_shunt_capacitance_f)

    @property
    def effective_calibration_gain(self) -> float:
        if self.calibration_gain is not None:
            return float(self.calibration_gain)
        if self.normalize_to_engineering_units:
            return 1.0 / (self.divider_gain * self.opamp_gain)
        return 1.0

    def frequency_response_components(
        self, frequencies_hz: Sequence[float] | FloatArray,
    ) -> dict[str, ComplexArray]:
        self.validate()
        f = np.asarray(frequencies_hz, dtype=float)
        s = 1j * 2.0 * math.pi * f
        divider = np.full_like(s, self.divider_gain, dtype=complex)
        if self.divider_capacitance_f > 0.0:
            divider /= 1.0 + s * self.divider_thevenin_ohm * self.divider_capacitance_f
        opamp = np.full_like(s, self.opamp_gain, dtype=complex)
        if self.opamp_bandwidth_hz > 0.0:
            opamp /= 1.0 + s / (2.0 * math.pi * self.opamp_bandwidth_hz)
        adc_rc = np.ones_like(s, dtype=complex)
        if self.adc_series_resistance_ohm > 0.0 and self.adc_shunt_capacitance_f > 0.0:
            adc_rc /= 1.0 + s * self.adc_series_resistance_ohm * self.adc_shunt_capacitance_f
        raw = divider * opamp * adc_rc
        calibrated = raw * self.effective_calibration_gain
        return {
            "divider": divider,
            "opamp": opamp,
            "adc_rc": adc_rc,
            "raw_analog": raw,
            "calibrated_analog": calibrated,
        }

    def normalized_continuous_tf(self) -> tuple[FloatArray, FloatArray]:
        """Return the engineering-unit analog sense path as a continuous TF."""
        self.validate()
        gain = self.divider_gain * self.opamp_gain * self.effective_calibration_gain
        numerator = np.asarray([gain], dtype=float)
        denominator = np.asarray([1.0], dtype=float)
        taus = [
            self.divider_thevenin_ohm * self.divider_capacitance_f,
            (1.0 / (2.0 * math.pi * self.opamp_bandwidth_hz)
             if self.opamp_bandwidth_hz > 0.0 else 0.0),
            self.adc_series_resistance_ohm * self.adc_shunt_capacitance_f,
        ]
        for tau in taus:
            if tau > 0.0:
                denominator = np.convolve(denominator, np.asarray([tau, 1.0], dtype=float))
        return numerator, denominator


@dataclass(frozen=True)
class ADCSamplingConfig:
    control_sample_time_s: float = 20.0e-6
    adc_clock_hz: float = 60.0e6
    acquisition_time_s: float = 300.0e-9
    conversion_cycles: float = 13.0
    soc_count: int = 3
    recursive_previous_weight: float = 0.25
    soc_sample_offsets_s: tuple[float, ...] | None = None

    def validate(self) -> None:
        if self.control_sample_time_s <= 0.0 or self.adc_clock_hz <= 0.0:
            raise ValueError("ADC and control clocks must be positive")
        if self.acquisition_time_s <= 0.0:
            raise ValueError("ADC acquisition time must be positive")
        if self.conversion_cycles <= 0.0:
            raise ValueError("ADC conversion cycles must be positive")
        if self.soc_count <= 0:
            raise ValueError("SOC count must be positive")
        if not (0.0 <= self.recursive_previous_weight < 1.0):
            raise ValueError("recursive previous weight must be in [0,1)")
        if self.soc_sample_offsets_s is not None and len(self.soc_sample_offsets_s) != self.soc_count:
            raise ValueError("explicit SOC sample offsets must match SOC count")

    @property
    def conversion_time_s(self) -> float:
        return self.conversion_cycles / self.adc_clock_hz

    @property
    def soc_slot_time_s(self) -> float:
        return self.acquisition_time_s + self.conversion_time_s

    @property
    def sample_offsets_s(self) -> FloatArray:
        self.validate()
        if self.soc_sample_offsets_s is not None:
            return np.asarray(self.soc_sample_offsets_s, dtype=float)
        # Sequential SOC2/SOC3/SOC4 model.  Each offset is the acquisition
        # aperture centre relative to the EPWM5 SOCB trigger.
        return np.asarray([
            index * self.soc_slot_time_s + 0.5 * self.acquisition_time_s
            for index in range(self.soc_count)
        ], dtype=float)

    @property
    def eoc_delay_s(self) -> float:
        if self.soc_sample_offsets_s is not None:
            return float(np.max(self.sample_offsets_s) + 0.5 * self.acquisition_time_s
                         + self.conversion_time_s)
        return self.soc_count * self.soc_slot_time_s

    @property
    def current_sample_weight(self) -> float:
        return (1.0 - self.recursive_previous_weight) / self.soc_count

    @property
    def effective_sample_offset_s(self) -> float:
        return float(np.mean(self.sample_offsets_s))

    def frequency_response(self, frequencies_hz: Sequence[float] | FloatArray) -> ComplexArray:
        """Response of the SOC aperture and recursive multi-SOC average.

        The digital sequence is indexed at the EPWM5 SOCB trigger.  A sample
        taken later in the frame therefore contributes ``exp(+j*w*tau)``.  The
        command-path delay subsequently places the new PCMD at its actual PWM
        update time, producing the physically correct net sample-to-actuation
        delay.
        """
        f = np.asarray(frequencies_hz, dtype=float)
        omega = 2.0 * math.pi * f
        aperture = np.sinc(f * self.acquisition_time_s)
        samples = np.zeros_like(f, dtype=complex)
        weight = self.current_sample_weight
        for offset in self.sample_offsets_s:
            samples += weight * aperture * np.exp(1j * omega * offset)
        z_inv = np.exp(-1j * omega * self.control_sample_time_s)
        return np.asarray(samples / (1.0 - self.recursive_previous_weight * z_inv), dtype=complex)

    def simplified_digital_filter(self) -> DigitalTransferFunction:
        return DigitalTransferFunction(
            np.asarray([1.0 - self.recursive_previous_weight], dtype=float),
            np.asarray([1.0, -self.recursive_previous_weight], dtype=float),
            self.control_sample_time_s,
            "ADC-recursive-average(z)",
            input_name="sampled_voltage",
            output_name="measured_voltage",
        )


@dataclass(frozen=True)
class CommandTimingConfig:
    computation_delay_s: float = 1.0e-6
    include_zero_order_hold: bool = True

    def validate(self) -> None:
        if self.computation_delay_s < 0.0:
            raise ValueError("computation delay cannot be negative")

    @staticmethod
    def pwm_zero_wait_s(switching_frequency_hz: float, envelope: DelayEnvelope) -> float:
        if switching_frequency_hz <= 0.0:
            raise ValueError("switching frequency must be positive")
        period = 1.0 / switching_frequency_hz
        if envelope == DelayEnvelope.MINIMUM:
            return 0.0
        if envelope == DelayEnvelope.MAXIMUM:
            return period
        return 0.5 * period

    def application_delay_s(
        self,
        adc: ADCSamplingConfig,
        switching_frequency_hz: float,
        envelope: DelayEnvelope,
    ) -> float:
        self.validate()
        return (
            adc.eoc_delay_s
            + self.computation_delay_s
            + self.pwm_zero_wait_s(switching_frequency_hz, envelope)
        )

    def frequency_response(
        self,
        frequencies_hz: Sequence[float] | FloatArray,
        *,
        adc: ADCSamplingConfig,
        switching_frequency_hz: float,
        envelope: DelayEnvelope,
    ) -> ComplexArray:
        f = np.asarray(frequencies_hz, dtype=float)
        omega = 2.0 * math.pi * f
        delay = self.application_delay_s(adc, switching_frequency_hz, envelope)
        response = np.exp(-1j * omega * delay)
        if self.include_zero_order_hold:
            response = response * np.sinc(f * adc.control_sample_time_s) * np.exp(
                -1j * omega * 0.5 * adc.control_sample_time_s)
        return np.asarray(response, dtype=complex)


@dataclass(frozen=True)
class StabilityMargins:
    gain_crossovers_hz: tuple[float, ...]
    phase_margins_deg: tuple[float, ...]
    phase_crossovers_hz: tuple[float, ...]
    gain_margins_db: tuple[float, ...]
    critical_gain_crossover_hz: float | None
    phase_margin_deg: float | None
    critical_phase_crossover_hz: float | None
    gain_margin_db: float | None
    delay_margin_s: float | None


def _log_interpolate_crossing(
    frequencies: FloatArray,
    values: FloatArray,
    target: float,
) -> list[tuple[float, int, float]]:
    results: list[tuple[float, int, float]] = []
    log_f = np.log(np.maximum(frequencies, 1e-300))
    shifted = values - target
    for index in range(len(frequencies) - 1):
        a, b = shifted[index], shifted[index + 1]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if a == 0.0:
            results.append((float(frequencies[index]), index, 0.0))
            continue
        if a * b > 0.0 or b == a:
            continue
        fraction = float(-a / (b - a))
        lf = log_f[index] + fraction * (log_f[index + 1] - log_f[index])
        results.append((float(math.exp(lf)), index, fraction))
    return results


def _linear_between(values: FloatArray, index: int, fraction: float) -> float:
    return float(values[index] + fraction * (values[index + 1] - values[index]))


def calculate_stability_margins(
    frequencies_hz: FloatArray,
    open_loop: ComplexArray,
) -> StabilityMargins:
    frequencies = np.asarray(frequencies_hz, dtype=float)
    response = np.asarray(open_loop, dtype=complex)
    magnitude_db = 20.0 * np.log10(np.maximum(np.abs(response), 1e-300))
    phase_deg = np.unwrap(np.angle(response)) * 180.0 / math.pi

    gain_crossings = _log_interpolate_crossing(frequencies, magnitude_db, 0.0)
    gc_frequencies: list[float] = []
    phase_margins: list[float] = []
    for frequency, index, fraction in gain_crossings:
        phase = _linear_between(phase_deg, index, fraction)
        # Map the local phase to the equivalent branch nearest -180 degrees.
        while phase > 0.0:
            phase -= 360.0
        while phase <= -360.0:
            phase += 360.0
        gc_frequencies.append(frequency)
        phase_margins.append(180.0 + phase)

    phase_crossings_all: list[tuple[float, int, float]] = []
    minimum_phase = float(np.nanmin(phase_deg))
    maximum_phase = float(np.nanmax(phase_deg))
    k_min = math.floor((minimum_phase + 180.0) / 360.0) - 1
    k_max = math.ceil((maximum_phase + 180.0) / 360.0) + 1
    for k in range(k_min, k_max + 1):
        target = -180.0 + 360.0 * k
        phase_crossings_all.extend(_log_interpolate_crossing(frequencies, phase_deg, target))
    # Remove numerical duplicate crossings.
    phase_crossings_all.sort(key=lambda item: item[0])
    unique_pc: list[tuple[float, int, float]] = []
    for item in phase_crossings_all:
        if not unique_pc or abs(math.log(item[0] / unique_pc[-1][0])) > 1e-6:
            unique_pc.append(item)

    pc_frequencies: list[float] = []
    gain_margins: list[float] = []
    for frequency, index, fraction in unique_pc:
        mag = _linear_between(magnitude_db, index, fraction)
        pc_frequencies.append(frequency)
        gain_margins.append(-mag)

    critical_gc = None
    critical_pm = None
    delay_margin = None
    if gc_frequencies:
        critical_index = int(np.argmin(np.asarray(phase_margins, dtype=float)))
        critical_gc = gc_frequencies[critical_index]
        critical_pm = phase_margins[critical_index]
        delay_margin = math.radians(critical_pm) / (2.0 * math.pi * critical_gc)

    critical_pc = None
    critical_gm = None
    if pc_frequencies:
        positive = [
            (margin, frequency) for margin, frequency in zip(gain_margins, pc_frequencies, strict=True)
            if margin >= 0.0
        ]
        chosen = min(positive, default=min(zip(gain_margins, pc_frequencies, strict=True)))
        critical_gm, critical_pc = chosen

    return StabilityMargins(
        gain_crossovers_hz=tuple(gc_frequencies),
        phase_margins_deg=tuple(phase_margins),
        phase_crossovers_hz=tuple(pc_frequencies),
        gain_margins_db=tuple(gain_margins),
        critical_gain_crossover_hz=critical_gc,
        phase_margin_deg=critical_pm,
        critical_phase_crossover_hz=critical_pc,
        gain_margin_db=critical_gm,
        delay_margin_s=delay_margin,
    )


@dataclass(frozen=True)
class DiscreteClosedLoopApproximation:
    open_loop_numerator: FloatArray
    open_loop_denominator: FloatArray
    closed_loop_denominator: FloatArray
    closed_loop_poles: ComplexArray
    stable: bool
    integer_delay_samples: int
    fractional_delay_samples: float


def _pad_to_same_length(a: FloatArray, b: FloatArray) -> tuple[FloatArray, FloatArray]:
    length = max(len(a), len(b))
    return np.pad(a, (0, length - len(a))), np.pad(b, (0, length - len(b)))


def _fractional_delay_thiran(delay_samples: float, sample_time_s: float) -> DigitalTransferFunction:
    delay = float(np.clip(delay_samples, 0.0, 1.0))
    if delay <= 1e-12:
        return DigitalTransferFunction(np.asarray([1.0]), np.asarray([1.0]), sample_time_s, "fractional-delay")
    coefficient = (1.0 - delay) / (1.0 + delay)
    return DigitalTransferFunction(
        np.asarray([coefficient, 1.0], dtype=float),
        np.asarray([1.0, coefficient], dtype=float),
        sample_time_s,
        "Thiran-fractional-delay",
    )


def _continuous_to_discrete_tf(
    numerator: FloatArray,
    denominator: FloatArray,
    sample_time_s: float,
    *,
    name: str,
) -> DigitalTransferFunction:
    num_z, den_z, _ = signal.cont2discrete(
        (np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float)),
        sample_time_s,
        method="zoh",
    )[:3]
    return DigitalTransferFunction(
        np.asarray(num_z, dtype=float).reshape(-1),
        np.asarray(den_z, dtype=float).reshape(-1),
        sample_time_s,
        name,
    )


@dataclass(frozen=True)
class DigitalLoopAnalysis:
    small_signal: SmallSignalAnalysis
    controller_config: ControllerConfig
    controller: DigitalTransferFunction
    fm_lut: FrequencyModulatorLUT
    fm_operating_point: FMOperatingPoint
    analog_sense: AnalogSenseConfig
    adc_sampling: ADCSamplingConfig
    command_timing: CommandTimingConfig
    frequencies_hz: FloatArray
    responses: dict[str, ComplexArray]
    margins_minimum_delay: StabilityMargins
    margins_nominal_delay: StabilityMargins
    margins_maximum_delay: StabilityMargins
    discrete_approximation: DiscreteClosedLoopApproximation
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def nominal_open_loop(self) -> ComplexArray:
        return self.responses["open_loop_nominal"]

    @property
    def nominal_closed_loop(self) -> ComplexArray:
        return self.responses["closed_loop_nominal"]

    @property
    def likely_stable(self) -> bool:
        margin = self.margins_nominal_delay.phase_margin_deg
        gain_margin = self.margins_nominal_delay.gain_margin_db
        margin_ok = margin is not None and margin > 0.0
        gain_ok = gain_margin is None or gain_margin > 0.0
        return bool(margin_ok and gain_ok and self.discrete_approximation.stable)


def build_digital_loop_analysis(
    small_signal: SmallSignalAnalysis,
    *,
    controller_config: ControllerConfig,
    fm_lut: FrequencyModulatorLUT | None = None,
    command_pu: float | None = None,
    analog_sense: AnalogSenseConfig | None = None,
    adc_sampling: ADCSamplingConfig | None = None,
    command_timing: CommandTimingConfig | None = None,
    frequencies_hz: Sequence[float] | FloatArray | None = None,
) -> DigitalLoopAnalysis:
    """Connect the complete digital voltage-loop signal chain."""
    controller = controller_config.transfer_function()
    sample_time = controller.sample_time_s
    if not math.isclose(sample_time, small_signal.sample_time_s, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("controller and LLC ZOH plant sample times must match")
    lut = fm_lut or FrequencyModulatorLUT.firmware_default()
    analog = analog_sense or AnalogSenseConfig()
    adc = adc_sampling or ADCSamplingConfig(control_sample_time_s=sample_time)
    timing = command_timing or CommandTimingConfig()
    if not math.isclose(adc.control_sample_time_s, sample_time, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("ADC and controller sample times must match")
    analog.validate()
    adc.validate()
    timing.validate()

    fsw = small_signal.operating_point.switching_frequency_hz
    fm = evaluate_fm_operating_point(
        lut, switching_frequency_hz=fsw, command_pu=command_pu)

    if frequencies_hz is None:
        nyquist = 0.5 / sample_time
        upper = min(0.49 * nyquist, 0.25 * fsw)
        lower = max(0.1, upper / 2.0e5)
        frequencies = np.geomspace(lower, upper, 2400)
    else:
        frequencies = np.asarray(frequencies_hz, dtype=float)
    if np.any(frequencies <= 0.0):
        raise ValueError("Bode frequencies must be positive")

    plant = small_signal.continuous_transfer.frequency_response(frequencies)
    controller_response = controller.frequency_response(frequencies)
    analog_components = analog.frequency_response_components(frequencies)
    adc_response = adc.frequency_response(frequencies)
    sense = analog_components["calibrated_analog"] * adc_response
    fm_plant = fm.gain_hz_per_pu * plant

    responses: dict[str, ComplexArray] = {
        "power_stage": plant,
        "fm_power_stage": fm_plant,
        "controller": controller_response,
        "sense_analog_raw": analog_components["raw_analog"],
        "sense_analog_calibrated": analog_components["calibrated_analog"],
        "adc_sampling": adc_response,
        "sense_total": sense,
    }

    margins: dict[DelayEnvelope, StabilityMargins] = {}
    for envelope in DelayEnvelope:
        delay = timing.frequency_response(
            frequencies,
            adc=adc,
            switching_frequency_hz=fsw,
            envelope=envelope,
        )
        open_loop = controller_response * fm_plant * sense * delay
        closed_loop = open_loop / (1.0 + open_loop)
        sensitivity = 1.0 / (1.0 + open_loop)
        suffix = envelope.value
        responses[f"delay_{suffix}"] = delay
        responses[f"open_loop_{suffix}"] = open_loop
        responses[f"closed_loop_{suffix}"] = closed_loop
        responses[f"sensitivity_{suffix}"] = sensitivity
        margins[envelope] = calculate_stability_margins(frequencies, open_loop)

    # Closed-loop output impedance with the nominal loop sensitivity.
    zout = small_signal.output_impedance_transfer.frequency_response(frequencies)
    responses["closed_loop_output_impedance"] = (
        zout * responses["sensitivity_nominal"])

    # All-discrete approximation for z-plane pole checking.  The exact mixed
    # Bode above remains the primary result; this approximation collapses
    # intra-frame sampling and zero-event timing into a Thiran fractional delay.
    plant_d = DigitalTransferFunction(
        small_signal.discrete_plant.numerator,
        small_signal.discrete_plant.denominator,
        sample_time,
        "Gvf-ZOH(z)",
        input_name="frequency_hz",
        output_name="output_voltage_v",
    ).scaled(fm.gain_hz_per_pu, name="Gpcmd(z)")
    analog_num, analog_den = analog.normalized_continuous_tf()
    analog_d = _continuous_to_discrete_tf(
        analog_num, analog_den, sample_time, name="analog-sense-ZOH(z)")
    adc_d = adc.simplified_digital_filter()

    application_delay = timing.application_delay_s(adc, fsw, DelayEnvelope.NOMINAL)
    sample_to_actuation_delay = max(0.0, application_delay - adc.effective_sample_offset_s)
    delay_in_samples = sample_to_actuation_delay / sample_time
    integer_delay = int(math.floor(delay_in_samples))
    fractional_delay = delay_in_samples - integer_delay
    fractional_d = _fractional_delay_thiran(fractional_delay, sample_time)

    open_d = controller.cascade(plant_d).cascade(analog_d).cascade(adc_d).cascade(fractional_d)
    open_d = open_d.with_delay(integer_delay)
    open_num, open_den = _pad_to_same_length(open_d.numerator, open_d.denominator)
    closed_den = open_den + open_num
    # Normalize for robust root computation.
    if abs(closed_den[0]) > 1e-18:
        closed_den = closed_den / closed_den[0]
    closed_poles = np.roots(closed_den).astype(complex) if len(closed_den) > 1 else np.asarray([], dtype=complex)
    discrete_approx = DiscreteClosedLoopApproximation(
        open_loop_numerator=open_num,
        open_loop_denominator=open_den,
        closed_loop_denominator=closed_den,
        closed_loop_poles=closed_poles,
        stable=bool(np.all(np.abs(closed_poles) < 1.0)),
        integer_delay_samples=integer_delay,
        fractional_delay_samples=fractional_delay,
    )

    warnings: list[str] = []
    if small_signal.continuous_transfer.dc_gain * fm.gain_hz_per_pu <= 0.0:
        warnings.append(
            "PCMD-to-output low-frequency gain is non-positive; the firmware error polarity may create positive feedback at this operating point."
        )
    if min(fm.command_headroom_low, fm.command_headroom_high) < 0.03:
        warnings.append("PCMD operating point is within 3% of saturation; linear loop headroom is limited.")
    if not math.isclose(fm.frequency_hz, fsw, rel_tol=0.01):
        warnings.append(
            "The selected/custom FM LUT does not reproduce the plant operating frequency within 1%; PCMD linearization and plant work point are inconsistent."
        )
    if controller_kind(controller_config) in {ControllerKind.PI, ControllerKind.PIF}:
        if getattr(controller_config, "output_min", 0.0) != 0.0 or getattr(controller_config, "output_max", 1.0) != 1.0:
            warnings.append("Controller output limits differ from the requested normalized PCMD range 0..1.")
    warnings.append(
        "Linear Bode validity requires voltage-loop ownership: no current-limit min-selector takeover, burst, soft-start, saturation, OVP/UVP/OPP or hardware trip."
    )

    return DigitalLoopAnalysis(
        small_signal=small_signal,
        controller_config=controller_config,
        controller=controller,
        fm_lut=lut,
        fm_operating_point=fm,
        analog_sense=analog,
        adc_sampling=adc,
        command_timing=timing,
        frequencies_hz=frequencies,
        responses=responses,
        margins_minimum_delay=margins[DelayEnvelope.MINIMUM],
        margins_nominal_delay=margins[DelayEnvelope.NOMINAL],
        margins_maximum_delay=margins[DelayEnvelope.MAXIMUM],
        discrete_approximation=discrete_approx,
        warnings=tuple(warnings),
    )


def export_controller_c99(
    controller: DigitalTransferFunction,
    path: str | Path,
    *,
    function_name: str = "llc_voltage_controller_run",
    output_min: float = 0.0,
    output_max: float = 1.0,
) -> Path:
    """Export a generic Direct-Form-I controller matching the z^-1 coefficients."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    num = controller.numerator
    den = controller.denominator
    order_x = len(num) - 1
    order_y = len(den) - 1
    def c_float(value: float) -> str:
        text = f"{float(value):.9g}"
        if "e" not in text.lower() and "." not in text:
            text += ".0"
        return text + "f"

    lines = [
        "/* Auto-generated LLC digital controller. C99, Direct Form I. */",
        "#include <stddef.h>",
        "",
        f"#define LLC_CTRL_NX ({order_x + 1}u)",
        f"#define LLC_CTRL_NY ({order_y}u)",
        "",
        "typedef struct {",
        "    float x_hist[LLC_CTRL_NX];",
        "    float y_hist[(LLC_CTRL_NY > 0u) ? LLC_CTRL_NY : 1u];",
        "} llc_controller_state_t;",
        "",
        f"static const float llc_ctrl_b[LLC_CTRL_NX] = {{{', '.join(c_float(value) for value in num)}}};",
        f"static const float llc_ctrl_a[(LLC_CTRL_NY > 0u) ? LLC_CTRL_NY : 1u] = "
        + "{" + (", ".join(c_float(value) for value in den[1:]) if order_y else "0.0f") + "};",
        "",
        f"float {function_name}(llc_controller_state_t *state, float error)",
        "{",
        "    size_t i;",
        "    float output = 0.0f;",
        "",
        "    for(i = LLC_CTRL_NX - 1u; i > 0u; --i) {",
        "        state->x_hist[i] = state->x_hist[i - 1u];",
        "    }",
        "    state->x_hist[0] = error;",
        "    for(i = 0u; i < LLC_CTRL_NX; ++i) {",
        "        output += llc_ctrl_b[i] * state->x_hist[i];",
        "    }",
        "    for(i = 0u; i < LLC_CTRL_NY; ++i) {",
        "        output -= llc_ctrl_a[i] * state->y_hist[i];",
        "    }",
        f"    if(output > {c_float(output_max)}) output = {c_float(output_max)};",
        f"    if(output < {c_float(output_min)}) output = {c_float(output_min)};",
        "    if(LLC_CTRL_NY > 0u) {",
        "        for(i = LLC_CTRL_NY - 1u; i > 0u; --i) {",
        "            state->y_hist[i] = state->y_hist[i - 1u];",
        "        }",
        "        state->y_hist[0] = output;",
        "    }",
        "    return output;",
        "}",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def export_digital_loop_analysis(result: DigitalLoopAnalysis, directory: str | Path) -> dict[str, Path]:
    """Export loop curves, settings, FM LUT and controller C99 source."""
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)

    csv_path = output / "digital_loop_bode.csv"
    columns: dict[str, FloatArray] = {"frequency_hz": result.frequencies_hz}
    for name, response in result.responses.items():
        columns[f"{name}_magnitude_db"] = 20.0 * np.log10(np.maximum(np.abs(response), 1e-300))
        columns[f"{name}_phase_deg"] = np.unwrap(np.angle(response)) * 180.0 / math.pi
    import pandas as pd
    pd.DataFrame(columns).to_csv(csv_path, index=False)

    lut_path = output / "fm_lut.csv"
    value_name = "tbprd" if result.fm_lut.mode == FMLUTMode.PCMD_TO_TBPRD else "frequency_hz"
    pd.DataFrame({"pcmd": result.fm_lut.pcmd, value_name: result.fm_lut.values}).to_csv(lut_path, index=False)

    controller_path = export_controller_c99(
        result.controller,
        output / "llc_voltage_controller.c",
        output_min=getattr(result.controller_config, "output_min", 0.0),
        output_max=getattr(result.controller_config, "output_max", 1.0),
    )

    settings_path = output / "digital_loop_settings.json"
    payload = {
        "controller_kind": controller_kind(result.controller_config).value,
        "controller_config": asdict(result.controller_config),
        "controller_numerator_z_minus": result.controller.numerator.tolist(),
        "controller_denominator_z_minus": result.controller.denominator.tolist(),
        "controller_difference_equation": result.controller.difference_equation(),
        "fm_mode": result.fm_lut.mode.value,
        "fm_operating_point": asdict(result.fm_operating_point),
        "analog_sense": asdict(result.analog_sense),
        "adc_sampling": {
            **asdict(result.adc_sampling),
            "sample_offsets_s": result.adc_sampling.sample_offsets_s.tolist(),
            "eoc_delay_s": result.adc_sampling.eoc_delay_s,
        },
        "command_timing": asdict(result.command_timing),
        "margins_nominal": asdict(result.margins_nominal_delay),
        "margins_minimum_delay": asdict(result.margins_minimum_delay),
        "margins_maximum_delay": asdict(result.margins_maximum_delay),
        "discrete_closed_loop_poles": [
            {"real": float(value.real), "imag": float(value.imag), "magnitude": float(abs(value))}
            for value in result.discrete_approximation.closed_loop_poles
        ],
        "warnings": list(result.warnings),
    }
    settings_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "bode_csv": csv_path,
        "fm_lut_csv": lut_path,
        "controller_c99": controller_path,
        "settings_json": settings_path,
    }
