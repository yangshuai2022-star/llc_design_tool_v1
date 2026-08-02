"""External analog filters, ADC sampling, calibration and delay models."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .config import ExternalSenseConfig


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def _first_order_lowpass(s: ComplexArray, pole_hz: float) -> ComplexArray:
    if pole_hz <= 0.0 or not math.isfinite(pole_hz):
        return np.ones_like(s, dtype=complex)
    return 1.0 / (1.0 + s / (2.0 * math.pi * pole_hz))


def _rc_pole_hz(resistance_ohm: float, capacitance_f: float) -> float:
    if resistance_ohm <= 0.0 or capacitance_f <= 0.0:
        return math.inf
    return 1.0 / (2.0 * math.pi * resistance_ohm * capacitance_f)


@dataclass(frozen=True)
class SenseFrequencyResponse:
    frequencies_hz: FloatArray
    raw_analog: ComplexArray
    calibrated_analog: ComplexArray
    adc_aperture: ComplexArray
    multi_soc_recursive: ComplexArray
    digital_filter: ComplexArray
    pure_delay: ComplexArray
    total: ComplexArray


@dataclass(frozen=True)
class SenseChainSummary:
    name: str
    raw_dc_gain: float
    adc_codes_per_unit: float
    amplifier_pole_hz: float
    source_rc_pole_hz: float
    output_rc_pole_hz: float
    second_rc_pole_hz: float
    nominal_latency_s: float
    gain_50hz_db: float
    phase_50hz_deg: float
    gain_60hz_db: float
    phase_60hz_deg: float
    gain_100hz_db: float
    gain_120hz_db: float
    gain_switching_db: float


def analog_response(
    config: ExternalSenseConfig,
    frequencies_hz: Sequence[float] | FloatArray,
    *,
    calibrated: bool = True,
) -> ComplexArray:
    """Evaluate external sensor/amplifier/RC frequency response."""

    f = np.asarray(frequencies_hz, dtype=float)
    s = 1j * 2.0 * math.pi * f
    response = np.full(f.shape, config.raw_dc_gain, dtype=complex)
    response *= _first_order_lowpass(s, config.amplifier_bandwidth_hz)
    response *= _first_order_lowpass(
        s, _rc_pole_hz(config.source_resistance_ohm, config.shunt_capacitance_f)
    )
    response *= _first_order_lowpass(
        s, _rc_pole_hz(config.output_resistance_ohm, config.adc_capacitance_f)
    )
    response *= _first_order_lowpass(
        s, _rc_pole_hz(config.second_resistance_ohm, config.second_capacitance_f)
    )
    if calibrated and config.normalize_to_engineering_units:
        response /= config.raw_dc_gain
    return np.asarray(response, dtype=complex)


def adc_aperture_response(
    config: ExternalSenseConfig,
    frequencies_hz: Sequence[float] | FloatArray,
) -> ComplexArray:
    """ADC acquisition aperture response including half-window phase delay."""

    f = np.asarray(frequencies_hz, dtype=float)
    ta = config.timing.acquisition_time_s
    if ta <= 0.0:
        return np.ones_like(f, dtype=complex)
    # np.sinc(x)=sin(pi*x)/(pi*x); a rectangular average of width Ta is
    # sinc(f*Ta)*exp(-j*pi*f*Ta).
    return np.sinc(f * ta) * np.exp(-1j * math.pi * f * ta)


def multi_soc_recursive_response(
    config: ExternalSenseConfig,
    frequencies_hz: Sequence[float] | FloatArray,
) -> ComplexArray:
    """Response of equally weighted SOC samples plus previous-output feedback."""

    f = np.asarray(frequencies_hz, dtype=float)
    timing = config.timing
    w = timing.recursive_previous_weight
    fresh_weight = (1.0 - w) / timing.soc_count
    sample_sum = np.zeros_like(f, dtype=complex)
    for index in range(timing.soc_count):
        delay = index * timing.soc_spacing_s
        sample_sum += np.exp(-1j * 2.0 * math.pi * f * delay)
    numerator = fresh_weight * sample_sum
    if w <= 0.0:
        return numerator
    z_inv = np.exp(-1j * 2.0 * math.pi * f * timing.sample_time_s)
    return numerator / (1.0 - w * z_inv)


def digital_filter_response(
    config: ExternalSenseConfig,
    frequencies_hz: Sequence[float] | FloatArray,
) -> ComplexArray:
    f = np.asarray(frequencies_hz, dtype=float)
    alpha = config.timing.digital_filter.alpha
    if alpha >= 1.0:
        return np.ones_like(f, dtype=complex)
    z_inv = np.exp(-1j * 2.0 * math.pi * f * config.timing.sample_time_s)
    return alpha / (1.0 - (1.0 - alpha) * z_inv)


def pure_delay_response(
    config: ExternalSenseConfig,
    frequencies_hz: Sequence[float] | FloatArray,
) -> ComplexArray:
    f = np.asarray(frequencies_hz, dtype=float)
    timing = config.timing
    delay = (
        timing.conversion_time_s
        + timing.computation_delay_s
        + timing.pwm_update_delay_s
    )
    response = np.exp(-1j * 2.0 * math.pi * f * delay)
    if timing.include_zero_order_hold:
        ts = timing.sample_time_s
        response *= np.sinc(f * ts) * np.exp(-1j * math.pi * f * ts)
    return response


def sense_frequency_response(
    config: ExternalSenseConfig,
    frequencies_hz: Sequence[float] | FloatArray,
) -> SenseFrequencyResponse:
    config.validate()
    f = np.asarray(frequencies_hz, dtype=float)
    raw = analog_response(config, f, calibrated=False)
    calibrated = analog_response(config, f, calibrated=True)
    aperture = adc_aperture_response(config, f)
    multi_soc = multi_soc_recursive_response(config, f)
    digital = digital_filter_response(config, f)
    delay = pure_delay_response(config, f)
    total = calibrated * aperture * multi_soc * digital * delay
    return SenseFrequencyResponse(
        frequencies_hz=f,
        raw_analog=raw,
        calibrated_analog=calibrated,
        adc_aperture=aperture,
        multi_soc_recursive=multi_soc,
        digital_filter=digital,
        pure_delay=delay,
        total=total,
    )


def _interp_complex_log(frequencies: FloatArray, response: ComplexArray, query: float) -> complex:
    q = float(np.clip(query, frequencies[0], frequencies[-1]))
    logf = np.log10(frequencies)
    real = np.interp(math.log10(q), logf, response.real)
    imag = np.interp(math.log10(q), logf, response.imag)
    return complex(real, imag)


def _gain_phase(value: complex) -> tuple[float, float]:
    return (
        20.0 * math.log10(max(abs(value), 1e-300)),
        math.degrees(math.atan2(value.imag, value.real)),
    )


def summarize_sense_chain(
    config: ExternalSenseConfig,
    *,
    switching_frequency_hz: float = 50.0e3,
) -> SenseChainSummary:
    points = np.unique(np.asarray([
        0.1, 50.0, 60.0, 100.0, 120.0, switching_frequency_hz,
        min(0.49 * config.timing.sample_rate_hz, switching_frequency_hz),
    ], dtype=float))
    points = points[points > 0.0]
    response = sense_frequency_response(config, points).total
    gp50 = _gain_phase(_interp_complex_log(points, response, 50.0))
    gp60 = _gain_phase(_interp_complex_log(points, response, 60.0))
    gp100 = _gain_phase(_interp_complex_log(points, response, 100.0))
    gp120 = _gain_phase(_interp_complex_log(points, response, 120.0))
    gpsw = _gain_phase(_interp_complex_log(points, response, switching_frequency_hz))
    return SenseChainSummary(
        name=config.name,
        raw_dc_gain=config.raw_dc_gain,
        adc_codes_per_unit=config.adc_codes_per_unit,
        amplifier_pole_hz=config.amplifier_bandwidth_hz,
        source_rc_pole_hz=_rc_pole_hz(
            config.source_resistance_ohm, config.shunt_capacitance_f),
        output_rc_pole_hz=_rc_pole_hz(
            config.output_resistance_ohm, config.adc_capacitance_f),
        second_rc_pole_hz=_rc_pole_hz(
            config.second_resistance_ohm, config.second_capacitance_f),
        nominal_latency_s=config.timing.nominal_latency_s,
        gain_50hz_db=gp50[0],
        phase_50hz_deg=gp50[1],
        gain_60hz_db=gp60[0],
        phase_60hz_deg=gp60[1],
        gain_100hz_db=gp100[0],
        gain_120hz_db=gp120[0],
        gain_switching_db=gpsw[0],
    )


__all__ = [
    "SenseChainSummary",
    "SenseFrequencyResponse",
    "adc_aperture_response",
    "analog_response",
    "digital_filter_response",
    "multi_soc_recursive_response",
    "pure_delay_response",
    "sense_frequency_response",
    "summarize_sense_chain",
]
