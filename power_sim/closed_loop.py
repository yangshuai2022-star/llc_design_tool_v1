"""V9 digital closed-loop co-simulation scheduler.

The scheduler is deliberately power-stage agnostic.  A plant only needs to
advance for a requested duration under a held LLC switching frequency and
provide a measured output.  This lets the exact same Sampler -> C(z) -> FM
chain run against deterministic test plants or future pluggable power-stage
backends without changing the controller/sampler/modulator contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import math
from typing import Protocol, Callable

import numpy as np
from scipy import signal

from power_control_tools.models import DigitalTransferFunction
from llc_design.control.digital_loop import calculate_stability_margins

from .digital_control import (
    ControllerLimitConfig,
    DigitalTransferRuntime,
    LLCFMConfig,
    LLCFMRuntime,
    SamplerConfig,
    SamplerRuntime,
)


class ClosedLoopPlant(Protocol):
    @property
    def time_s(self) -> float: ...
    def advance(self, duration_s: float, switching_frequency_hz: float) -> None: ...
    def output_voltage_v(self) -> float: ...
    def set_bus_voltage_v(self, value: float) -> None: ...
    def set_load_fraction(self, value: float) -> None: ...


@dataclass(frozen=True)
class StepProfile:
    initial: float
    step_time_s: float | None = None
    final: float | None = None

    def value_at(self, time_s: float) -> float:
        if self.step_time_s is not None and self.final is not None and time_s >= self.step_time_s:
            return float(self.final)
        return float(self.initial)

    def events(self) -> tuple[tuple[float, float], ...]:
        if self.step_time_s is None or self.final is None:
            return ()
        return ((float(self.step_time_s), float(self.final)),)


@dataclass(frozen=True)
class ClosedLoopTiming:
    duration_s: float
    computation_delay_s: float = 0.0
    pwm_update_delay_s: float = 0.0

    def validate(self, sample_time_s: float) -> None:
        if self.duration_s <= 0.0:
            raise ValueError("duration_s must be positive")
        if self.computation_delay_s < 0.0 or self.pwm_update_delay_s < 0.0:
            raise ValueError("control delays cannot be negative")
        # Delays longer than one sample are supported by the event queue; warn at
        # the analysis layer rather than silently truncating them.
        if sample_time_s <= 0.0:
            raise ValueError("sample_time_s must be positive")


@dataclass(frozen=True)
class ClosedLoopScenario:
    reference_v: StepProfile
    bus_voltage_v: StepProfile | None = None
    load_fraction: StepProfile | None = None


@dataclass(frozen=True)
class ClosedLoopSample:
    time_s: float
    reference_v: float
    plant_output_v: float
    feedback_v: float
    error_v: float
    controller_raw: float
    controller_output: float
    controller_saturated: bool
    frequency_command_hz: float
    frequency_actual_hz: float
    frequency_applied_hz: float
    tbprd: int | None
    modulator_saturated: bool
    modulator_quantized: bool


@dataclass(frozen=True)
class ClosedLoopDiagnostics:
    converged: bool
    regulation_error_v: float
    output_std_v: float
    overshoot_percent: float
    settling_time_s: float | None
    controller_saturation_fraction: float
    modulator_saturation_fraction: float
    limit_cycle_detected: bool
    dominant_oscillation_hz: float | None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClosedLoopResult:
    samples: tuple[ClosedLoopSample, ...]
    diagnostics: ClosedLoopDiagnostics
    metadata: dict[str, object] = field(default_factory=dict)

    def array(self, name: str) -> np.ndarray:
        return np.asarray([getattr(s, name) for s in self.samples], dtype=float)


@dataclass(frozen=True)
class LinearClosedLoopAnalysis:
    frequencies_hz: np.ndarray
    open_loop: np.ndarray
    closed_loop: np.ndarray
    sensitivity: np.ndarray
    closed_loop_poles: np.ndarray
    stable: bool
    max_pole_radius: float
    phase_margin_deg: float | None
    gain_margin_db: float | None
    crossover_hz: float | None
    warnings: tuple[str, ...] = ()


def _tf_freq_response(tf: DigitalTransferFunction, frequencies_hz: np.ndarray) -> np.ndarray:
    d = tf.normalized()
    w = 2.0 * np.pi * frequencies_hz / d.sample_rate_hz
    _, h = signal.freqz(np.asarray(d.b), np.asarray(d.a), worN=w)
    return np.asarray(h, dtype=complex)


def analyze_linear_closed_loop(
    controller: DigitalTransferFunction,
    plant: DigitalTransferFunction,
    *,
    modulator_gain: float = 1.0,
    sense_filter: DigitalTransferFunction | None = None,
    delay_samples: int = 0,
    points: int = 2000,
) -> LinearClosedLoopAnalysis:
    c = controller.normalized(); g = plant.normalized()
    if not math.isclose(c.sample_rate_hz, g.sample_rate_hz, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("controller and plant sample rates must match")
    if sense_filter is not None and not math.isclose(c.sample_rate_hz, sense_filter.sample_rate_hz, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("sense filter sample rate must match controller")
    if delay_samples < 0:
        raise ValueError("delay_samples cannot be negative")

    fs = c.sample_rate_hz
    frequencies = np.geomspace(max(1e-3, fs * 1e-5), fs * 0.49, int(points))
    h = _tf_freq_response(c, frequencies) * _tf_freq_response(g, frequencies) * float(modulator_gain)
    if sense_filter is not None:
        h *= _tf_freq_response(sense_filter, frequencies)
    if delay_samples:
        h *= np.exp(-1j * 2.0 * np.pi * frequencies / fs * delay_samples)
    margins = calculate_stability_margins(frequencies, h)

    num = np.convolve(np.asarray(c.b), np.asarray(g.b)) * float(modulator_gain)
    den = np.convolve(np.asarray(c.a), np.asarray(g.a))
    if sense_filter is not None:
        num = np.convolve(num, np.asarray(sense_filter.b))
        den = np.convolve(den, np.asarray(sense_filter.a))
    if delay_samples:
        num = np.concatenate((np.zeros(delay_samples), num))
    n = max(len(num), len(den))
    num = np.pad(num, (0, n - len(num)))
    den = np.pad(den, (0, n - len(den)))
    characteristic = den + num
    if abs(characteristic[0]) > 1e-30:
        characteristic = characteristic / characteristic[0]
    poles = np.roots(characteristic).astype(complex) if len(characteristic) > 1 else np.asarray([], dtype=complex)
    radius = float(np.max(np.abs(poles))) if poles.size else 0.0
    open_loop = h
    closed_loop = h / (1.0 + h)
    sensitivity = 1.0 / (1.0 + h)
    warnings: list[str] = []
    dc_loop = (np.sum(c.b) / np.sum(c.a) if abs(np.sum(c.a)) > 1e-15 else np.inf)
    dc_plant = (np.sum(g.b) / np.sum(g.a) if abs(np.sum(g.a)) > 1e-15 else np.inf)
    if np.isfinite(dc_loop) and np.isfinite(dc_plant) and dc_loop * dc_plant * modulator_gain < 0.0:
        warnings.append("Low-frequency loop sign is negative inside a 1+L feedback convention; verify controller/modulator/plant polarity.")
    if delay_samples >= 1:
        warnings.append(f"Linear model includes {delay_samples} full sample(s) of command delay.")
    return LinearClosedLoopAnalysis(
        frequencies, open_loop, closed_loop, sensitivity, poles, bool(np.all(np.abs(poles) < 1.0)), radius,
        margins.phase_margin_deg, margins.gain_margin_db, margins.critical_gain_crossover_hz, tuple(warnings)
    )


def llc_small_signal_to_digital_plant(small_signal) -> DigitalTransferFunction:
    """Convert the existing LLC ZOH plant into the Control Tools TF contract.

    This keeps the linear-stability pre-check on the exact same discrete plant
    already used by the LLC small-signal page.
    """
    plant = small_signal.discrete_plant
    return DigitalTransferFunction(
        tuple(float(v) for v in np.asarray(plant.numerator, dtype=float).reshape(-1)),
        tuple(float(v) for v in np.asarray(plant.denominator, dtype=float).reshape(-1)),
        1.0 / float(plant.sample_time_s),
        "LLC Gvf ZOH(z)",
        source="llc_design.small_signal",
    ).normalized()


class FirstOrderLLCPlant:
    """Deterministic inverse-gain LLC surrogate used only to verify the scheduler.

    The sign dVo/df < 0 is intentional and lets tests catch
    modulator/control-polarity mistakes in the generic digital scheduler.
    """
    def __init__(
        self,
        *,
        nominal_output_v: float = 53.0,
        nominal_frequency_hz: float = 100_000.0,
        frequency_gain_v_per_hz: float = -5e-4,
        time_constant_s: float = 8e-4,
        nominal_bus_v: float = 400.0,
        line_gain_v_per_v: float = 0.03,
        load_droop_v_per_pu: float = 1.0,
    ):
        self.nominal_output_v = float(nominal_output_v)
        self.nominal_frequency_hz = float(nominal_frequency_hz)
        self.frequency_gain_v_per_hz = float(frequency_gain_v_per_hz)
        self.time_constant_s = float(time_constant_s)
        self.nominal_bus_v = float(nominal_bus_v)
        self.line_gain_v_per_v = float(line_gain_v_per_v)
        self.load_droop_v_per_pu = float(load_droop_v_per_pu)
        self._time = 0.0
        self._vout = self.nominal_output_v
        self._vbus = self.nominal_bus_v
        self._load = 1.0

    @property
    def time_s(self) -> float: return self._time
    def output_voltage_v(self) -> float: return self._vout
    def set_bus_voltage_v(self, value: float) -> None: self._vbus = float(value)
    def set_load_fraction(self, value: float) -> None: self._load = float(value)

    def advance(self, duration_s: float, switching_frequency_hz: float) -> None:
        dt = float(duration_s)
        if dt < -1e-15:
            raise ValueError("plant cannot advance backwards")
        if dt <= 0.0:
            return
        target = (
            self.nominal_output_v
            + self.frequency_gain_v_per_hz * (float(switching_frequency_hz) - self.nominal_frequency_hz)
            + self.line_gain_v_per_v * (self._vbus - self.nominal_bus_v)
            - self.load_droop_v_per_pu * (self._load - 1.0)
        )
        alpha = math.exp(-dt / self.time_constant_s)
        self._vout = target + (self._vout - target) * alpha
        self._time += dt

    def discrete_transfer(self, sample_rate_hz: float) -> DigitalTransferFunction:
        fs = float(sample_rate_hz)
        a = math.exp(-1.0 / (fs * self.time_constant_s))
        # input is frequency [Hz], output is voltage [V]
        b = self.frequency_gain_v_per_hz * (1.0 - a)
        return DigitalTransferFunction((0.0, b), (1.0, -a), fs, "LLC surrogate Gvf(z)")


def _diagnose(samples: tuple[ClosedLoopSample, ...], sample_rate_hz: float, final_reference: float) -> ClosedLoopDiagnostics:
    if not samples:
        return ClosedLoopDiagnostics(False, math.inf, math.inf, 0.0, None, 0.0, 0.0, False, None, ("No control samples were generated.",))
    y = np.asarray([s.plant_output_v for s in samples], dtype=float)
    t = np.asarray([s.time_s for s in samples], dtype=float)
    tail_n = max(5, len(y) // 5)
    tail = y[-tail_n:]
    mean = float(np.mean(tail)); std = float(np.std(tail))
    err = float(final_reference - mean)
    scale = max(abs(final_reference), 1e-9)
    peak = float(np.max(y)) if final_reference >= y[0] else float(np.min(y))
    overshoot = max(0.0, (peak - final_reference) / scale * 100.0) if final_reference >= y[0] else max(0.0, (final_reference - peak) / scale * 100.0)
    band = 0.02 * scale
    settling = None
    bad = np.flatnonzero(np.abs(y - final_reference) > band)
    if bad.size == 0:
        settling = 0.0
    elif bad[-1] + 1 < len(t):
        settling = float(t[bad[-1] + 1])
    ctrl_sat = float(np.mean([s.controller_saturated for s in samples]))
    mod_sat = float(np.mean([s.modulator_saturated for s in samples]))
    f_tail = np.asarray([s.frequency_actual_hz for s in samples[-tail_n:]], dtype=float)
    rounded = np.round(f_tail, decimals=6)
    unique = np.unique(rounded)
    limit_cycle = bool(1 < len(unique) <= 4 and np.std(f_tail) > 1e-9)
    osc = None
    if len(tail) >= 16 and std > 1e-8:
        detrended = tail - np.mean(tail)
        spec = np.abs(np.fft.rfft(detrended))
        freqs = np.fft.rfftfreq(len(detrended), d=1.0 / sample_rate_hz)
        if len(spec) > 1:
            idx = int(np.argmax(spec[1:]) + 1)
            osc = float(freqs[idx])
    converged = bool(abs(err) <= 0.01 * scale and std <= 0.002 * scale and not mod_sat)
    notes: list[str] = []
    if limit_cycle:
        notes.append("Final switching-frequency command toggles among a small number of timer levels; timer-quantization limit cycle is possible.")
    if mod_sat > 0.0:
        notes.append("Frequency modulator reached Fmin/Fmax during the run.")
    if ctrl_sat > 0.0:
        notes.append("Controller output saturation occurred during the run.")
    return ClosedLoopDiagnostics(converged, err, std, float(overshoot), settling, ctrl_sat, mod_sat, limit_cycle, osc, tuple(notes))


def run_closed_loop(
    plant: ClosedLoopPlant,
    *,
    controller: DigitalTransferFunction,
    controller_limits: ControllerLimitConfig,
    sampler: SamplerConfig,
    modulator: LLCFMConfig,
    timing: ClosedLoopTiming,
    scenario: ClosedLoopScenario,
) -> ClosedLoopResult:
    sampler_rt = SamplerRuntime(sampler)
    sampler_rt.reset(plant.output_voltage_v())
    controller_rt = DigitalTransferRuntime(controller, controller_limits)
    mod_rt = LLCFMRuntime(modulator)
    timing.validate(sampler.sample_time_s)
    if not math.isclose(controller.sample_rate_hz, sampler.sample_rate_hz, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("controller and sampler rates must match")

    if scenario.bus_voltage_v is not None:
        plant.set_bus_voltage_v(scenario.bus_voltage_v.initial)
    if scenario.load_fraction is not None:
        plant.set_load_fraction(scenario.load_fraction.initial)

    initial_mod = mod_rt.map(0.0)
    current_frequency = initial_mod.actual_frequency_hz
    event_queue: list[tuple[float, int, str, object]] = []
    sequence = 0
    ts = sampler.sample_time_s
    k = 0
    while True:
        t = sampler.sample_phase_s + k * ts
        if t > timing.duration_s + 1e-15:
            break
        heapq.heappush(event_queue, (t, sequence, "sample", None)); sequence += 1
        k += 1
    if scenario.bus_voltage_v is not None:
        for t, value in scenario.bus_voltage_v.events():
            heapq.heappush(event_queue, (t, sequence, "bus", value)); sequence += 1
    if scenario.load_fraction is not None:
        for t, value in scenario.load_fraction.events():
            heapq.heappush(event_queue, (t, sequence, "load", value)); sequence += 1

    records: list[ClosedLoopSample] = []
    pending_by_sample: dict[int, tuple[float, float, float, bool, object]] = {}
    sample_index = 0
    while event_queue:
        event_time, _, kind, payload = heapq.heappop(event_queue)
        if event_time > timing.duration_s + 1e-15:
            continue
        if event_time > plant.time_s:
            plant.advance(event_time - plant.time_s, current_frequency)
        if kind == "bus":
            plant.set_bus_voltage_v(float(payload)); continue
        if kind == "load":
            plant.set_load_fraction(float(payload)); continue
        if kind == "apply":
            idx, fm_step = payload
            current_frequency = fm_step.actual_frequency_hz
            # record fields were created at sample time; actuator value is the
            # command scheduled by that sample even if application is delayed.
            continue
        if kind != "sample":
            continue

        measured = plant.output_voltage_v()
        feedback = sampler_rt.sample(measured)
        reference = scenario.reference_v.value_at(event_time)
        error = reference - feedback
        cstep = controller_rt.step(error)
        fmstep = mod_rt.map(cstep.output)
        apply_time = event_time + timing.computation_delay_s + timing.pwm_update_delay_s
        heapq.heappush(event_queue, (apply_time, sequence, "apply", (sample_index, fmstep))); sequence += 1
        records.append(ClosedLoopSample(
            event_time, reference, measured, feedback, error, cstep.raw_output, cstep.output, cstep.saturated,
            fmstep.frequency_command_hz, fmstep.actual_frequency_hz, current_frequency, fmstep.tbprd, fmstep.saturated, fmstep.quantized,
        ))
        sample_index += 1

    if plant.time_s < timing.duration_s:
        plant.advance(timing.duration_s - plant.time_s, current_frequency)
    result_samples = tuple(records)
    final_ref = scenario.reference_v.value_at(timing.duration_s)
    diagnostics = _diagnose(result_samples, sampler.sample_rate_hz, final_ref)
    return ClosedLoopResult(result_samples, diagnostics, {
        "sample_rate_hz": sampler.sample_rate_hz,
        "sample_phase_s": sampler.sample_phase_s,
        "computation_delay_s": timing.computation_delay_s,
        "pwm_update_delay_s": timing.pwm_update_delay_s,
        "modulator_mode": modulator.mode.value,
    })
