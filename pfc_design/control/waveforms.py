"""Averaged line-cycle and local switching-cycle PFC waveform simulation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from llc_design.control.digital_loop import (
    PIControllerConfig,
    PIFControllerConfig,
    TwoP2ZControllerConfig,
)

from .config import ExternalSenseConfig, PFCControlLabConfig


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PFCWaveformMetrics:
    input_voltage_rms_v: float
    input_current_rms_a: float
    input_current_peak_a: float
    real_input_power_w: float
    power_factor: float
    displacement_factor: float
    current_thd_percent: float
    bus_voltage_average_v: float
    bus_voltage_ripple_pp_v: float
    bus_capacitor_current_rms_a: float
    duty_min: float
    duty_max: float
    current_error_rms_a: float


@dataclass(frozen=True)
class PFCLineCycleWaveforms:
    time_s: FloatArray
    signals: Mapping[str, FloatArray]
    units: Mapping[str, str]
    metrics: PFCWaveformMetrics
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class PFCSwitchingWaveforms:
    time_s: FloatArray
    signals: Mapping[str, FloatArray]
    units: Mapping[str, str]
    line_angle_deg: float
    switching_frequency_hz: float


class _FirstOrderChain:
    def __init__(self, config: ExternalSenseConfig, initial: float) -> None:
        self.config = config
        self.states: list[float] = []
        self.poles: list[float] = []
        for resistance, capacitance in (
            (config.source_resistance_ohm, config.shunt_capacitance_f),
            (config.output_resistance_ohm, config.adc_capacitance_f),
            (config.second_resistance_ohm, config.second_capacitance_f),
        ):
            if resistance > 0.0 and capacitance > 0.0:
                self.poles.append(1.0 / (resistance * capacitance))
                self.states.append(float(initial))
        if config.amplifier_bandwidth_hz > 0.0:
            self.poles.insert(0, 2.0 * math.pi * config.amplifier_bandwidth_hz)
            self.states.insert(0, float(initial))
        delay_samples = int(round(config.timing.nominal_latency_s * config.timing.sample_rate_hz))
        self.delay = [float(initial)] * max(delay_samples + 1, 1)
        self.previous = float(initial)
        self.digital = float(initial)

    def step(self, value: float, dt: float) -> float:
        x = float(value)
        for index, pole_rad_s in enumerate(self.poles):
            decay = math.exp(-pole_rad_s * dt)
            self.states[index] = decay * self.states[index] + (1.0 - decay) * x
            x = self.states[index]
        self.delay.append(x)
        delayed = self.delay.pop(0)
        w = self.config.timing.recursive_previous_weight
        recursive = (1.0 - w) * delayed + w * self.previous
        self.previous = recursive
        alpha = self.config.timing.digital_filter.alpha
        self.digital += alpha * (recursive - self.digital)
        return self.digital


class _FirmwareController:
    def __init__(self, config, initial_output: float = 0.0) -> None:
        self.config = config
        self.error_prev = 0.0
        self.i_state = 0.0
        self.output_prev = float(initial_output)
        self.y1 = float(initial_output)
        self.y2 = float(initial_output)
        self.x1 = 0.0
        self.x2 = 0.0
        if isinstance(config, (PIControllerConfig, PIFControllerConfig)):
            self.i_state = initial_output / max(config.kp, 1e-12)

    def step(self, error: float) -> float:
        cfg = self.config
        if isinstance(cfg, (PIControllerConfig, PIFControllerConfig)):
            ki2 = cfg.sample_time_s / (2.0 * cfg.ti_s)
            i_new = self.i_state + ki2 * (error + self.error_prev)
            out_raw = cfg.kp * (error + i_new)
            out_sat = min(max(out_raw, cfg.output_min), cfg.output_max)
            if not ((out_raw > cfg.output_max and error > 0.0)
                    or (out_raw < cfg.output_min and error < 0.0)):
                self.i_state = i_new
            if isinstance(cfg, PIFControllerConfig):
                alpha = cfg.alpha
                output = (1.0 - alpha) * self.output_prev + alpha * out_sat
                self.output_prev = output
            else:
                output = out_sat
            self.error_prev = error
            return float(output)
        if isinstance(cfg, TwoP2ZControllerConfig):
            output = (
                -cfg.a1 * self.y1
                -cfg.a2 * self.y2
                + cfg.b0 * error
                + cfg.b1 * self.x1
                + cfg.b2 * self.x2
            )
            output = min(max(output, cfg.output_min), cfg.output_max)
            self.x2, self.x1 = self.x1, error
            self.y2, self.y1 = self.y1, output
            return float(output)
        raise TypeError(f"unsupported controller config: {type(cfg)!r}")


def _fundamental_metrics(time_s: FloatArray, voltage: FloatArray, current: FloatArray, line_hz: float):
    n = len(time_s)
    dt = float(np.mean(np.diff(time_s)))
    theta = 2.0 * math.pi * line_hz * time_s
    # RMS phasor coefficients: x(t)=a*cos+b*sin, RMS=sqrt(a^2+b^2)/sqrt(2).
    def phasor(x: FloatArray) -> complex:
        a = 2.0 / n * float(np.sum(x * np.cos(theta)))
        b = 2.0 / n * float(np.sum(x * np.sin(theta)))
        return complex(a, -b) / math.sqrt(2.0)
    v1 = phasor(voltage)
    i1 = phasor(current)
    displacement = math.cos(np.angle(v1) - np.angle(i1))
    current_rms = float(np.sqrt(np.mean(current**2)))
    i1_rms = abs(i1)
    thd = math.sqrt(max(current_rms**2 - i1_rms**2, 0.0)) / max(i1_rms, 1e-12)
    real_power = float(np.mean(voltage * current))
    voltage_rms = float(np.sqrt(np.mean(voltage**2)))
    pf = real_power / max(voltage_rms * current_rms, 1e-12)
    return voltage_rms, current_rms, real_power, pf, displacement, thd


def simulate_pfc_line_cycle(config: PFCControlLabConfig) -> PFCLineCycleWaveforms:
    """Simulate the supplied firmware's nested loops with an averaged boost plant.

    This is an engineering control/waveform model, not a transistor-level model.
    The 50 kHz current loop, 25 kHz AMC and 10 kHz bus-voltage loop execute at
    their actual integer decimation ratios.  External sensor RC networks and
    nominal ADC/PWM delays are included.
    """

    config.validate()
    stage = config.power_stage
    fw = config.firmware
    dt = 1.0 / fw.current_loop_rate_hz
    total_time = config.waveform_line_cycles / stage.line_frequency_hz
    sample_count = int(round(total_time / dt))
    time = np.arange(sample_count, dtype=float) * dt

    vac_chain = _FirstOrderChain(config.vac_sense, 0.0)
    current_chain = _FirstOrderChain(config.current_sense, stage.current_reference_a)
    vbus_chain = _FirstOrderChain(config.vbus_sense, stage.bus_voltage_v)

    required_gcmd = stage.output_power_w / (stage.efficiency * stage.vin_rms_v**2)
    if fw.vff_bypass:
        initial_vloop = required_gcmd / fw.vac_rms_feedforward_gain
    else:
        initial_vloop = required_gcmd * fw.vac_rms_feedforward_gain * stage.vin_rms_v**2
    initial_vloop = min(max(initial_vloop,
                            config.voltage_controller.output_min),
                        config.voltage_controller.output_max)
    voltage_ctrl = _FirmwareController(config.voltage_controller, initial_vloop)
    current_ctrl = _FirmwareController(config.current_controller, 0.0)

    names = (
        "vac", "vac_measured", "vac_abs_measured", "vac_rms_estimate",
        "i_ref", "i_ref_abs", "i_inductor", "i_inductor_signed",
        "i_measured", "i_measured_signed", "i_input_signed", "current_error",
        "vbus", "vbus_measured", "vbus_ripple", "voltage_error",
        "gcmd", "vloop", "duty_ff", "duty_pi", "duty_total",
        "effective_duty_min", "minimum_pulse_active", "indu_comp",
        "input_power", "load_current", "boost_output_current",
        "bus_cap_current", "pwm_state", "line_angle_deg",
        "current_update_strobe", "amc_update_strobe", "voltage_update_strobe",
    )
    data = {name: np.zeros(sample_count, dtype=float) for name in names}

    vbus = stage.bus_voltage_v
    current = stage.current_reference_a
    vloop = initial_vloop
    gcmd = required_gcmd
    i_ref = stage.current_reference_a
    duty_ff = stage.ideal_duty
    indu_comp = stage.indu_comp
    vac_rms_sq = stage.vin_rms_v**2
    vac_rms = stage.vin_rms_v
    voltage_error = 0.0
    current_div = max(int(round(fw.current_loop_rate_hz / fw.amc_rate_hz)), 1)
    voltage_div = max(int(round(fw.current_loop_rate_hz / fw.voltage_loop_rate_hz)), 1)
    effective_duty_min = max(
        stage.duty_min,
        stage.minimum_effective_pulse_s * stage.switching_frequency_hz,
    )

    for index, t in enumerate(time):
        voltage_update_strobe = 0.0
        amc_update_strobe = 0.0
        vac = stage.line_peak_v * math.sin(2.0 * math.pi * stage.line_frequency_hz * t)
        vac_measured = vac_chain.step(vac, dt)
        current_measured = current_chain.step(current, dt)
        vbus_measured = vbus_chain.step(vbus, dt)

        if index % voltage_div == 0:
            voltage_update_strobe = 1.0
            vac_rms_sq += fw.vac_rms_lpf_alpha * (vac_measured**2 - vac_rms_sq)
            vac_rms = math.sqrt(max(vac_rms_sq, 1.0))
            voltage_error = stage.bus_voltage_v - vbus_measured
            vloop = voltage_ctrl.step(voltage_error)
            if fw.vff_bypass:
                gcmd = vloop * fw.vac_rms_feedforward_gain
            else:
                divisor = max((fw.vac_rms_feedforward_gain * vac_rms)**2, 1.0)
                gcmd = vloop / divisor * fw.vac_rms_feedforward_gain
            gcmd = min(max(gcmd, 0.0), fw.gcmd_max_a_per_v)

        if index % current_div == 0:
            amc_update_strobe = 1.0
            vac_abs_measured = abs(vac_measured)
            i_ref = gcmd * vac_abs_measured
            duty_ff = 1.0 - vac_abs_measured / max(stage.bus_voltage_v, 1.0)
            duty_ff = min(max(duty_ff, effective_duty_min), stage.duty_max)
            indu_comp = min(max(fw.indu_comp_gain * i_ref,
                                fw.indu_comp_min), fw.indu_comp_max)

        current_error = i_ref - abs(current_measured)
        duty_pi = current_ctrl.step(current_error)
        duty_unclamped = duty_ff + duty_pi * indu_comp
        duty = min(max(duty_unclamped, effective_duty_min), stage.duty_max)
        minimum_pulse_active = 1.0 if duty_unclamped <= effective_duty_min else 0.0

        vac_abs = abs(vac)
        di_dt = (
            vac_abs - (1.0 - duty) * vbus
            - stage.equivalent_series_resistance_ohm * current
        ) / stage.boost_inductance_h
        current = max(current + di_dt * dt, 0.0)
        load_current = stage.output_power_w / max(vbus, 25.0)
        output_stage_current = (1.0 - duty) * current
        bus_cap_current = output_stage_current - load_current
        vbus += bus_cap_current / stage.bus_capacitance_f * dt
        vbus = min(max(vbus, 25.0), 1.5 * stage.bus_voltage_v)

        sign = 1.0 if vac >= 0.0 else -1.0
        input_current = sign * current
        data["vac"][index] = vac
        data["vac_measured"][index] = vac_measured
        data["vac_abs_measured"][index] = abs(vac_measured)
        data["vac_rms_estimate"][index] = vac_rms
        data["i_ref"][index] = sign * i_ref
        data["i_ref_abs"][index] = i_ref
        data["i_inductor"][index] = current
        data["i_inductor_signed"][index] = sign * current
        data["i_measured"][index] = current_measured
        data["i_measured_signed"][index] = sign * current_measured
        data["i_input_signed"][index] = input_current
        data["current_error"][index] = current_error
        data["vbus"][index] = vbus
        data["vbus_measured"][index] = vbus_measured
        data["vbus_ripple"][index] = vbus - stage.bus_voltage_v
        data["voltage_error"][index] = voltage_error
        data["gcmd"][index] = gcmd
        data["vloop"][index] = vloop
        data["duty_ff"][index] = duty_ff
        data["duty_pi"][index] = duty_pi
        data["duty_total"][index] = duty
        data["effective_duty_min"][index] = effective_duty_min
        data["minimum_pulse_active"][index] = minimum_pulse_active
        data["indu_comp"][index] = indu_comp
        data["input_power"][index] = vac * input_current
        data["load_current"][index] = load_current
        data["boost_output_current"][index] = output_stage_current
        data["bus_cap_current"][index] = bus_cap_current
        data["pwm_state"][index] = 1.0 if vac >= 0.0 else -1.0
        data["line_angle_deg"][index] = (360.0 * stage.line_frequency_hz * t) % 360.0
        data["current_update_strobe"][index] = 1.0
        data["amc_update_strobe"][index] = amc_update_strobe
        data["voltage_update_strobe"][index] = voltage_update_strobe

    samples_per_line = int(round(fw.current_loop_rate_hz / stage.line_frequency_hz))
    analysis_slice = slice(sample_count - samples_per_line, sample_count)
    t_last = time[analysis_slice] - time[analysis_slice][0]
    vac_last = data["vac"][analysis_slice]
    current_last = data["i_input_signed"][analysis_slice]
    v_rms, i_rms, pin, pf, displacement, thd = _fundamental_metrics(
        t_last, vac_last, current_last, stage.line_frequency_hz)
    vbus_last = data["vbus"][analysis_slice]
    cap_current_last = data["bus_cap_current"][analysis_slice]
    error_last = data["current_error"][analysis_slice]
    duty_last = data["duty_total"][analysis_slice]
    metrics = PFCWaveformMetrics(
        input_voltage_rms_v=v_rms,
        input_current_rms_a=i_rms,
        input_current_peak_a=float(np.max(np.abs(current_last))),
        real_input_power_w=pin,
        power_factor=pf,
        displacement_factor=displacement,
        current_thd_percent=100.0 * thd,
        bus_voltage_average_v=float(np.mean(vbus_last)),
        bus_voltage_ripple_pp_v=float(np.ptp(vbus_last)),
        bus_capacitor_current_rms_a=float(np.sqrt(np.mean(cap_current_last**2))),
        duty_min=float(np.min(duty_last)),
        duty_max=float(np.max(duty_last)),
        current_error_rms_a=float(np.sqrt(np.mean(error_last**2))),
    )
    warnings: list[str] = []
    if metrics.power_factor < 0.98:
        warnings.append("Simulated power factor is below 0.98; inspect Vac filtering and current-loop tuning.")
    if metrics.current_thd_percent > 8.0:
        warnings.append("Simulated current THD exceeds 8%; inspect zero crossing, minimum pulse and current bandwidth.")
    if abs(metrics.bus_voltage_average_v - stage.bus_voltage_v) > 0.05 * stage.bus_voltage_v:
        warnings.append("Bus voltage did not settle within 5% of its command in the selected simulation window.")

    units = {
        "vac": "V", "vac_measured": "V", "vac_abs_measured": "V",
        "vac_rms_estimate": "V", "i_ref": "A", "i_ref_abs": "A",
        "i_inductor": "A", "i_inductor_signed": "A",
        "i_measured": "A", "i_measured_signed": "A",
        "i_input_signed": "A", "current_error": "A",
        "vbus": "V", "vbus_measured": "V", "vbus_ripple": "V",
        "voltage_error": "V", "gcmd": "A/V", "vloop": "controller unit",
        "duty_ff": "pu", "duty_pi": "pu", "duty_total": "pu",
        "effective_duty_min": "pu", "minimum_pulse_active": "logic",
        "indu_comp": "pu", "input_power": "W", "load_current": "A",
        "boost_output_current": "A", "bus_cap_current": "A",
        "pwm_state": "state", "line_angle_deg": "deg",
        "current_update_strobe": "logic", "amc_update_strobe": "logic",
        "voltage_update_strobe": "logic",
    }
    return PFCLineCycleWaveforms(
        time_s=time,
        signals=data,
        units=units,
        metrics=metrics,
        warnings=tuple(warnings),
    )


def build_pfc_switching_waveforms(
    config: PFCControlLabConfig,
    *,
    line_angle_deg: float | None = None,
    samples: int | None = None,
    cycles: int | None = None,
    samples_per_cycle: int | None = None,
) -> PFCSwitchingWaveforms:
    """Build multiple local PWM periods at a selected AC-line angle.

    ``samples`` is retained as a backward-compatible alias for
    ``samples_per_cycle``.  The GUI defaults to two switching periods, while the
    full AC-period view is generated separately by :func:`simulate_pfc_line_cycle`.
    """

    config.validate()
    stage = config.power_stage
    angle = stage.line_angle_deg if line_angle_deg is None else float(line_angle_deg)
    if cycles is None:
        # Backward compatibility: the legacy ``samples=...`` API represented one
        # switching period.  The GUI/config path omits ``samples`` and therefore
        # uses the new multi-cycle default.
        cycle_count = 1 if samples is not None else config.switching_cycles
    else:
        cycle_count = int(cycles)
    if samples_per_cycle is None:
        samples_per_cycle = samples if samples is not None else config.switching_samples_per_cycle
    samples_per_cycle = int(samples_per_cycle)
    if cycle_count < 1:
        raise ValueError("at least one switching cycle is required")
    if samples_per_cycle < 100:
        raise ValueError("at least 100 samples per switching cycle are required")

    theta = math.radians(angle)
    vin = stage.line_peak_v * math.sin(theta)
    vin_abs = abs(vin)
    gcmd = stage.input_conductance_a_per_v
    i_avg = gcmd * vin_abs
    effective_duty_min = max(
        stage.duty_min,
        stage.minimum_effective_pulse_s * stage.switching_frequency_hz,
    )
    duty = min(max(1.0 - vin_abs / stage.bus_voltage_v,
                   effective_duty_min), stage.duty_max)
    period = 1.0 / stage.switching_frequency_hz
    total_samples = cycle_count * samples_per_cycle
    time = np.arange(total_samples, dtype=float) * period / samples_per_cycle
    local_time = np.mod(time, period)
    on_time = duty * period
    gate = (local_time < on_time).astype(float)
    slope_on = vin_abs / stage.boost_inductance_h
    slope_off = (vin_abs - stage.bus_voltage_v) / stage.boost_inductance_h
    ripple_pp = slope_on * on_time
    valley = max(i_avg - 0.5 * ripple_pp, 0.0)
    current = np.where(
        local_time < on_time,
        valley + slope_on * local_time,
        valley + ripple_pp + slope_off * (local_time - on_time),
    )
    switch_node = np.where(gate > 0.5, 0.0, stage.bus_voltage_v)
    inductor_voltage = vin_abs - switch_node
    hf_low_gate = 1.0 - gate
    lf_gate = np.full_like(time, 1.0 if vin >= 0.0 else 0.0)
    output_current = np.where(gate > 0.5, 0.0, current)
    sign = 1.0 if vin >= 0.0 else -1.0
    load_current = stage.output_power_w / stage.bus_voltage_v
    bus_cap_current = output_current - load_current
    signals = {
        "hf_high_gate": gate,
        "hf_low_gate": hf_low_gate,
        "lf_polarity_gate": lf_gate,
        "duty_command": np.full_like(time, duty),
        "vac_instantaneous": np.full_like(time, vin),
        "switch_node_voltage": switch_node,
        "inductor_voltage": inductor_voltage,
        "inductor_current": current,
        "input_current_signed": sign * current,
        "boost_output_current": output_current,
        "high_side_current": gate * current,
        "low_side_current": (1.0 - gate) * current,
        "bus_cap_current": bus_cap_current,
        "cycle_index": np.floor(time / period),
    }
    units = {
        "hf_high_gate": "logic", "hf_low_gate": "logic",
        "lf_polarity_gate": "logic", "duty_command": "pu",
        "vac_instantaneous": "V", "switch_node_voltage": "V",
        "inductor_voltage": "V", "inductor_current": "A",
        "input_current_signed": "A", "boost_output_current": "A",
        "high_side_current": "A", "low_side_current": "A",
        "bus_cap_current": "A", "cycle_index": "index",
    }
    return PFCSwitchingWaveforms(
        time_s=time,
        signals=signals,
        units=units,
        line_angle_deg=angle,
        switching_frequency_hz=stage.switching_frequency_hz,
    )


__all__ = [
    "PFCLineCycleWaveforms",
    "PFCSwitchingWaveforms",
    "PFCWaveformMetrics",
    "build_pfc_switching_waveforms",
    "simulate_pfc_line_cycle",
]
