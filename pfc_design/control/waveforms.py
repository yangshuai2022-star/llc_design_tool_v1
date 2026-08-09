"""Multi-rate averaged line-cycle and work-point switching PFC simulation.

V7 fixes the most important modeling ambiguity from the early control-lab:
``iL`` is the rectified boost-inductor current magnitude while ``Iac`` is the
signed grid current.  The bus voltage is produced by the capacitor energy
balance, not by an imposed 2*fline ripple.  A local switching view is derived
from a selected point of the settled AC-cycle solution.
"""
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

from .config import ExternalSenseConfig, LoadModel, PFCControlLabConfig


FloatArray = NDArray[np.float64]


PFC_PWM_STATE_NAMES = {
    1: "positiveHalf",
    2: "negativeZeroCrossing1",
    3: "negativeZeroCrossing2",
    4: "negativeZeroCrossing3",
    5: "negativeHalf",
    6: "positiveZeroCrossing1",
    7: "positiveZeroCrossing2",
    8: "positiveZeroCrossing3",
}


@dataclass(frozen=True)
class PFCWaveformMetrics:
    input_voltage_rms_v: float
    input_current_rms_a: float
    input_current_peak_a: float
    real_input_power_w: float
    apparent_power_va: float
    power_factor: float
    displacement_factor: float
    distortion_factor: float
    current_thd_percent: float
    fundamental_current_rms_a: float
    harmonic_orders: tuple[int, ...]
    harmonic_current_rms_a: tuple[float, ...]
    bus_voltage_average_v: float
    bus_voltage_ripple_pp_v: float
    bus_capacitor_current_rms_a: float
    duty_min: float
    duty_max: float
    current_error_rms_a: float
    zero_cross_current_error_rms_a: float
    minimum_pulse_fraction: float


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
    source_time_s: float | None = None


class _SampledSenseChain:
    """Continuous analog poles followed by sampled ADC/digital filtering."""

    def __init__(self, config: ExternalSenseConfig, initial: float) -> None:
        self.config = config
        self.poles: list[float] = []
        if config.amplifier_bandwidth_hz > 0.0:
            self.poles.append(2.0 * math.pi * config.amplifier_bandwidth_hz)
        for resistance, capacitance in (
            (config.source_resistance_ohm, config.shunt_capacitance_f),
            (config.output_resistance_ohm, config.adc_capacitance_f),
            (config.second_resistance_ohm, config.second_capacitance_f),
        ):
            if resistance > 0.0 and capacitance > 0.0:
                self.poles.append(1.0 / (resistance * capacitance))
        self.states = [float(initial)] * len(self.poles)
        self.sample_period = 1.0 / config.timing.sample_rate_hz
        self.next_sample_s = 0.0
        delay_count = max(int(round(config.timing.nominal_latency_s / self.sample_period)), 0)
        self.delay_fifo = [float(initial)] * (delay_count + 1)
        self.previous_recursive = float(initial)
        self.digital = float(initial)
        self.output = float(initial)

    def step(self, value: float, time_s: float, dt: float) -> float:
        x = float(value)
        for idx, pole in enumerate(self.poles):
            decay = math.exp(-pole * dt)
            self.states[idx] = decay * self.states[idx] + (1.0 - decay) * x
            x = self.states[idx]
        # ADC output is sample-and-hold.  If an integration step crosses more
        # than one sample instant, catch up deterministically.
        while time_s + 0.5 * dt >= self.next_sample_s:
            self.delay_fifo.append(x)
            delayed = self.delay_fifo.pop(0)
            w = self.config.timing.recursive_previous_weight
            recursive = (1.0 - w) * delayed + w * self.previous_recursive
            self.previous_recursive = recursive
            alpha = self.config.timing.digital_filter.alpha
            self.digital += alpha * (recursive - self.digital)
            self.output = self.digital
            self.next_sample_s += self.sample_period
        return self.output


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

    def reset(self) -> None:
        self.error_prev = 0.0
        self.i_state = 0.0
        self.output_prev = 0.0
        self.y1 = self.y2 = 0.0
        self.x1 = self.x2 = 0.0

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
                -cfg.a1 * self.y1 - cfg.a2 * self.y2
                + cfg.b0 * error + cfg.b1 * self.x1 + cfg.b2 * self.x2
            )
            output = min(max(output, cfg.output_min), cfg.output_max)
            self.x2, self.x1 = self.x1, error
            self.y2, self.y1 = self.y1, output
            return float(output)
        raise TypeError(f"unsupported controller config: {type(cfg)!r}")


class _ZeroCrossMachine:
    """Control-rate approximation of the supplied TTPL commutation states."""

    def __init__(self) -> None:
        self.state = 1
        self.sign_filtered = 1
        self.count = 0

    def step(self, vac: float) -> tuple[int, bool]:
        if vac > 7.5:
            self.sign_filtered = 1
        elif vac < -7.5:
            self.sign_filtered = 0
        reset_pi = False
        s = self.state
        if s == 1 and (vac < 15.0 or self.sign_filtered == 0):
            self.state, self.count, reset_pi = 2, 0, True
        elif s == 2 and vac <= 0.0:
            self.state, self.count = 3, 0
        elif s == 3 and vac < -15.0:
            self.state, self.count = 4, 0
        elif s == 4:
            self.count += 1; reset_pi = True
            if self.count >= 10 and self.sign_filtered == 0:
                self.state, self.count = 5, 0
        elif s == 5 and (vac > -15.0 or self.sign_filtered == 1):
            self.state, self.count, reset_pi = 6, 0, True
        elif s == 6 and vac >= 0.0:
            self.state, self.count = 7, 0
        elif s == 7 and vac > 15.0:
            self.state, self.count, reset_pi = 8, 0, True
        elif s == 8:
            self.count += 1; reset_pi = True
            if self.count >= 10 and self.sign_filtered == 1:
                self.state, self.count = 1, 0
        return self.state, reset_pi

    @property
    def zero_cross_active(self) -> bool:
        return self.state not in (1, 5)

    @property
    def lf_state(self) -> float:
        return 1.0 if self.state == 1 else (-1.0 if self.state == 5 else 0.0)

    @property
    def zc_deadband_fraction(self) -> float:
        if self.state in (4, 8):
            soft_duty = min(0.05 + 0.075 * self.count, 1.0)
            return max(1.0 - soft_duty, 0.0)
        return 1.0 if self.zero_cross_active else 0.0


def _harmonic_metrics(
    time_s: FloatArray,
    voltage: FloatArray,
    current: FloatArray,
    line_hz: float,
    max_harmonic: int = 25,
):
    n = len(time_s)
    theta = 2.0 * math.pi * line_hz * (time_s - time_s[0])

    def phasor(x: FloatArray, harmonic: int = 1) -> complex:
        a = 2.0 / n * float(np.sum(x * np.cos(harmonic * theta)))
        b = 2.0 / n * float(np.sum(x * np.sin(harmonic * theta)))
        return complex(a, -b) / math.sqrt(2.0)

    v1 = phasor(voltage, 1)
    i1 = phasor(current, 1)
    v_rms = float(np.sqrt(np.mean(voltage**2)))
    i_rms = float(np.sqrt(np.mean(current**2)))
    p = float(np.mean(voltage * current))
    s = v_rms * i_rms
    pf = p / max(s, 1e-12)
    displacement = math.cos(np.angle(v1) - np.angle(i1))
    i1_rms = abs(i1)
    orders = tuple(range(1, max_harmonic + 1))
    harmonics = tuple(float(abs(phasor(current, h))) for h in orders)
    # Compute THD from explicit integer harmonics rather than the residual
    # ``sqrt(Irms^2-I1^2)``.  The residual formula is very sensitive to a
    # one-sample line-cycle window mismatch and can collapse to zero when the
    # DFT estimate of I1 is a few ppm above the direct RMS value.  Explicit
    # harmonics are also the quantity shown in the PF/THD panel.
    harmonic_rss = math.sqrt(sum(value * value for value in harmonics[1:]))
    thd = harmonic_rss / max(i1_rms, 1e-12)
    distortion = i1_rms / max(math.sqrt(i1_rms * i1_rms + harmonic_rss * harmonic_rss), 1e-12)
    return v_rms, i_rms, p, s, pf, displacement, distortion, thd, i1_rms, orders, harmonics


def _last_cycle_slice(time: FloatArray, line_hz: float) -> slice:
    dt = float(np.mean(np.diff(time)))
    count = max(int(round((1.0 / line_hz) / dt)), 4)
    return slice(max(len(time) - count, 0), len(time))


def simulate_pfc_line_cycle(config: PFCControlLabConfig) -> PFCLineCycleWaveforms:
    """Run a multi-rate digital controller around an averaged CCM boost plant."""
    config.validate()
    stage, fw = config.power_stage, config.firmware
    # Integrate faster than the 50 kHz control ISR so current and bus state are
    # not numerically tied to controller execution.  10 substeps/current tick is
    # a practical default and keeps full-cycle runs fast.
    integration_rate = max(getattr(config, "waveform_integration_rate_hz", 500.0e3), fw.current_loop_rate_hz * 4.0)
    dt = 1.0 / integration_rate
    total_time = config.waveform_line_cycles / stage.line_frequency_hz
    count = int(round(total_time * integration_rate)) + 1
    time = np.arange(count, dtype=float) * dt

    vac_chain = _SampledSenseChain(config.vac_sense, 0.0)
    current_chain = _SampledSenseChain(config.current_sense, 0.0)
    vbus_chain = _SampledSenseChain(config.vbus_sense, stage.bus_voltage_v)

    required_gcmd = stage.output_power_w / (stage.efficiency * stage.vin_rms_v**2)
    raw_vloop0 = required_gcmd * fw.vac_rms_feedforward_gain * stage.vin_rms_v**2
    if fw.vff_bypass:
        raw_vloop0 = required_gcmd / fw.vac_rms_feedforward_gain
    raw_vloop0 = min(max(raw_vloop0, config.voltage_controller.output_min), config.voltage_controller.output_max)
    voltage_ctrl = _FirmwareController(config.voltage_controller, raw_vloop0)
    current_ctrl = _FirmwareController(config.current_controller, 0.0)
    zc = _ZeroCrossMachine()

    names = (
        "vac", "vac_measured", "vac_abs_measured", "vac_rms_estimate",
        "i_ref", "i_ref_abs", "i_inductor", "i_inductor_signed",
        "i_measured", "i_measured_signed", "i_input_signed", "current_error",
        "vbus", "vbus_measured", "vbus_ripple", "voltage_error",
        "gcmd", "vloop", "vloop_raw", "duty_ff", "duty_pi", "duty_total",
        "duty_unclamped", "effective_duty_min", "minimum_pulse_active", "indu_comp",
        "input_power", "load_current", "boost_output_current", "bus_cap_current",
        "pwm_state", "pwm_state_code", "zero_cross_active", "lf_gate_state",
        "zc_deadband_fraction", "current_pi_reset_strobe", "line_angle_deg",
        "current_update_strobe", "amc_update_strobe", "voltage_update_strobe",
    )
    data = {name: np.zeros(count, dtype=float) for name in names}

    vbus = stage.bus_voltage_v
    current = 0.0
    vac_rms_sq = stage.vin_rms_v**2
    vac_rms = stage.vin_rms_v
    raw_vloop = raw_vloop0
    vloop = raw_vloop0 / max((fw.vac_rms_feedforward_gain * stage.vin_rms_v)**2, 1e-12)
    gcmd = required_gcmd
    i_ref = 0.0
    duty_ff = 0.5
    duty_pi = 0.0
    indu_comp = fw.indu_comp_min
    duty_command = 0.5
    current_error = 0.0
    voltage_error = 0.0

    effective_duty_min = max(stage.duty_min, stage.minimum_effective_pulse_s * stage.switching_frequency_hz)
    current_period = 1.0 / fw.current_loop_rate_hz
    amc_period = 1.0 / fw.amc_rate_hz
    voltage_period = 1.0 / fw.voltage_loop_rate_hz
    next_current = next_amc = next_voltage = 0.0

    if stage.load_model == LoadModel.RESISTIVE:
        load_resistance = stage.bus_voltage_v**2 / stage.output_power_w
    else:
        load_resistance = math.inf

    for idx, t in enumerate(time):
        omega_t = 2.0 * math.pi * stage.line_frequency_hz * t
        vac = stage.line_peak_v * math.sin(omega_t)
        vac_measured = vac_chain.step(vac, t, dt)
        current_measured = current_chain.step(current, t, dt)
        vbus_measured = vbus_chain.step(vbus, t, dt)
        current_strobe = amc_strobe = voltage_strobe = reset_strobe = 0.0

        if t + 0.5 * dt >= next_voltage:
            voltage_strobe = 1.0
            vac_rms_sq += fw.vac_rms_lpf_alpha * (vac_measured**2 - vac_rms_sq)
            vac_rms = math.sqrt(max(vac_rms_sq, 1.0))
            voltage_error = stage.bus_voltage_v - vbus_measured
            raw_vloop = voltage_ctrl.step(voltage_error)
            if fw.vff_bypass:
                vloop = raw_vloop
            else:
                denom = max((fw.vac_rms_feedforward_gain * vac_rms)**2, 1.0)
                vloop = raw_vloop / denom
            gcmd = min(max(vloop * fw.vac_rms_feedforward_gain, 0.0), fw.gcmd_max_a_per_v)
            next_voltage += voltage_period

        if t + 0.5 * dt >= next_amc:
            amc_strobe = 1.0
            vac_abs = abs(vac_measured)
            i_ref = gcmd * vac_abs
            duty_ff = 1.0 - vac_abs / max(stage.bus_voltage_v, 1.0)
            duty_ff = min(max(duty_ff, stage.duty_min), stage.duty_max)
            indu_comp = min(max(fw.indu_comp_gain * i_ref, fw.indu_comp_min), fw.indu_comp_max)
            next_amc += amc_period

        if t + 0.5 * dt >= next_current:
            current_strobe = 1.0
            state_code, reset = zc.step(vac_measured)
            if reset:
                current_ctrl.reset(); reset_strobe = 1.0
            current_error = i_ref - abs(current_measured)
            duty_pi = current_ctrl.step(current_error)
            duty_unclamped = duty_ff + duty_pi * indu_comp
            duty_command = min(max(duty_unclamped, effective_duty_min), stage.duty_max)
            # Firmware parks ZC1/ZC2 at minimum duty. ZC3 keeps command but
            # repeatedly clears PI while dead-band soft-starts.
            if state_code in (2, 3, 6, 7):
                duty_command = effective_duty_min
            next_current += current_period
        else:
            state_code = zc.state
            duty_unclamped = duty_ff + duty_pi * indu_comp

        vac_abs = abs(vac)
        di_dt = (
            vac_abs - (1.0 - duty_command) * vbus
            - stage.equivalent_series_resistance_ohm * current
        ) / stage.boost_inductance_h
        current = max(current + di_dt * dt, 0.0)

        if stage.load_model == LoadModel.CONSTANT_POWER:
            load_current = stage.output_power_w / max(vbus, 25.0)
        else:
            load_current = vbus / load_resistance
        boost_output_current = (1.0 - duty_command) * current
        bus_cap_current = boost_output_current - load_current
        vbus += bus_cap_current / stage.bus_capacitance_f * dt
        vbus = min(max(vbus, 25.0), 1.6 * stage.bus_voltage_v)

        sign = 1.0 if vac >= 0.0 else -1.0
        iac = sign * current
        minimum_active = 1.0 if duty_unclamped <= effective_duty_min + 1e-12 else 0.0
        values = {
            "vac": vac, "vac_measured": vac_measured, "vac_abs_measured": abs(vac_measured),
            "vac_rms_estimate": vac_rms, "i_ref": sign * i_ref, "i_ref_abs": i_ref,
            "i_inductor": current, "i_inductor_signed": sign * current,
            "i_measured": current_measured, "i_measured_signed": sign * current_measured,
            "i_input_signed": iac, "current_error": current_error,
            "vbus": vbus, "vbus_measured": vbus_measured, "vbus_ripple": vbus-stage.bus_voltage_v,
            "voltage_error": voltage_error, "gcmd": gcmd, "vloop": vloop, "vloop_raw": raw_vloop,
            "duty_ff": duty_ff, "duty_pi": duty_pi, "duty_total": duty_command,
            "duty_unclamped": duty_unclamped, "effective_duty_min": effective_duty_min,
            "minimum_pulse_active": minimum_active, "indu_comp": indu_comp,
            "input_power": vac * iac, "load_current": load_current,
            "boost_output_current": boost_output_current, "bus_cap_current": bus_cap_current,
            "pwm_state": 1.0 if vac >= 0.0 else -1.0, "pwm_state_code": float(state_code),
            "zero_cross_active": 1.0 if zc.zero_cross_active else 0.0,
            "lf_gate_state": zc.lf_state, "zc_deadband_fraction": zc.zc_deadband_fraction,
            "current_pi_reset_strobe": reset_strobe,
            "line_angle_deg": (360.0 * stage.line_frequency_hz * t) % 360.0,
            "current_update_strobe": current_strobe,
            "amc_update_strobe": amc_strobe, "voltage_update_strobe": voltage_strobe,
        }
        for key, value in values.items():
            data[key][idx] = value

    sl = _last_cycle_slice(time, stage.line_frequency_hz)
    t_last = time[sl]
    vac_last = data["vac"][sl]
    iac_last = data["i_input_signed"][sl]
    (v_rms, i_rms, pin, apparent, pf, displacement, distortion, thd,
     i1_rms, orders, harmonics) = _harmonic_metrics(t_last, vac_last, iac_last, stage.line_frequency_hz)
    vbus_last = data["vbus"][sl]
    cap_last = data["bus_cap_current"][sl]
    err_last = data["current_error"][sl]
    duty_last = data["duty_total"][sl]
    zc_mask = data["zero_cross_active"][sl] > 0.5
    zc_error_rms = float(np.sqrt(np.mean(err_last[zc_mask]**2))) if np.any(zc_mask) else 0.0
    min_pulse_fraction = float(np.mean(data["minimum_pulse_active"][sl] > 0.5))
    metrics = PFCWaveformMetrics(
        input_voltage_rms_v=v_rms,
        input_current_rms_a=i_rms,
        input_current_peak_a=float(np.max(np.abs(iac_last))),
        real_input_power_w=pin,
        apparent_power_va=apparent,
        power_factor=pf,
        displacement_factor=displacement,
        distortion_factor=distortion,
        current_thd_percent=100.0 * thd,
        fundamental_current_rms_a=i1_rms,
        harmonic_orders=orders,
        harmonic_current_rms_a=harmonics,
        bus_voltage_average_v=float(np.mean(vbus_last)),
        bus_voltage_ripple_pp_v=float(np.ptp(vbus_last)),
        bus_capacitor_current_rms_a=float(np.sqrt(np.mean(cap_last**2))),
        duty_min=float(np.min(duty_last)),
        duty_max=float(np.max(duty_last)),
        current_error_rms_a=float(np.sqrt(np.mean(err_last**2))),
        zero_cross_current_error_rms_a=zc_error_rms,
        minimum_pulse_fraction=min_pulse_fraction,
    )
    warnings: list[str] = []
    if metrics.power_factor < 0.98:
        warnings.append("Power factor below 0.98; inspect current-loop tuning, Vac sensing and zero-crossing behavior.")
    if metrics.current_thd_percent > 8.0:
        warnings.append("Current THD above 8%; inspect current bandwidth, zero crossing and minimum pulse constraint.")
    if abs(metrics.bus_voltage_average_v - stage.bus_voltage_v) > 0.05 * stage.bus_voltage_v:
        warnings.append("Bus voltage did not settle within 5% of command over the selected run.")

    units = {key: "" for key in names}
    for key in ("vac","vac_measured","vac_abs_measured","vac_rms_estimate","vbus","vbus_measured","vbus_ripple","voltage_error"):
        units[key] = "V"
    for key in ("i_ref","i_ref_abs","i_inductor","i_inductor_signed","i_measured","i_measured_signed","i_input_signed","current_error","load_current","boost_output_current","bus_cap_current"):
        units[key] = "A"
    units.update({"gcmd":"A/V","input_power":"W","line_angle_deg":"deg","pwm_state_code":"state"})
    return PFCLineCycleWaveforms(time, data, units, metrics, tuple(warnings))


def _nearest_line_cycle_index(
    line_cycle: PFCLineCycleWaveforms, angle_deg: float, line_hz: float
) -> int:
    """Locate an electrical angle only inside the settled final AC period."""
    angles = np.asarray(line_cycle.signals["line_angle_deg"], dtype=float)
    final_slice = _last_cycle_slice(np.asarray(line_cycle.time_s, dtype=float), line_hz)
    start = int(final_slice.start or 0)
    final_angles = angles[final_slice]
    if final_angles.size == 0:
        raise ValueError("line-cycle result contains no samples")
    delta = np.abs(((final_angles - angle_deg + 180.0) % 360.0) - 180.0)
    return start + int(np.argmin(delta))


def build_pfc_switching_waveforms(
    config: PFCControlLabConfig,
    *,
    line_cycle: PFCLineCycleWaveforms | None = None,
    line_angle_deg: float | None = None,
    samples: int | None = None,
    cycles: int | None = None,
    samples_per_cycle: int | None = None,
) -> PFCSwitchingWaveforms:
    """Expand local PWM cycles from a selected settled AC-cycle work point."""
    config.validate()
    stage = config.power_stage
    angle = stage.line_angle_deg if line_angle_deg is None else float(line_angle_deg) % 360.0
    cycle_count = int(cycles if cycles is not None else (1 if samples is not None else config.switching_cycles))
    spp = int(samples_per_cycle if samples_per_cycle is not None else (samples if samples is not None else config.switching_samples_per_cycle))
    if cycle_count < 1 or spp < 100:
        raise ValueError("invalid switching waveform resolution")

    source_time = None
    if line_cycle is not None:
        idx = _nearest_line_cycle_index(line_cycle, angle, stage.line_frequency_hz)
        s = line_cycle.signals
        vin = float(s["vac"][idx])
        vbus = float(s["vbus"][idx])
        i_avg = float(s["i_inductor"][idx])
        duty = float(s["duty_total"][idx])
        lf_state = float(s.get("lf_gate_state", np.zeros_like(s["vac"]))[idx])
        state_code = float(s.get("pwm_state_code", np.zeros_like(s["vac"]))[idx])
        source_time = float(line_cycle.time_s[idx])
    else:
        theta = math.radians(angle)
        vin = stage.line_peak_v * math.sin(theta)
        vbus = stage.bus_voltage_v
        i_avg = stage.input_conductance_a_per_v * abs(vin)
        effective_duty_min = max(stage.duty_min, stage.minimum_effective_pulse_s * stage.switching_frequency_hz)
        duty = min(max(1.0 - abs(vin) / max(vbus, 1.0), effective_duty_min), stage.duty_max)
        lf_state = 1.0 if vin >= 0.0 else -1.0
        state_code = 1.0 if vin >= 0.0 else 5.0

    fsw = stage.switching_frequency_hz
    period = 1.0 / fsw
    total = cycle_count * spp
    time = np.arange(total, dtype=float) * period / spp
    tau = np.mod(time, period)
    deadtime = min(getattr(stage, "deadtime_s", 100e-9), 0.2 * period)
    on_end = max(duty * period - deadtime, deadtime)
    hf_high = ((tau >= deadtime) & (tau < on_end)).astype(float)
    low_start = min(duty * period + deadtime, period - deadtime)
    hf_low = ((tau >= low_start) & (tau < period - deadtime)).astype(float)
    effective_on = tau < duty * period
    switch_node = np.where(effective_on, 0.0, vbus)
    vin_abs = abs(vin)
    # Reconstruct current by *continuous* switching-time integration.  The old
    # implementation rebuilt the same triangular ripple from ``tau`` on every
    # cycle, which introduced an artificial current jump at each period boundary
    # whenever the selected AC workpoint had a small non-zero dI/dt.  That jump
    # looked like current-loop instability even though it was only a plotting
    # artifact.  Start from a valley around the averaged workpoint and integrate
    # the actual commanded switching state sample by sample.
    slope_on_at_avg = (
        vin_abs - stage.equivalent_series_resistance_ohm * i_avg
    ) / stage.boost_inductance_h
    ripple_pp_est = max(slope_on_at_avg * duty * period, 0.0)
    i_state = max(i_avg - 0.5 * ripple_pp_est, 0.0)
    iL = np.zeros_like(time)
    step_dt = period / spp
    resistance = stage.equivalent_series_resistance_ohm
    inductance = stage.boost_inductance_h
    for sample in range(total):
        source_v = vin_abs if effective_on[sample] else (vin_abs - vbus)
        if resistance > 1e-12:
            # Exact integration of dI/dt = (V - R*I)/L over one display sample.
            decay = math.exp(-resistance * step_dt / inductance)
            steady = source_v / resistance
            i_state = steady + (i_state - steady) * decay
        else:
            i_state += source_v / inductance * step_dt
        i_state = max(i_state, 0.0)
        iL[sample] = i_state
    output_current = np.where(effective_on, 0.0, iL)
    load_current = stage.output_power_w / max(vbus, 25.0)
    bus_cap_current = output_current - load_current
    sign = 1.0 if vin >= 0.0 else -1.0
    signals = {
        "hf_high_gate": hf_high,
        "hf_low_gate": hf_low,
        "lf_polarity_gate": np.full_like(time, lf_state),
        "pwm_state_code": np.full_like(time, state_code),
        "duty_command": np.full_like(time, duty),
        "vac_instantaneous": np.full_like(time, vin),
        "vbus_workpoint": np.full_like(time, vbus),
        "switch_node_voltage": switch_node,
        "inductor_voltage": vin_abs - switch_node,
        "inductor_current": iL,
        "inductor_current_average": np.full_like(time, i_avg),
        "current_deviation": iL - i_avg,
        "input_current_signed": sign * iL,
        "boost_output_current": output_current,
        "high_side_current": hf_high * iL,
        "low_side_current": hf_low * iL,
        "bus_cap_current": bus_cap_current,
        "cycle_index": np.floor(time / period),
    }
    units = {
        "hf_high_gate":"logic","hf_low_gate":"logic","lf_polarity_gate":"state","pwm_state_code":"state",
        "duty_command":"pu","vac_instantaneous":"V","vbus_workpoint":"V","switch_node_voltage":"V",
        "inductor_voltage":"V","inductor_current":"A","inductor_current_average":"A","current_deviation":"A","input_current_signed":"A","boost_output_current":"A",
        "high_side_current":"A","low_side_current":"A","bus_cap_current":"A","cycle_index":"index",
    }
    return PFCSwitchingWaveforms(time, signals, units, angle, fsw, source_time)


__all__ = [
    "PFCLineCycleWaveforms",
    "PFCSwitchingWaveforms",
    "PFCWaveformMetrics",
    "PFC_PWM_STATE_NAMES",
    "build_pfc_switching_waveforms",
    "simulate_pfc_line_cycle",
]
