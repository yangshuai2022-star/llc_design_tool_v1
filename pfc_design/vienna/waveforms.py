"""Three-phase Vienna PFC average-cycle and local switching waveform solvers.

The V7 model intentionally keeps the control architecture close to the supplied
single-phase PFC firmware while using a topology-specific Vienna plant/modulator:

* slow DC-voltage loop -> conductance command;
* 25 kHz three-phase reference generation ia*=G*va, ib*=G*vb, ic*=G*vc;
* three 50 kHz stationary-frame current controllers;
* optional third-harmonic common-mode injection and inductor-voltage-drop FF;
* split-bus midpoint balance as an auxiliary common-mode modulation offset;
* zero-state duty D0 = 1-|m| for the Vienna center switch.

The line-cycle solver is an averaged three-wire model.  The switching solver is
not independent: it reconstructs local switching states from one selected
workpoint in the final settled line cycle.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from pfc_design.control.waveforms import (
    _FirmwareController,
    _SampledSenseChain,
    _harmonic_metrics,
)
from .config import ViennaControlLabConfig

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ViennaWaveformMetrics:
    phase_current_rms_a: tuple[float, float, float]
    phase_current_thd_percent: tuple[float, float, float]
    phase_power_factor: tuple[float, float, float]
    phase_displacement_factor: tuple[float, float, float]
    phase_distortion_factor: tuple[float, float, float]
    total_input_power_w: float
    overall_power_factor: float
    current_unbalance_percent: float
    bus_voltage_average_v: float
    bus_voltage_ripple_pp_v: float
    midpoint_delta_average_v: float
    midpoint_delta_pp_v: float
    midpoint_current_rms_a: float
    harmonic_orders: tuple[int, ...]
    phase_a_harmonic_rms_a: tuple[float, ...]
    phase_b_harmonic_rms_a: tuple[float, ...]
    phase_c_harmonic_rms_a: tuple[float, ...]


@dataclass(frozen=True)
class ViennaLineCycleWaveforms:
    time_s: FloatArray
    signals: Mapping[str, FloatArray]
    units: Mapping[str, str]
    metrics: ViennaWaveformMetrics
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class ViennaSwitchingWaveforms:
    time_s: FloatArray
    signals: Mapping[str, FloatArray]
    units: Mapping[str, str]
    line_angle_deg: float
    switching_frequency_hz: float
    source_time_s: float | None = None


def _phase_values(vpk: float, wt: float) -> tuple[float, float, float]:
    return (
        vpk * math.sin(wt),
        vpk * math.sin(wt - 2.0 * math.pi / 3.0),
        vpk * math.sin(wt + 2.0 * math.pi / 3.0),
    )


def _sector(angle_deg: float) -> int:
    return int((angle_deg % 360.0) // 60.0) + 1


def _last_cycle_slice(time_s: FloatArray, line_hz: float) -> slice:
    if len(time_s) < 2:
        return slice(0, len(time_s))
    start_time = float(time_s[-1]) - 1.0 / line_hz
    start = int(np.searchsorted(time_s, start_time, side="left"))
    return slice(max(start, 0), len(time_s))


def _third_harmonic_common_mode(vabc: NDArray[np.float64], enabled: bool) -> float:
    """Carrier-based common-mode injection used by Vienna modulation.

    The injected value is the midpoint between the instantaneous maximum and
    minimum phase voltages.  It is common to all three phases and therefore
    cancels from the three-wire line-current dynamics, while increasing usable
    modulation range and changing neutral-point state allocation.
    """
    if not enabled:
        return 0.0
    return 0.5 * (float(np.max(vabc)) + float(np.min(vabc)))


def _apply_minimum_zero_state_pulse(
    modulation: NDArray[np.float64],
    *,
    modulation_limit: float,
    minimum_pulse_fraction: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Convert signed Vienna modulation to center-switch zero-state duty.

    ``m`` represents the signed active-state fraction.  The center switch is ON
    during the zero state, hence ``D0 = 1-|m|``.  A minimum-pulse constraint is
    applied to both a very narrow active state and a very narrow zero state.
    The third returned array is a 0/1 diagnostic flag.
    """
    m = np.clip(np.asarray(modulation, dtype=float), -modulation_limit, modulation_limit)
    sign = np.where(m >= 0.0, 1.0, -1.0)
    active = np.abs(m)
    min_frac = max(float(minimum_pulse_fraction), 0.0)
    pulse_active = np.zeros_like(active)
    if min_frac > 0.0:
        narrow_active = (active > 0.0) & (active < min_frac)
        if np.any(narrow_active):
            active[narrow_active] = 0.0
            pulse_active[narrow_active] = 1.0

        zero = 1.0 - active
        narrow_zero = (zero > 0.0) & (zero < min_frac)
        if np.any(narrow_zero):
            # A center-switch pulse narrower than the hardware minimum is
            # suppressed.  This produces a full active state for that cycle.
            active[narrow_zero] = 1.0
            pulse_active[narrow_zero] = 1.0

    zero_duty = 1.0 - active
    m_effective = sign * active
    m_effective = np.where(active == 0.0, 0.0, m_effective)
    return m_effective, zero_duty, pulse_active


def _mismatch3(values: NDArray[np.float64], gain: tuple[float, float, float], offset: tuple[float, float, float]) -> NDArray[np.float64]:
    return values * np.asarray(gain, dtype=float) + np.asarray(offset, dtype=float)


def simulate_vienna_line_cycle(config: ViennaControlLabConfig) -> ViennaLineCycleWaveforms:
    """Simulate the final settled three-phase line cycle of a Vienna PFC.

    The plant is an averaged three-wire model.  Converter phase commands are
    projected to a floating neutral by removing their common-mode component;
    therefore third-harmonic and midpoint-balance common-mode injection do not
    directly force line current.  Midpoint current is produced by the averaged
    zero-state occupancy ``sum(D0_x * i_x)`` and drives the split capacitors.
    """
    config.validate()
    stage, fw = config.power_stage, config.firmware
    rate = max(config.waveform_integration_rate_hz, 4.0 * fw.current_loop_rate_hz)
    dt = 1.0 / rate
    total = config.waveform_line_cycles / stage.line_frequency_hz
    n = int(round(total * rate)) + 1
    time = np.arange(n, dtype=float) * dt

    i_sense = [_SampledSenseChain(config.phase_current_sense, 0.0) for _ in range(3)]
    v_sense = [_SampledSenseChain(config.phase_voltage_sense, 0.0) for _ in range(3)]
    vp_sense = _SampledSenseChain(config.split_bus_sense, stage.bus_voltage_v / 2.0)
    vn_sense = _SampledSenseChain(config.split_bus_sense, stage.bus_voltage_v / 2.0)
    current_ctrl = [_FirmwareController(config.current_controller, 0.0) for _ in range(3)]
    voltage_ctrl = _FirmwareController(config.voltage_controller, stage.input_conductance_a_per_v)
    balance_ctrl = _FirmwareController(config.balance_controller, 0.0)

    names: list[str] = []
    for phase in "abc":
        names += [
            f"v{phase}", f"v{phase}_meas",
            f"i{phase}", f"i{phase}_meas", f"i{phase}_ref",
            f"pi_{phase}", f"mod_{phase}", f"duty_{phase}",
            f"active_duty_{phase}", f"vconv_avg_{phase}",
            f"inductor_ff_{phase}", f"min_pulse_{phase}", f"p_{phase}",
        ]
    names += [
        "third_harmonic_injection",
        "vdc", "vdc_plus", "vdc_minus", "vdc_delta", "vdc_measured",
        "gcmd", "vloop", "balance_output", "midpoint_current",
        "input_power_total", "load_current", "bus_series_current", "sector",
        "current_update_strobe", "reference_update_strobe",
        "voltage_update_strobe", "balance_update_strobe",
    ]
    data = {key: np.zeros(n, dtype=float) for key in names}

    currents = np.zeros(3, dtype=float)
    refs = np.zeros(3, dtype=float)
    refs_prev = np.zeros(3, dtype=float)
    pi = np.zeros(3, dtype=float)
    mods = np.zeros(3, dtype=float)
    zero_duty = np.ones(3, dtype=float)
    inductor_ff = np.zeros(3, dtype=float)

    vp = stage.bus_voltage_v / 2.0 + 0.5 * config.initial_midpoint_imbalance_v
    vn = stage.bus_voltage_v / 2.0 - 0.5 * config.initial_midpoint_imbalance_v
    gcmd = stage.input_conductance_a_per_v
    vloop = gcmd
    balance = 0.0

    next_i = next_r = next_v = next_b = 0.0
    ti = 1.0 / fw.current_loop_rate_hz
    tr = 1.0 / fw.reference_rate_hz
    tv = 1.0 / fw.voltage_loop_rate_hz
    tb = 1.0 / fw.balance_loop_rate_hz
    min_pulse_fraction = stage.minimum_effective_pulse_s * stage.switching_frequency_hz

    if stage.load_model == LoadModel.RESISTIVE:
        rload = stage.bus_voltage_v**2 / stage.output_power_w
    else:
        rload = math.inf

    for idx, t in enumerate(time):
        wt = 2.0 * math.pi * stage.line_frequency_hz * t
        volts = np.asarray(_phase_values(stage.phase_peak_v, wt), dtype=float)

        vmeas_raw = np.asarray([v_sense[k].step(volts[k], t, dt) for k in range(3)])
        imeas_raw = np.asarray([i_sense[k].step(currents[k], t, dt) for k in range(3)])
        vmeas = _mismatch3(vmeas_raw, config.phase_voltage_gain_scale, config.phase_voltage_offset_v)
        imeas = _mismatch3(imeas_raw, config.phase_current_gain_scale, config.phase_current_offset_a)

        vp_m = vp_sense.step(vp, t, dt) * config.split_bus_gain_scale[0] + config.split_bus_offset_v[0]
        vn_m = vn_sense.step(vn, t, dt) * config.split_bus_gain_scale[1] + config.split_bus_offset_v[1]
        vdc_m = vp_m + vn_m

        si = sr = sv = sb = 0.0
        if t + 0.5 * dt >= next_v:
            sv = 1.0
            vloop = voltage_ctrl.step(stage.bus_voltage_v - vdc_m)
            gcmd = max(vloop, 0.0)
            next_v += tv

        if t + 0.5 * dt >= next_b:
            sb = 1.0
            balance = balance_ctrl.step(vp_m - vn_m)
            balance = min(max(balance, -fw.balance_injection_limit), fw.balance_injection_limit)
            next_b += tb

        if t + 0.5 * dt >= next_r:
            sr = 1.0
            refs_prev[:] = refs
            refs = gcmd * vmeas
            if fw.inductor_voltage_drop_feedforward_enabled:
                dref_dt = (refs - refs_prev) / tr
                inductor_ff = stage.phase_series_resistance_ohm * refs + stage.boost_inductance_h * dref_dt
            else:
                inductor_ff[:] = 0.0
            next_r += tr

        if t + 0.5 * dt >= next_i:
            si = 1.0
            for k in range(3):
                pi[k] = current_ctrl[k].step(refs[k] - imeas[k])
            next_i += ti

        vdc = max(vp + vn, 50.0)
        half = max(0.5 * vdc, 25.0)
        third = _third_harmonic_common_mode(vmeas, fw.third_harmonic_injection_enabled)

        # Desired converter phase voltage for reference-current tracking:
        # vconv* = vgrid - R*i* - L*di*/dt.  The current controller provides a
        # signed modulation correction; positive current error must reduce the
        # converter voltage, therefore it is subtracted.
        mff = (vmeas - third - inductor_ff) / half
        m_raw = mff - pi - balance
        mods, zero_duty, min_pulse = _apply_minimum_zero_state_pulse(
            m_raw,
            modulation_limit=stage.modulation_limit,
            minimum_pulse_fraction=min_pulse_fraction,
        )

        # Convert signed active-state fractions to actual split-bus voltage.
        raw_vconv = np.where(mods >= 0.0, mods * vp, mods * vn)
        # Three-wire source has no neutral conductor.  Common-mode converter
        # voltage moves the floating star point and is removed from phase
        # current dynamics.
        vconv_phase = raw_vconv - float(np.mean(raw_vconv))
        di = (
            volts - vconv_phase - stage.phase_series_resistance_ohm * currents
        ) / stage.boost_inductance_h
        currents = currents + di * dt
        # Eliminate numerical common-mode current drift in the three-wire model.
        currents = currents - float(np.mean(currents))

        pin = float(np.dot(volts, currents))
        i_bus = pin / max(vdc, 50.0)
        load_i = stage.output_power_w / max(vdc, 50.0) if math.isinf(rload) else vdc / rload
        i_series = i_bus - load_i

        # During center-switch zero states the corresponding phase current enters
        # the midpoint. Positive Imid discharges C+ and charges C-, reducing
        # positive delta V = Vdc+ - Vdc-.
        i_mid = float(np.dot(zero_duty, currents))
        vp += (i_series - i_mid) / stage.upper_bus_capacitance_f * dt
        vn += (i_series + i_mid) / stage.lower_bus_capacitance_f * dt
        vp = max(vp, 10.0)
        vn = max(vn, 10.0)

        angle = (360.0 * stage.line_frequency_hz * t) % 360.0
        for k, phase in enumerate("abc"):
            data[f"v{phase}"][idx] = volts[k]
            data[f"v{phase}_meas"][idx] = vmeas[k]
            data[f"i{phase}"][idx] = currents[k]
            data[f"i{phase}_meas"][idx] = imeas[k]
            data[f"i{phase}_ref"][idx] = refs[k]
            data[f"pi_{phase}"][idx] = pi[k]
            data[f"mod_{phase}"][idx] = mods[k]
            data[f"duty_{phase}"][idx] = zero_duty[k]
            data[f"active_duty_{phase}"][idx] = abs(mods[k])
            data[f"vconv_avg_{phase}"][idx] = vconv_phase[k]
            data[f"inductor_ff_{phase}"][idx] = inductor_ff[k]
            data[f"min_pulse_{phase}"][idx] = min_pulse[k]
            data[f"p_{phase}"][idx] = volts[k] * currents[k]

        data["third_harmonic_injection"][idx] = third
        data["vdc"][idx] = vp + vn
        data["vdc_plus"][idx] = vp
        data["vdc_minus"][idx] = vn
        data["vdc_delta"][idx] = vp - vn
        data["vdc_measured"][idx] = vdc_m
        data["gcmd"][idx] = gcmd
        data["vloop"][idx] = vloop
        data["balance_output"][idx] = balance
        data["midpoint_current"][idx] = i_mid
        data["input_power_total"][idx] = pin
        data["load_current"][idx] = load_i
        data["bus_series_current"][idx] = i_series
        data["sector"][idx] = _sector(angle)
        data["current_update_strobe"][idx] = si
        data["reference_update_strobe"][idx] = sr
        data["voltage_update_strobe"][idx] = sv
        data["balance_update_strobe"][idx] = sb

    sl = _last_cycle_slice(time, stage.line_frequency_hz)
    tl = time[sl]
    phase_metrics = [
        _harmonic_metrics(tl, data[f"v{ph}"][sl], data[f"i{ph}"][sl], stage.line_frequency_hz)
        for ph in "abc"
    ]
    irms = tuple(float(x[1]) for x in phase_metrics)
    thd = tuple(float(100.0 * x[7]) for x in phase_metrics)
    pf = tuple(float(x[4]) for x in phase_metrics)
    displacement = tuple(float(x[5]) for x in phase_metrics)
    distortion = tuple(float(x[6]) for x in phase_metrics)
    total_p = float(np.mean(data["input_power_total"][sl]))
    s3 = math.sqrt(3.0) * stage.line_line_rms_v * float(np.mean(irms))
    overall = total_p / max(s3, 1e-12)
    unbalance = 100.0 * (max(irms) - min(irms)) / max(float(np.mean(irms)), 1e-12)
    vdc_arr = data["vdc"][sl]
    delta_arr = data["vdc_delta"][sl]
    imid_arr = data["midpoint_current"][sl]

    metrics = ViennaWaveformMetrics(
        phase_current_rms_a=irms,
        phase_current_thd_percent=thd,
        phase_power_factor=pf,
        phase_displacement_factor=displacement,
        phase_distortion_factor=distortion,
        total_input_power_w=total_p,
        overall_power_factor=overall,
        current_unbalance_percent=unbalance,
        bus_voltage_average_v=float(np.mean(vdc_arr)),
        bus_voltage_ripple_pp_v=float(np.ptp(vdc_arr)),
        midpoint_delta_average_v=float(np.mean(delta_arr)),
        midpoint_delta_pp_v=float(np.ptp(delta_arr)),
        midpoint_current_rms_a=float(np.sqrt(np.mean(imid_arr**2))),
        harmonic_orders=phase_metrics[0][9],
        phase_a_harmonic_rms_a=phase_metrics[0][10],
        phase_b_harmonic_rms_a=phase_metrics[1][10],
        phase_c_harmonic_rms_a=phase_metrics[2][10],
    )

    warnings: list[str] = []
    if overall < 0.98:
        warnings.append("Vienna overall PF below 0.98.")
    if max(thd) > 8.0:
        warnings.append("Vienna phase current THD exceeds 8%.")
    if abs(metrics.midpoint_delta_average_v) > 0.02 * stage.bus_voltage_v:
        warnings.append("Vienna DC midpoint average imbalance exceeds 2% of total bus.")
    if metrics.current_unbalance_percent > 2.0:
        warnings.append("Vienna phase-current RMS unbalance exceeds 2%.")

    units = {key: "" for key in names}
    for ph in "abc":
        units[f"v{ph}"] = units[f"v{ph}_meas"] = units[f"vconv_avg_{ph}"] = units[f"inductor_ff_{ph}"] = "V"
        units[f"i{ph}"] = units[f"i{ph}_meas"] = units[f"i{ph}_ref"] = "A"
        units[f"mod_{ph}"] = units[f"duty_{ph}"] = units[f"active_duty_{ph}"] = "pu"
    units.update({"vdc":"V","vdc_plus":"V","vdc_minus":"V","vdc_delta":"V","vdc_measured":"V","midpoint_current":"A","input_power_total":"W","load_current":"A","bus_series_current":"A","third_harmonic_injection":"V"})
    return ViennaLineCycleWaveforms(time, data, units, metrics, tuple(warnings))


def _select_final_cycle_workpoint(
    config: ViennaControlLabConfig,
    line: ViennaLineCycleWaveforms,
    angle_deg: float,
) -> int:
    stage = config.power_stage
    phase_angle = (360.0 * stage.line_frequency_hz * line.time_s) % 360.0
    sl = _last_cycle_slice(np.asarray(line.time_s), stage.line_frequency_hz)
    start = sl.start or 0
    delta = np.abs(((phase_angle[start:] - angle_deg + 180.0) % 360.0) - 180.0)
    return start + int(np.argmin(delta))


def build_vienna_switching_waveforms(
    config: ViennaControlLabConfig,
    line: ViennaLineCycleWaveforms | None = None,
    *,
    line_angle_deg: float | None = None,
) -> ViennaSwitchingWaveforms:
    """Reconstruct local three-level switching from a settled AC workpoint."""
    config.validate()
    stage = config.power_stage
    angle = config.switching_line_angle_deg if line_angle_deg is None else float(line_angle_deg) % 360.0
    if line is None:
        line = simulate_vienna_line_cycle(config)

    idx = _select_final_cycle_workpoint(config, line, angle)
    vgrid = np.array([line.signals[f"v{p}"][idx] for p in "abc"], dtype=float)
    currents = np.array([line.signals[f"i{p}"][idx] for p in "abc"], dtype=float)
    mods = np.array([line.signals[f"mod_{p}"][idx] for p in "abc"], dtype=float)
    zero_duty = np.array([line.signals[f"duty_{p}"][idx] for p in "abc"], dtype=float)
    vp = float(line.signals["vdc_plus"][idx])
    vn = float(line.signals["vdc_minus"][idx])

    fs = stage.switching_frequency_hz
    period = 1.0 / fs
    spp = config.switching_samples_per_cycle
    cycles = config.switching_cycles
    t = np.arange(cycles * spp, dtype=float) * period / spp
    tau = np.mod(t, period)
    carrier = tau / period
    gates = np.vstack([(carrier < zero_duty[k]).astype(float) for k in range(3)])

    vconv_raw = np.zeros_like(gates)
    vconv_phase = np.zeros_like(gates)
    phase_current = np.zeros_like(gates)
    upper_diode = np.zeros_like(gates)
    lower_diode = np.zeros_like(gates)
    midpoint = np.zeros_like(t)
    upper_bus_current = np.zeros_like(t)
    lower_bus_current = np.zeros_like(t)

    i_state = currents.copy()
    dt = period / spp
    for sample in range(len(t)):
        raw = np.zeros(3, dtype=float)
        for k in range(3):
            if gates[k, sample] > 0.5:
                raw[k] = 0.0
            elif mods[k] >= 0.0:
                raw[k] = vp
            else:
                raw[k] = -vn
        phase_voltage = raw - float(np.mean(raw))
        di = (vgrid - phase_voltage - stage.phase_series_resistance_ohm * i_state) / stage.boost_inductance_h
        i_state = i_state + di * dt
        i_state = i_state - float(np.mean(i_state))

        vconv_raw[:, sample] = raw
        vconv_phase[:, sample] = phase_voltage
        phase_current[:, sample] = i_state
        midpoint[sample] = float(np.dot(gates[:, sample], i_state))
        off = 1.0 - gates[:, sample]
        upper_diode[:, sample] = off * np.maximum(i_state, 0.0)
        lower_diode[:, sample] = off * np.maximum(-i_state, 0.0)
        upper_bus_current[sample] = float(np.sum(upper_diode[:, sample]))
        lower_bus_current[sample] = float(np.sum(lower_diode[:, sample]))

    load_current = stage.output_power_w / max(vp + vn, 50.0)
    signals: dict[str, FloatArray] = {
        "carrier": carrier,
        "sector": np.full_like(t, _sector(angle), dtype=float),
        "midpoint_current": midpoint,
        "upper_bus_current": upper_bus_current,
        "lower_bus_current": lower_bus_current,
        "load_current": np.full_like(t, load_current),
        "upper_cap_current": upper_bus_current - load_current,
        "lower_cap_current": lower_bus_current - load_current,
    }
    for k, phase in enumerate("abc"):
        signals[f"gate_{phase}"] = gates[k]
        signals[f"vconv_raw_{phase}"] = vconv_raw[k]
        signals[f"vconv_{phase}"] = vconv_phase[k]
        signals[f"vphase_{phase}"] = np.full_like(t, vgrid[k])
        signals[f"inductor_voltage_{phase}"] = vgrid[k] - vconv_phase[k]
        signals[f"current_{phase}"] = phase_current[k]
        signals[f"zero_duty_{phase}"] = np.full_like(t, zero_duty[k])
        signals[f"active_duty_{phase}"] = np.full_like(t, abs(mods[k]))
        # Backward-compatible alias: Vienna duty means center-switch zero-state duty.
        signals[f"duty_{phase}"] = np.full_like(t, zero_duty[k])
        signals[f"upper_diode_{phase}"] = upper_diode[k]
        signals[f"lower_diode_{phase}"] = lower_diode[k]

    units = {key: "" for key in signals}
    for ph in "abc":
        units[f"vconv_raw_{ph}"] = units[f"vconv_{ph}"] = units[f"vphase_{ph}"] = units[f"inductor_voltage_{ph}"] = "V"
        units[f"current_{ph}"] = units[f"upper_diode_{ph}"] = units[f"lower_diode_{ph}"] = "A"
    units.update({"midpoint_current":"A","upper_bus_current":"A","lower_bus_current":"A","upper_cap_current":"A","lower_cap_current":"A","load_current":"A"})
    return ViennaSwitchingWaveforms(
        time_s=t,
        signals=signals,
        units=units,
        line_angle_deg=angle,
        switching_frequency_hz=fs,
        source_time_s=float(line.time_s[idx]),
    )


# Imported late to avoid exposing control config implementation details above.
from pfc_design.control.config import LoadModel  # noqa: E402

__all__ = [
    "ViennaLineCycleWaveforms",
    "ViennaSwitchingWaveforms",
    "ViennaWaveformMetrics",
    "simulate_vienna_line_cycle",
    "build_vienna_switching_waveforms",
]
