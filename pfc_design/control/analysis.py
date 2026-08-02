"""Small-signal current-loop and bus-voltage-loop analysis for PFC Control Lab."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from llc_design.control.digital_loop import (
    DigitalTransferFunction,
    StabilityMargins,
    calculate_stability_margins,
    controller_kind,
)

from .config import LoadModel, PFCControlLabConfig
from .sensing import (
    SenseChainSummary,
    SenseFrequencyResponse,
    sense_frequency_response,
    summarize_sense_chain,
)


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class PFCOperatingPoint:
    vin_instantaneous_abs_v: float
    line_angle_deg: float
    current_reference_a: float
    ideal_duty: float
    indu_comp: float
    input_conductance_a_per_v: float
    input_current_rms_a: float


@dataclass(frozen=True)
class LoopResult:
    name: str
    controller: DigitalTransferFunction
    responses: dict[str, ComplexArray]
    margins: StabilityMargins
    likely_stable: bool


@dataclass(frozen=True)
class PFCControlLabAnalysis:
    config: PFCControlLabConfig
    frequencies_hz: FloatArray
    operating_point: PFCOperatingPoint
    current_loop: LoopResult
    voltage_loop: LoopResult
    current_sense_response: SenseFrequencyResponse
    vac_sense_response: SenseFrequencyResponse
    vbus_sense_response: SenseFrequencyResponse
    current_sense_summary: SenseChainSummary
    vac_sense_summary: SenseChainSummary
    vbus_sense_summary: SenseChainSummary
    angle_margins: dict[float, StabilityMargins]
    warnings: tuple[str, ...]

    def summary_dict(self) -> dict[str, Any]:
        return {
            "operating_point": asdict(self.operating_point),
            "current_loop": {
                "controller": self.current_loop.controller.name,
                "crossover_hz": self.current_loop.margins.critical_gain_crossover_hz,
                "phase_margin_deg": self.current_loop.margins.phase_margin_deg,
                "gain_margin_db": self.current_loop.margins.gain_margin_db,
                "stable": self.current_loop.likely_stable,
            },
            "voltage_loop": {
                "controller": self.voltage_loop.controller.name,
                "crossover_hz": self.voltage_loop.margins.critical_gain_crossover_hz,
                "phase_margin_deg": self.voltage_loop.margins.phase_margin_deg,
                "gain_margin_db": self.voltage_loop.margins.gain_margin_db,
                "stable": self.voltage_loop.likely_stable,
            },
            "sensing": {
                "current": asdict(self.current_sense_summary),
                "vac": asdict(self.vac_sense_summary),
                "vbus": asdict(self.vbus_sense_summary),
            },
            "warnings": list(self.warnings),
        }


def _controller_response(controller: DigitalTransferFunction, frequencies_hz: FloatArray) -> ComplexArray:
    return controller.frequency_response(frequencies_hz)


def _zoh_response(frequencies_hz: FloatArray, sample_time_s: float) -> ComplexArray:
    f = np.asarray(frequencies_hz, dtype=float)
    return np.sinc(f * sample_time_s) * np.exp(-1j * math.pi * f * sample_time_s)


def _current_plant_response(
    config: PFCControlLabConfig,
    frequencies_hz: FloatArray,
) -> ComplexArray:
    stage = config.power_stage
    s = 1j * 2.0 * math.pi * frequencies_hz
    return stage.bus_voltage_v / (
        stage.boost_inductance_h * s + stage.equivalent_series_resistance_ohm
    )


def _current_loop_at_indu_comp(
    config: PFCControlLabConfig,
    frequencies_hz: FloatArray,
    current_sense: SenseFrequencyResponse,
    indu_comp: float,
) -> tuple[dict[str, ComplexArray], StabilityMargins, DigitalTransferFunction]:
    controller = config.current_controller.transfer_function()
    control = _controller_response(controller, frequencies_hz)
    plant = _current_plant_response(config, frequencies_hz)
    pwm = _zoh_response(frequencies_hz, 1.0 / config.firmware.current_loop_rate_hz)
    pwm *= np.exp(-1j * 2.0 * math.pi * frequencies_hz * (
        config.firmware.current_computation_delay_s
        + config.firmware.current_pwm_update_delay_s
    ))
    forward = control * indu_comp * pwm * plant
    loop = forward * current_sense.total
    closed_measured = loop / (1.0 + loop)
    closed_actual = forward / (1.0 + loop)
    sensitivity = 1.0 / (1.0 + loop)
    return ({
        "plant_gid": plant,
        "controller_ci": control,
        "pwm_zoh": pwm,
        "sense_hi": current_sense.total,
        "sense_hi_analog": current_sense.calibrated_analog,
        "sense_hi_adc": (
            current_sense.adc_aperture
            * current_sense.multi_soc_recursive
            * current_sense.digital_filter
            * current_sense.pure_delay
        ),
        "forward_current": forward,
        "open_current": loop,
        "closed_current_measured": closed_measured,
        "closed_current_actual": closed_actual,
        "sensitivity_current": sensitivity,
    }, calculate_stability_margins(frequencies_hz, loop), controller)


def _bus_power_plant_response(
    config: PFCControlLabConfig,
    frequencies_hz: FloatArray,
    current_closed_actual: ComplexArray,
) -> ComplexArray:
    stage = config.power_stage
    s = 1j * 2.0 * math.pi * frequencies_hz
    if stage.load_model == LoadModel.RESISTIVE:
        incremental_load_w_per_v = 2.0 * stage.output_power_w / stage.bus_voltage_v
    else:
        incremental_load_w_per_v = 0.0
    denominator = (
        stage.bus_capacitance_f * stage.bus_voltage_v * s
        + incremental_load_w_per_v
    )
    # ESR creates the familiar capacitor zero.  The plant input is conductance
    # command [A/V], so dPin/dg = Vrms^2 [W/(A/V)].
    esr_zero = 1.0 + s * stage.bus_capacitance_f * stage.bus_cap_esr_ohm
    return (
        stage.vin_rms_v**2
        * current_closed_actual
        * esr_zero
        / denominator
    )


def _amc_vff_response(config: PFCControlLabConfig, frequencies_hz: FloatArray) -> ComplexArray:
    firmware = config.firmware
    vrms = config.power_stage.vin_rms_v
    if firmware.vff_bypass:
        dc_gain = firmware.vac_rms_feedforward_gain
    else:
        # Firmware: vloop /= (Kff*Vrms)^2; gcmd = vloop*Kff.
        dc_gain = 1.0 / (firmware.vac_rms_feedforward_gain * vrms**2)
    ts = 1.0 / firmware.amc_rate_hz
    response = np.full(frequencies_hz.shape, dc_gain, dtype=complex) * _zoh_response(
        frequencies_hz, ts
    )
    response *= np.exp(-1j * 2.0 * math.pi * frequencies_hz * (
        firmware.amc_update_delay_s + firmware.voltage_computation_delay_s
    ))
    return response


def _likely_stable(margins: StabilityMargins, controller: DigitalTransferFunction) -> bool:
    pm = margins.phase_margin_deg
    gm = margins.gain_margin_db
    return bool(
        controller.stable
        and (pm is None or pm > 0.0)
        and (gm is None or gm > 0.0)
    )


def build_pfc_control_lab_analysis(config: PFCControlLabConfig) -> PFCControlLabAnalysis:
    """Build both nested loops and all external sensing paths."""

    config.validate()
    frequencies = np.geomspace(
        config.frequency_start_hz,
        config.frequency_stop_hz,
        config.frequency_points,
    )
    current_sense = sense_frequency_response(config.current_sense, frequencies)
    vac_sense = sense_frequency_response(config.vac_sense, frequencies)
    vbus_sense = sense_frequency_response(config.vbus_sense, frequencies)

    stage = config.power_stage
    operating = PFCOperatingPoint(
        vin_instantaneous_abs_v=stage.vin_instantaneous_abs_v,
        line_angle_deg=stage.line_angle_deg,
        current_reference_a=stage.current_reference_a,
        ideal_duty=stage.ideal_duty,
        indu_comp=stage.indu_comp,
        input_conductance_a_per_v=stage.input_conductance_a_per_v,
        input_current_rms_a=stage.output_power_w /
        (stage.efficiency * stage.vin_rms_v),
    )

    current_responses, current_margins, current_controller = _current_loop_at_indu_comp(
        config, frequencies, current_sense, operating.indu_comp
    )
    current_result = LoopResult(
        name="PFC current inner loop",
        controller=current_controller,
        responses=current_responses,
        margins=current_margins,
        likely_stable=_likely_stable(current_margins, current_controller),
    )

    voltage_controller = config.voltage_controller.transfer_function()
    cv = _controller_response(voltage_controller, frequencies)
    amc_vff = _amc_vff_response(config, frequencies)
    bus_plant = _bus_power_plant_response(
        config, frequencies, current_responses["closed_current_actual"]
    )
    outer_forward = cv * amc_vff * bus_plant
    outer_open = outer_forward * vbus_sense.total
    outer_closed = outer_open / (1.0 + outer_open)
    outer_sensitivity = 1.0 / (1.0 + outer_open)
    voltage_responses = {
        "controller_cv": cv,
        "amc_vff": amc_vff,
        "current_closed_for_outer": current_responses["closed_current_actual"],
        "bus_plant_gvg": bus_plant,
        "sense_hv": vbus_sense.total,
        "sense_hv_analog": vbus_sense.calibrated_analog,
        "sense_hv_adc": (
            vbus_sense.adc_aperture
            * vbus_sense.multi_soc_recursive
            * vbus_sense.digital_filter
            * vbus_sense.pure_delay
        ),
        "forward_voltage": outer_forward,
        "open_voltage": outer_open,
        "closed_voltage": outer_closed,
        "sensitivity_voltage": outer_sensitivity,
    }
    voltage_margins = calculate_stability_margins(frequencies, outer_open)
    voltage_result = LoopResult(
        name="PFC bus-voltage outer loop",
        controller=voltage_controller,
        responses=voltage_responses,
        margins=voltage_margins,
        likely_stable=_likely_stable(voltage_margins, voltage_controller),
    )

    angle_margins: dict[float, StabilityMargins] = {}
    original = stage
    for angle in (10.0, 20.0, 30.0, 45.0, 60.0, 75.0, 90.0):
        vin = abs(original.line_peak_v * math.sin(math.radians(angle)))
        iref = original.input_conductance_a_per_v * vin
        indu_comp = min(max(config.firmware.indu_comp_gain * iref,
                            config.firmware.indu_comp_min),
                        config.firmware.indu_comp_max)
        _, margin, _ = _current_loop_at_indu_comp(
            config, frequencies, current_sense, indu_comp
        )
        angle_margins[angle] = margin

    warnings: list[str] = []
    if current_margins.critical_gain_crossover_hz is None:
        warnings.append("Current-loop 0 dB crossover is outside the plotted range.")
    if voltage_margins.critical_gain_crossover_hz is None:
        warnings.append("Voltage-loop 0 dB crossover is outside the plotted range.")
    if current_margins.phase_margin_deg is not None and current_margins.phase_margin_deg < 35.0:
        warnings.append("Current-loop phase margin is below 35 degrees.")
    if voltage_margins.phase_margin_deg is not None and voltage_margins.phase_margin_deg < 35.0:
        warnings.append("Voltage-loop phase margin is below 35 degrees.")
    fc_i = current_margins.critical_gain_crossover_hz
    fc_v = voltage_margins.critical_gain_crossover_hz
    if fc_i and fc_v and fc_i / fc_v < 5.0:
        warnings.append("Current/voltage loop bandwidth separation is below 5:1.")
    if fc_v and fc_v > 0.5 * stage.line_frequency_hz:
        warnings.append(
            "Voltage-loop crossover is high relative to line frequency; 2x-line ripple may modulate input current."
        )
    pulse = stage.minimum_effective_pulse_s
    if pulse > 0.0 and pulse * stage.switching_frequency_hz >= stage.duty_min:
        warnings.append("Minimum effective pulse width is active at the configured duty minimum.")
    if config.firmware.vff_bypass:
        warnings.append("Vrms feed-forward bypass is enabled; outer-loop gain becomes line-voltage dependent.")

    return PFCControlLabAnalysis(
        config=config,
        frequencies_hz=frequencies,
        operating_point=operating,
        current_loop=current_result,
        voltage_loop=voltage_result,
        current_sense_response=current_sense,
        vac_sense_response=vac_sense,
        vbus_sense_response=vbus_sense,
        current_sense_summary=summarize_sense_chain(
            config.current_sense, switching_frequency_hz=stage.switching_frequency_hz),
        vac_sense_summary=summarize_sense_chain(
            config.vac_sense, switching_frequency_hz=stage.switching_frequency_hz),
        vbus_sense_summary=summarize_sense_chain(
            config.vbus_sense, switching_frequency_hz=stage.switching_frequency_hz),
        angle_margins=angle_margins,
        warnings=tuple(warnings),
    )


__all__ = [
    "LoopResult",
    "PFCControlLabAnalysis",
    "PFCOperatingPoint",
    "build_pfc_control_lab_analysis",
]
