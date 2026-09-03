"""Configuration objects for the PFC double-loop control laboratory.

The control-lab defaults mirror the supplied C firmware:

* current loop: 50 kHz, Tustin PI, Kp=0.01, Ti=75 us;
* AMC/reference layer: 25 kHz;
* bus-voltage loop: 10 kHz, Tustin PI, Kp=0.1, Ti=2 ms;
* fixed-frequency 50 kHz totem-pole PFC;
* duty feed-forward ``1 - |Vac|/Vbus_ref``;
* current-loop gain scheduling ``clamp(0.085*Iref, 0.7, 1.0)``.

Protection state machines, OVP/UVP, minimum-pulse parking and zero-crossing
commutation are nonlinear.  They are handled by waveform/validity analysis and
are intentionally excluded from linear Bode calculations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math

from llc_design.control.digital_loop import (
    ControllerKind,
    PIControllerConfig,
    PIFControllerConfig,
    TwoP2ZControllerConfig,
)


class LoadModel(str, Enum):
    """DC-bus load model used by the outer voltage loop."""

    CONSTANT_POWER = "constant_power"
    RESISTIVE = "resistive"


@dataclass(frozen=True)
class DigitalFilterConfig:
    """Optional first-order firmware filter in engineering units.

    ``alpha=1`` bypasses the filter.  Otherwise the transfer is
    ``alpha / (1 - (1-alpha) z^-1)``.
    """

    alpha: float = 1.0

    def validate(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("digital-filter alpha must lie in (0, 1]")


@dataclass(frozen=True)
class ADCTimingConfig:
    """ADC acquisition, conversion, triggering and digital delay settings."""

    sample_rate_hz: float
    adc_clock_hz: float = 60.0e6
    acquisition_time_s: float = 300.0e-9
    conversion_cycles: float = 13.0
    soc_count: int = 1
    soc_spacing_s: float = 0.0
    recursive_previous_weight: float = 0.0
    computation_delay_s: float = 1.0e-6
    pwm_update_delay_s: float = 10.0e-6
    include_zero_order_hold: bool = True
    digital_filter: DigitalFilterConfig = field(default_factory=DigitalFilterConfig)

    @property
    def sample_time_s(self) -> float:
        return 1.0 / self.sample_rate_hz

    @property
    def conversion_time_s(self) -> float:
        return self.conversion_cycles / self.adc_clock_hz

    @property
    def nominal_latency_s(self) -> float:
        sequence = max(self.soc_count - 1, 0) * self.soc_spacing_s
        return (
            0.5 * self.acquisition_time_s
            + self.conversion_time_s
            + sequence
            + self.computation_delay_s
            + self.pwm_update_delay_s
        )

    def validate(self) -> None:
        if self.sample_rate_hz <= 0.0 or self.adc_clock_hz <= 0.0:
            raise ValueError("ADC and control sample rates must be positive")
        if self.acquisition_time_s < 0.0 or self.conversion_cycles < 0.0:
            raise ValueError("ADC acquisition/conversion settings cannot be negative")
        if self.soc_count < 1:
            raise ValueError("at least one SOC is required")
        if self.soc_spacing_s < 0.0:
            raise ValueError("SOC spacing cannot be negative")
        if not 0.0 <= self.recursive_previous_weight < 1.0:
            raise ValueError("recursive previous weight must lie in [0, 1)")
        if self.computation_delay_s < 0.0 or self.pwm_update_delay_s < 0.0:
            raise ValueError("digital delays cannot be negative")
        self.digital_filter.validate()


@dataclass(frozen=True)
class ExternalSenseConfig:
    """Physical external signal-conditioning chain for one measured quantity.

    The raw front-end gain is useful for ADC full-scale/noise calculations.  For
    control-loop Bode analysis, ``normalize_to_engineering_units=True`` models
    the firmware calibration that converts ADC counts back to A or V.  The DC
    gain is then one while all analog poles and delays remain present.
    """

    name: str
    front_end_gain_v_per_unit: float = 1.0
    amplifier_gain: float = 1.0
    amplifier_bandwidth_hz: float = 2.0e6
    source_resistance_ohm: float = 0.0
    shunt_capacitance_f: float = 0.0
    output_resistance_ohm: float = 0.0
    adc_capacitance_f: float = 0.0
    second_resistance_ohm: float = 0.0
    second_capacitance_f: float = 0.0
    normalize_to_engineering_units: bool = True
    adc_vref_v: float = 3.3
    adc_bits: int = 12
    timing: ADCTimingConfig = field(
        default_factory=lambda: ADCTimingConfig(sample_rate_hz=50.0e3)
    )

    @property
    def raw_dc_gain(self) -> float:
        return self.front_end_gain_v_per_unit * self.amplifier_gain

    @property
    def adc_codes_per_unit(self) -> float:
        return self.raw_dc_gain * ((2**self.adc_bits) / self.adc_vref_v)

    def validate(self) -> None:
        if self.front_end_gain_v_per_unit <= 0.0 or self.amplifier_gain <= 0.0:
            raise ValueError(f"{self.name}: signal-chain gains must be positive")
        if self.amplifier_bandwidth_hz <= 0.0:
            raise ValueError(f"{self.name}: amplifier bandwidth must be positive")
        for value, label in (
            (self.source_resistance_ohm, "source resistance"),
            (self.shunt_capacitance_f, "shunt capacitance"),
            (self.output_resistance_ohm, "ADC series resistance"),
            (self.adc_capacitance_f, "ADC capacitance"),
            (self.second_resistance_ohm, "second resistance"),
            (self.second_capacitance_f, "second capacitance"),
        ):
            if value < 0.0:
                raise ValueError(f"{self.name}: {label} cannot be negative")
        if self.adc_vref_v <= 0.0 or self.adc_bits < 1:
            raise ValueError(f"{self.name}: invalid ADC range")
        self.timing.validate()


@dataclass(frozen=True)
class PFCPowerStageConfig:
    """Averaged single-phase CCM totem-pole PFC operating point."""

    vin_rms_v: float = 230.0
    line_frequency_hz: float = 50.0
    bus_voltage_v: float = 400.0
    output_power_w: float = 3300.0
    switching_frequency_hz: float = 50.0e3
    boost_inductance_h: float = 220.0e-6
    equivalent_series_resistance_ohm: float = 0.055
    bus_capacitance_f: float = 1320.0e-6
    bus_cap_esr_ohm: float = 0.035
    efficiency: float = 0.97
    load_model: LoadModel = LoadModel.CONSTANT_POWER
    line_angle_deg: float = 60.0
    duty_min: float = 0.01
    duty_max: float = 0.98
    minimum_effective_pulse_s: float = 0.0
    deadtime_s: float = 100.0e-9

    @property
    def line_peak_v(self) -> float:
        return math.sqrt(2.0) * self.vin_rms_v

    @property
    def line_angle_rad(self) -> float:
        return math.radians(self.line_angle_deg)

    @property
    def vin_instantaneous_abs_v(self) -> float:
        return abs(self.line_peak_v * math.sin(self.line_angle_rad))

    @property
    def input_conductance_a_per_v(self) -> float:
        return self.output_power_w / (self.efficiency * self.vin_rms_v**2)

    @property
    def current_reference_a(self) -> float:
        return self.input_conductance_a_per_v * self.vin_instantaneous_abs_v

    @property
    def ideal_duty(self) -> float:
        duty = 1.0 - self.vin_instantaneous_abs_v / self.bus_voltage_v
        return min(max(duty, self.duty_min), self.duty_max)

    @property
    def indu_comp(self) -> float:
        return min(max(0.085 * self.current_reference_a, 0.7), 1.0)

    def validate(self) -> None:
        positive = {
            "Vin RMS": self.vin_rms_v,
            "line frequency": self.line_frequency_hz,
            "bus voltage": self.bus_voltage_v,
            "output power": self.output_power_w,
            "switching frequency": self.switching_frequency_hz,
            "boost inductance": self.boost_inductance_h,
            "bus capacitance": self.bus_capacitance_f,
        }
        for label, value in positive.items():
            if value <= 0.0:
                raise ValueError(f"{label} must be positive")
        if self.equivalent_series_resistance_ohm < 0.0 or self.bus_cap_esr_ohm < 0.0:
            raise ValueError("series resistances cannot be negative")
        if not 0.0 < self.efficiency <= 1.0:
            raise ValueError("efficiency must lie in (0, 1]")
        if not 0.0 < self.line_angle_deg < 180.0:
            raise ValueError("line angle must lie between 0 and 180 degrees")
        if not 0.0 <= self.duty_min < self.duty_max <= 1.0:
            raise ValueError("invalid duty limits")
        if self.minimum_effective_pulse_s < 0.0 or self.deadtime_s < 0.0:
            raise ValueError("minimum pulse width/deadtime cannot be negative")


@dataclass(frozen=True)
class PFCFirmwareAlgorithmConfig:
    """Multi-rate firmware algorithm constants from the supplied project."""

    current_loop_rate_hz: float = 50.0e3
    amc_rate_hz: float = 25.0e3
    voltage_loop_rate_hz: float = 10.0e3
    vac_rms_lpf_alpha: float = 0.006283
    vac_rms_feedforward_gain: float = 0.01
    gcmd_max_a_per_v: float = 0.18
    indu_comp_gain: float = 0.085
    indu_comp_min: float = 0.7
    indu_comp_max: float = 1.0
    vff_bypass: bool = False
    current_computation_delay_s: float = 1.0e-6
    current_pwm_update_delay_s: float = 10.0e-6
    amc_update_delay_s: float = 20.0e-6
    voltage_computation_delay_s: float = 4.0e-6

    def validate(self) -> None:
        if min(self.current_loop_rate_hz, self.amc_rate_hz, self.voltage_loop_rate_hz) <= 0.0:
            raise ValueError("firmware loop rates must be positive")
        if self.current_loop_rate_hz < self.amc_rate_hz:
            raise ValueError("current loop must not be slower than AMC")
        if self.current_loop_rate_hz < self.voltage_loop_rate_hz:
            raise ValueError("current loop must not be slower than voltage loop")
        if not 0.0 < self.vac_rms_lpf_alpha <= 1.0:
            raise ValueError("VAC RMS alpha must lie in (0, 1]")
        if self.vac_rms_feedforward_gain <= 0.0:
            raise ValueError("VAC RMS feed-forward gain must be positive")
        if self.gcmd_max_a_per_v <= 0.0:
            raise ValueError("gcmd maximum must be positive")
        if min(self.current_computation_delay_s, self.current_pwm_update_delay_s,
               self.amc_update_delay_s, self.voltage_computation_delay_s) < 0.0:
            raise ValueError("firmware delays cannot be negative")
        if not 0.0 < self.indu_comp_min <= self.indu_comp_max:
            raise ValueError("invalid induction compensation limits")


ControllerConfig = PIControllerConfig | PIFControllerConfig | TwoP2ZControllerConfig


def default_current_controller() -> PIControllerConfig:
    return PIControllerConfig(
        kp=1.0e-2,
        ti_s=0.75e-4,
        sample_time_s=20.0e-6,
        output_min=-2.0,
        output_max=0.98,
    )


def default_voltage_controller() -> PIControllerConfig:
    return PIControllerConfig(
        kp=0.1,
        ti_s=2.0e-3,
        sample_time_s=100.0e-6,
        output_min=-1.0,
        output_max=40.0,
    )


def default_current_sense() -> ExternalSenseConfig:
    # Firmware header: 30 mV/A current sensor.  The RC values remain editable
    # because the supplied C source does not encode the board-level components.
    return ExternalSenseConfig(
        name="PFC inductor current",
        front_end_gain_v_per_unit=30.0e-3,
        amplifier_gain=1.0,
        amplifier_bandwidth_hz=2.0e6,
        output_resistance_ohm=220.0,
        adc_capacitance_f=2.0e-9,
        normalize_to_engineering_units=True,
        timing=ADCTimingConfig(
            sample_rate_hz=50.0e3,
            adc_clock_hz=60.0e6,
            acquisition_time_s=300.0e-9,
            computation_delay_s=0.0,
            pwm_update_delay_s=0.0,
        ),
    )


def default_vac_sense() -> ExternalSenseConfig:
    return ExternalSenseConfig(
        name="AC input voltage",
        front_end_gain_v_per_unit=1.0 / 150.0,
        amplifier_gain=1.0,
        amplifier_bandwidth_hz=1.0e6,
        source_resistance_ohm=2.0e3,
        shunt_capacitance_f=1.0e-9,
        output_resistance_ohm=220.0,
        adc_capacitance_f=2.0e-9,
        normalize_to_engineering_units=True,
        timing=ADCTimingConfig(
            sample_rate_hz=50.0e3,
            adc_clock_hz=60.0e6,
            acquisition_time_s=300.0e-9,
            computation_delay_s=0.0,
            pwm_update_delay_s=0.0,
        ),
    )


def default_vbus_sense() -> ExternalSenseConfig:
    return ExternalSenseConfig(
        name="PFC bus voltage",
        front_end_gain_v_per_unit=1.6e3 / (117.0e3 + 1.6e3),
        amplifier_gain=1.0,
        amplifier_bandwidth_hz=1.0e6,
        source_resistance_ohm=(117.0e3 * 1.6e3) / (117.0e3 + 1.6e3),
        shunt_capacitance_f=1.0e-9,
        output_resistance_ohm=220.0,
        adc_capacitance_f=2.0e-9,
        normalize_to_engineering_units=True,
        timing=ADCTimingConfig(
            sample_rate_hz=10.0e3,
            adc_clock_hz=60.0e6,
            acquisition_time_s=300.0e-9,
            computation_delay_s=0.0,
            pwm_update_delay_s=0.0,
        ),
    )


@dataclass(frozen=True)
class PFCControlLabConfig:
    """Complete double-loop and waveform-analysis configuration."""

    power_stage: PFCPowerStageConfig = field(default_factory=PFCPowerStageConfig)
    firmware: PFCFirmwareAlgorithmConfig = field(default_factory=PFCFirmwareAlgorithmConfig)
    current_controller: ControllerConfig = field(default_factory=default_current_controller)
    voltage_controller: ControllerConfig = field(default_factory=default_voltage_controller)
    current_sense: ExternalSenseConfig = field(default_factory=default_current_sense)
    vac_sense: ExternalSenseConfig = field(default_factory=default_vac_sense)
    vbus_sense: ExternalSenseConfig = field(default_factory=default_vbus_sense)
    frequency_start_hz: float = 0.5
    frequency_stop_hz: float = 20.0e3
    frequency_points: int = 1600
    waveform_line_cycles: int = 8
    switching_cycles: int = 2
    switching_samples_per_cycle: int = 800
    waveform_integration_rate_hz: float = 500.0e3

    def validate(self) -> None:
        self.power_stage.validate()
        self.firmware.validate()
        self.current_sense.validate()
        self.vac_sense.validate()
        self.vbus_sense.validate()
        if self.frequency_start_hz <= 0.0 or self.frequency_stop_hz <= self.frequency_start_hz:
            raise ValueError("invalid Bode frequency range")
        if self.frequency_stop_hz >= 0.5 * self.firmware.current_loop_rate_hz:
            raise ValueError("Bode stop frequency must be below the current-loop Nyquist frequency")
        if self.frequency_points < 100:
            raise ValueError("at least 100 Bode points are required")
        if self.waveform_line_cycles < 3:
            raise ValueError("at least three line cycles are required for settled waveform metrics")
        if self.switching_cycles < 1 or self.switching_cycles > 50:
            raise ValueError("switching waveform cycles must lie in [1, 50]")
        if self.switching_samples_per_cycle < 100:
            raise ValueError("at least 100 switching samples per cycle are required")
        if self.waveform_integration_rate_hz < 4.0 * self.firmware.current_loop_rate_hz:
            raise ValueError("waveform integration rate must be at least 4x current-loop rate")


__all__ = [
    "ADCTimingConfig",
    "ControllerConfig",
    "ControllerKind",
    "DigitalFilterConfig",
    "ExternalSenseConfig",
    "LoadModel",
    "PFCControlLabConfig",
    "PFCFirmwareAlgorithmConfig",
    "PFCPowerStageConfig",
    "default_current_controller",
    "default_current_sense",
    "default_vac_sense",
    "default_vbus_sense",
    "default_voltage_controller",
]
