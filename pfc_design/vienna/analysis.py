"""Small-signal nested-loop analysis for three-phase Vienna PFC."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from numpy.typing import NDArray

from llc_design.control.digital_loop import DigitalTransferFunction, StabilityMargins, calculate_stability_margins
from pfc_design.control.sensing import SenseFrequencyResponse, SenseChainSummary, sense_frequency_response, summarize_sense_chain
from .config import ViennaControlLabConfig

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class ViennaLoopResult:
    name: str
    controller: DigitalTransferFunction
    responses: dict[str, ComplexArray]
    margins: StabilityMargins
    likely_stable: bool


@dataclass(frozen=True)
class ViennaControlLabAnalysis:
    config: ViennaControlLabConfig
    frequencies_hz: FloatArray
    current_loop: ViennaLoopResult
    voltage_loop: ViennaLoopResult
    balance_loop: ViennaLoopResult
    current_sense_response: SenseFrequencyResponse
    phase_voltage_sense_response: SenseFrequencyResponse
    split_bus_sense_response: SenseFrequencyResponse
    current_sense_summary: SenseChainSummary
    split_bus_sense_summary: SenseChainSummary
    warnings: tuple[str, ...]


def _zoh(f: FloatArray, ts: float) -> ComplexArray:
    return np.sinc(f*ts)*np.exp(-1j*math.pi*f*ts)


def _stable(m: StabilityMargins) -> bool:
    pm = m.phase_margin_deg
    gm = m.gain_margin_db
    return (pm is None or pm > 0) and (gm is None or gm > 0)


def build_vienna_control_lab_analysis(config: ViennaControlLabConfig) -> ViennaControlLabAnalysis:
    config.validate()
    stage, fw = config.power_stage, config.firmware
    f = np.geomspace(config.frequency_start_hz, config.frequency_stop_hz, config.frequency_points)
    s = 1j*2*math.pi*f
    hi = sense_frequency_response(config.phase_current_sense, f)
    hvac = sense_frequency_response(config.phase_voltage_sense, f)
    hvdc = sense_frequency_response(config.split_bus_sense, f)

    ci = config.current_controller.transfer_function(); ci_r = ci.frequency_response(f)
    current_plant = (0.5*stage.bus_voltage_v)/(stage.boost_inductance_h*s + stage.phase_series_resistance_ohm)
    pwm_i = _zoh(f, 1/fw.current_loop_rate_hz)*np.exp(-1j*2*math.pi*f*(fw.current_computation_delay_s+fw.pwm_update_delay_s))
    forward_i = ci_r*pwm_i*current_plant
    open_i = forward_i*hi.total
    ti_actual = forward_i/(1+open_i)
    current_responses = {
        "plant_gid": current_plant, "controller_ci": ci_r, "pwm_zoh": pwm_i,
        "sense_hi": hi.total, "sense_hi_analog": hi.calibrated_analog,
        "forward_current": forward_i, "open_current": open_i,
        "closed_current_actual": ti_actual, "sensitivity_current":1/(1+open_i),
    }
    mi = calculate_stability_margins(f, open_i)

    cv = config.voltage_controller.transfer_function(); cv_r=cv.frequency_response(f)
    ceq = stage.series_bus_capacitance_f
    # dP/dG = 3*Vphase_rms^2.  Vdc energy plant includes series split caps.
    gvg = (3*stage.phase_rms_v**2)/(max(stage.bus_voltage_v,1e-9)*ceq*s)
    pwm_v = _zoh(f,1/fw.voltage_loop_rate_hz)*np.exp(-1j*2*math.pi*f*fw.voltage_computation_delay_s)
    forward_v = cv_r*pwm_v*ti_actual*gvg
    open_v = forward_v*hvdc.total
    voltage_responses = {
        "controller_cv":cv_r,"current_closed_for_outer":ti_actual,
        "bus_plant_gvg":gvg,"sense_hv":hvdc.total,"forward_voltage":forward_v,
        "open_voltage":open_v,"closed_voltage":forward_v/(1+open_v),
        "sensitivity_voltage":1/(1+open_v),
    }
    mv=calculate_stability_margins(f,open_v)

    cb=config.balance_controller.transfer_function(); cb_r=cb.frequency_response(f)
    # A positive balance duty offset is defined to drive positive midpoint
    # current and reduce (Vdc+ - Vdc-).  The closed-loop sign is handled by
    # the feedback summing junction; this is the positive plant magnitude.
    inv_c_delta = (1.0/max(stage.upper_bus_capacitance_f,1e-12)
                   + 1.0/max(stage.lower_bus_capacitance_f,1e-12))
    gbal=fw.midpoint_current_gain_a_per_pu*inv_c_delta/s
    pwm_b=_zoh(f,1/fw.balance_loop_rate_hz)
    forward_b=cb_r*pwm_b*gbal
    open_b=forward_b*hvdc.total
    balance_responses={
        "controller_cb":cb_r,"balance_plant":gbal,"sense_balance":hvdc.total,
        "forward_balance":forward_b,"open_balance":open_b,
        "closed_balance":forward_b/(1+open_b),"sensitivity_balance":1/(1+open_b),
    }
    mb=calculate_stability_margins(f,open_b)

    warnings=[]
    if not _stable(mi): warnings.append("Vienna phase-current loop has non-positive margin.")
    if not _stable(mv): warnings.append("Vienna DC-voltage loop has non-positive margin.")
    if not _stable(mb): warnings.append("Vienna midpoint-balance loop has non-positive margin.")
    return ViennaControlLabAnalysis(
        config=config, frequencies_hz=f,
        current_loop=ViennaLoopResult("Phase current",ci,current_responses,mi,_stable(mi)),
        voltage_loop=ViennaLoopResult("DC voltage",cv,voltage_responses,mv,_stable(mv)),
        balance_loop=ViennaLoopResult("Midpoint balance",cb,balance_responses,mb,_stable(mb)),
        current_sense_response=hi, phase_voltage_sense_response=hvac, split_bus_sense_response=hvdc,
        current_sense_summary=summarize_sense_chain(config.phase_current_sense),
        split_bus_sense_summary=summarize_sense_chain(config.split_bus_sense),
        warnings=tuple(warnings),
    )


__all__=["ViennaControlLabAnalysis","ViennaLoopResult","build_vienna_control_lab_analysis"]
