"""Standard waveform-bundle construction for V8 steady-state solvers."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from ..dynamics.waveforms import (
    WaveformBundle,
    WaveformSignal,
    bridge_leg_waveforms,
    gate_waveforms,
    signal_statistics,
)


def _solve_periodic_output_network(
    rectified_current_a: NDArray[np.float64],
    *,
    output_voltage_mean_v: float,
    load_resistance_ohm: float,
    output_capacitance_f: float,
    output_cap_esr_ohm: float,
    sample_interval_s: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Solve the periodic resistive-load/output-capacitor network by FFT.

    The output capacitor is represented by ideal C in series with ESR.  This
    reconstruction enforces KCL at every retained discrete harmonic instead of
    integrating ``Irect - constant Io``, which becomes inaccurate when ESR and
    output ripple are non-negligible.
    """

    current = np.asarray(rectified_current_a, dtype=float)
    count = len(current)
    if count < 8 or sample_interval_s <= 0.0:
        raise ValueError("periodic output reconstruction requires valid samples and dt")
    current_ac = current - float(np.mean(current))
    spectrum = np.fft.rfft(current_ac)
    frequencies = np.fft.rfftfreq(count, d=sample_interval_s)
    output_spectrum = np.zeros_like(spectrum, dtype=np.complex128)
    for index in range(1, len(spectrum)):
        omega = 2.0 * math.pi * frequencies[index]
        z_cap = output_cap_esr_ohm + 1.0 / (1j * omega * output_capacitance_f)
        z_parallel = 1.0 / (1.0 / load_resistance_ohm + 1.0 / z_cap)
        output_spectrum[index] = spectrum[index] * z_parallel
    output_ac = np.fft.irfft(output_spectrum, n=count)
    output_voltage = output_voltage_mean_v + output_ac
    load_current = output_voltage / load_resistance_ohm
    capacitor_current = current - load_current
    capacitor_internal_voltage = output_voltage - output_cap_esr_ohm * capacitor_current
    return output_voltage, capacitor_internal_voltage, load_current, capacitor_current


def build_standard_waveform_bundle(
    *,
    time_s: NDArray[np.float64],
    switching_frequency_hz: float,
    model_name: str,
    bus_voltage_v: float,
    primary_topology: str,
    primary_deadtime_s: float,
    turns_ratio: float,
    load_resistance_ohm: float,
    output_capacitance_f: float,
    output_cap_esr_ohm: float,
    series_resistance_ohm: float,
    resonant_inductance_h: float,
    output_voltage_mean_v: float,
    bridge_voltage_v: NDArray[np.float64],
    resonant_current_a: NDArray[np.float64],
    resonant_capacitor_voltage_v: NDArray[np.float64],
    magnetizing_current_a: NDArray[np.float64],
    transformer_primary_voltage_v: NDArray[np.float64],
    warnings: tuple[str, ...] = (),
    metadata: Mapping[str, float | str] | None = None,
) -> WaveformBundle:
    """Create the common LLC waveform schema consumed by GUI/export modules.

    The supplied arrays are the solver-owned electrical states.  Secondary,
    output-capacitor, bridge-device and ideal-SR traces are derived here using
    one convention so FHA and harmonic-balance results remain directly
    comparable to the existing switched solver.
    """

    arrays = [
        np.asarray(time_s, dtype=float),
        np.asarray(bridge_voltage_v, dtype=float),
        np.asarray(resonant_current_a, dtype=float),
        np.asarray(resonant_capacitor_voltage_v, dtype=float),
        np.asarray(magnetizing_current_a, dtype=float),
        np.asarray(transformer_primary_voltage_v, dtype=float),
    ]
    length = len(arrays[0])
    if length < 8 or any(array.ndim != 1 or len(array) != length for array in arrays):
        raise ValueError("all waveform inputs must be equal-length one-dimensional arrays")
    if switching_frequency_hz <= 0.0 or turns_ratio <= 0.0:
        raise ValueError("switching frequency and turns ratio must be positive")
    if load_resistance_ohm <= 0.0 or output_capacitance_f <= 0.0:
        raise ValueError("load resistance and output capacitance must be positive")
    if resonant_inductance_h <= 0.0:
        raise ValueError("resonant inductance must be positive")

    time = arrays[0]
    bridge = arrays[1]
    ir = arrays[2]
    vcr = arrays[3]
    im = arrays[4]
    vp = arrays[5]
    dt = float(np.mean(np.diff(time)))
    phase = 2.0 * math.pi * switching_frequency_hz * time
    samples_per_cycle = max(1, int(round(1.0 / (switching_frequency_hz * dt))))

    primary_load = ir - im
    vs = vp / turns_ratio
    isec = turns_ratio * primary_load
    irect = np.abs(isec)
    vout, vco, output_load_current, ico = _solve_periodic_output_network(
        irect,
        output_voltage_mean_v=output_voltage_mean_v,
        load_resistance_ohm=load_resistance_ohm,
        output_capacitance_f=output_capacitance_f,
        output_cap_esr_ohm=output_cap_esr_ohm,
        sample_interval_s=dt,
    )
    vlr = bridge - vcr - vp - series_resistance_ohm * ir
    energy = 0.5 * resonant_inductance_h * ir * ir

    deadtime_fraction = primary_deadtime_s * switching_frequency_hz
    q1, q2, q3, q4 = gate_waveforms(
        phase, deadtime_fraction, primary_topology)
    v_leg_a, v_leg_b = bridge_leg_waveforms(
        q1, q2, q3, q4, bus_voltage_v, primary_topology)
    vds_q1 = bus_voltage_v - v_leg_a
    vds_q2 = v_leg_a
    vds_q3 = bus_voltage_v - v_leg_b
    vds_q4 = v_leg_b
    i_q1 = q1 * ir
    i_q2 = q2 * ir
    i_q3 = q3 * ir
    i_q4 = q4 * ir

    positive_secondary = (isec >= 0.0).astype(float)
    negative_secondary = 1.0 - positive_secondary
    gate_sr1 = positive_secondary
    gate_sr4 = positive_secondary
    gate_sr2 = negative_secondary
    gate_sr3 = negative_secondary
    i_sr1 = np.maximum(isec, 0.0)
    i_sr4 = i_sr1.copy()
    i_sr2 = np.maximum(-isec, 0.0)
    i_sr3 = i_sr2.copy()
    sr_block_voltage = np.maximum(np.abs(vs), vout)
    vds_sr1 = np.where(gate_sr1 > 0.5, 0.0, sr_block_voltage)
    vds_sr4 = vds_sr1.copy()
    vds_sr2 = np.where(gate_sr2 > 0.5, 0.0, sr_block_voltage)
    vds_sr3 = vds_sr2.copy()

    raw: dict[str, tuple[str, str, NDArray[np.float64], str, str, float]] = {
        "v_leg_a": ("A 桥臂中点电压 VA", "V", v_leg_a, "primary", "相对母线负端", switching_frequency_hz),
        "v_leg_b": ("B 桥臂中点电压 VB", "V", v_leg_b, "primary", "全桥 B 桥臂；半桥时为母线中点", switching_frequency_hz),
        "v_bridge": ("桥臂差模输出电压 Vab", "V", bridge, "primary", "施加到 LLC 谐振腔的桥臂电压", switching_frequency_hz),
        "i_resonant": ("谐振电流 Ir", "A", ir, "primary", "Lr/Cr 串联谐振电流", switching_frequency_hz),
        "v_resonant_cap": ("谐振电容电压 VCr", "V", vcr, "primary", "谐振电容两端电压", switching_frequency_hz),
        "v_resonant_inductor": ("谐振电感电压 VLr", "V", vlr, "primary", "由 KVL 重构的 Lr 端电压", switching_frequency_hz),
        "v_transformer_primary": ("变压器原边电压 Vp", "V", vp, "transformer", "理想整流钳位原边电压", switching_frequency_hz),
        "i_transformer_primary": ("变压器原边电流 Ip", "A", ir, "transformer", "流入变压器端口的总原边电流", switching_frequency_hz),
        "i_magnetizing": ("励磁电流 Im", "A", im, "transformer", "励磁支路电流", switching_frequency_hz),
        "i_primary_load": ("原边负载分量 Iload,p", "A", primary_load, "transformer", "Ir-Im", switching_frequency_hz),
        "v_transformer_secondary": ("变压器次级电压 Vs", "V", vs, "secondary", "未整流次级电压", switching_frequency_hz),
        "i_transformer_secondary": ("变压器次级电流 Is", "A", isec, "secondary", "未整流次级绕组电流", switching_frequency_hz),
        "v_rectified": ("SR 整流后电压 Vrect", "V", np.abs(vs), "secondary", "全桥 SR 理想整流电压", 2.0 * switching_frequency_hz),
        "i_rectified": ("SR 整流输出电流 Irect", "A", irect, "secondary", "理想同步整流后的脉动电流", 2.0 * switching_frequency_hz),
        "i_load_output": ("输出负载电流 Io", "A", output_load_current, "output", "输出端口电压除以负载电阻", 2.0 * switching_frequency_hz),
        "i_output_cap": ("输出电容电流 ICo", "A", ico, "output", "整流电流减去直流负载电流", 2.0 * switching_frequency_hz),
        "v_output_cap_internal": ("输出电容内部电压 VCo", "V", vco, "output", "不含 ESR 瞬时压降", 2.0 * switching_frequency_hz),
        "v_output": ("输出端电压 Vo", "V", vout, "output", "包含容量纹波与 ESR 纹波", 2.0 * switching_frequency_hz),
        "v_output_ripple": ("输出纹波 ΔVo", "V", vout - np.mean(vout), "output", "去除直流量后的输出纹波", 2.0 * switching_frequency_hz),
        "energy_lr": ("谐振电感储能 ELr", "J", energy, "magnetics", "0.5*Lr*Ir²", 2.0 * switching_frequency_hz),
        "gate_q1": ("Q1 门极逻辑", "pu", q1, "switching", "理想一次侧门极逻辑", switching_frequency_hz),
        "gate_q2": ("Q2 门极逻辑", "pu", q2, "switching", "理想一次侧门极逻辑", switching_frequency_hz),
        "gate_q3": ("Q3 门极逻辑", "pu", q3, "switching", "理想一次侧门极逻辑", switching_frequency_hz),
        "gate_q4": ("Q4 门极逻辑", "pu", q4, "switching", "理想一次侧门极逻辑", switching_frequency_hz),
        "vds_q1": ("Q1 VDS", "V", vds_q1, "switching", "理想开关状态，不含非线性 Coss", switching_frequency_hz),
        "vds_q2": ("Q2 VDS", "V", vds_q2, "switching", "理想开关状态，不含非线性 Coss", switching_frequency_hz),
        "vds_q3": ("Q3 VDS", "V", vds_q3, "switching", "理想开关状态，不含非线性 Coss", switching_frequency_hz),
        "vds_q4": ("Q4 VDS", "V", vds_q4, "switching", "理想开关状态，不含非线性 Coss", switching_frequency_hz),
        "ids_q1": ("Q1 支路电流", "A", i_q1, "switching", "理想导通窗口内有符号电流", switching_frequency_hz),
        "ids_q2": ("Q2 支路电流", "A", i_q2, "switching", "理想导通窗口内有符号电流", switching_frequency_hz),
        "ids_q3": ("Q3 支路电流", "A", i_q3, "switching", "理想导通窗口内有符号电流", switching_frequency_hz),
        "ids_q4": ("Q4 支路电流", "A", i_q4, "switching", "理想导通窗口内有符号电流", switching_frequency_hz),
        "gate_sr1": ("SR1 门极逻辑", "pu", gate_sr1, "sr_switching", "理想电流极性驱动", switching_frequency_hz),
        "gate_sr2": ("SR2 门极逻辑", "pu", gate_sr2, "sr_switching", "理想电流极性驱动", switching_frequency_hz),
        "gate_sr3": ("SR3 门极逻辑", "pu", gate_sr3, "sr_switching", "理想电流极性驱动", switching_frequency_hz),
        "gate_sr4": ("SR4 门极逻辑", "pu", gate_sr4, "sr_switching", "理想电流极性驱动", switching_frequency_hz),
        "ids_sr1": ("SR1 电流", "A", i_sr1, "sr_switching", "正向对角器件电流", switching_frequency_hz),
        "ids_sr2": ("SR2 电流", "A", i_sr2, "sr_switching", "反向对角器件电流", switching_frequency_hz),
        "ids_sr3": ("SR3 电流", "A", i_sr3, "sr_switching", "反向对角器件电流", switching_frequency_hz),
        "ids_sr4": ("SR4 电流", "A", i_sr4, "sr_switching", "正向对角器件电流", switching_frequency_hz),
        "vds_sr1": ("SR1 VDS", "V", vds_sr1, "sr_switching", "理想 SR 阻断电压", switching_frequency_hz),
        "vds_sr2": ("SR2 VDS", "V", vds_sr2, "sr_switching", "理想 SR 阻断电压", switching_frequency_hz),
        "vds_sr3": ("SR3 VDS", "V", vds_sr3, "sr_switching", "理想 SR 阻断电压", switching_frequency_hz),
        "vds_sr4": ("SR4 VDS", "V", vds_sr4, "sr_switching", "理想 SR 阻断电压", switching_frequency_hz),
    }

    signals: dict[str, WaveformSignal] = {}
    for key, (label, unit, values, group, description, fundamental_hz) in raw.items():
        values = np.asarray(values, dtype=float)
        signals[key] = WaveformSignal(
            key=key,
            label=label,
            unit=unit,
            values=values,
            statistics=signal_statistics(
                values,
                samples_per_period=samples_per_cycle,
                switching_frequency_hz=switching_frequency_hz,
                fundamental_frequency_hz=fundamental_hz,
            ),
            group=group,
            description=description,
        )

    return WaveformBundle(
        time_s=time,
        switching_frequency_hz=switching_frequency_hz,
        model_name=model_name,
        signals=signals,
        warnings=warnings,
        metadata=dict(metadata or {}),
    )
