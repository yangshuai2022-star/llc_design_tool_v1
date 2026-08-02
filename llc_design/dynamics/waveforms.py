"""Waveform reconstruction and measurement for LLC design work points."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from .plant import DynamicPhasorModel, DynamicPhasorSteadyState, LLCPlantInputs

if TYPE_CHECKING:  # pragma: no cover
    from ..core.spec import LLCDesignSpec
    from ..models.system import SystemAnalysis


@dataclass(frozen=True)
class SignalStatistics:
    """Oscilloscope-style measurements for one waveform."""

    average: float
    rms: float
    minimum: float
    maximum: float
    peak_to_peak: float
    absolute_peak: float
    crest_factor: float
    fundamental_rms: float
    thd_percent: float
    frequency_hz: float


@dataclass(frozen=True)
class WaveformSignal:
    key: str
    label: str
    unit: str
    values: NDArray[np.float64]
    statistics: SignalStatistics
    group: str = "general"
    description: str = ""


@dataclass(frozen=True)
class WaveformBundle:
    """A synchronized collection of LLC node voltage/current waveforms."""

    time_s: NDArray[np.float64]
    switching_frequency_hz: float
    model_name: str
    signals: Mapping[str, WaveformSignal]
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, float | str] = field(default_factory=dict)

    def signal(self, key: str) -> WaveformSignal:
        try:
            return self.signals[key]
        except KeyError as exc:
            raise KeyError(f"unknown waveform signal: {key}") from exc

    def to_dataframe(self) -> pd.DataFrame:
        data: dict[str, NDArray[np.float64]] = {"time_s": self.time_s}
        for key, signal in self.signals.items():
            data[key] = signal.values
        return pd.DataFrame(data)

    def statistics_dataframe(self) -> pd.DataFrame:
        rows = []
        for key, signal in self.signals.items():
            s = signal.statistics
            rows.append({
                "key": key,
                "label": signal.label,
                "unit": signal.unit,
                "group": signal.group,
                "frequency_hz": s.frequency_hz,
                "average": s.average,
                "rms": s.rms,
                "minimum": s.minimum,
                "maximum": s.maximum,
                "peak_to_peak": s.peak_to_peak,
                "absolute_peak": s.absolute_peak,
                "crest_factor": s.crest_factor,
                "fundamental_rms": s.fundamental_rms,
                "thd_percent": s.thd_percent,
            })
        return pd.DataFrame(rows)

    def export_csv(self, directory: str | Path) -> dict[str, Path]:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        waveform_path = output / "llc_waveforms.csv"
        statistics_path = output / "llc_waveform_statistics.csv"
        self.to_dataframe().to_csv(waveform_path, index=False)
        self.statistics_dataframe().to_csv(statistics_path, index=False)
        return {"waveforms": waveform_path, "statistics": statistics_path}


def _fft_measurements(
    values: NDArray[np.float64],
    samples_per_period: int,
    switching_frequency_hz: float,
    fundamental_frequency_hz: float | None = None,
) -> tuple[float, float, float]:
    """Return fundamental RMS, THD and dominant non-DC frequency."""
    values = np.asarray(values, dtype=float)
    if len(values) < 8 or samples_per_period < 4:
        return 0.0, 0.0, 0.0
    cycles = max(1, len(values) // samples_per_period)
    usable = values[: cycles * samples_per_period]
    centered = usable - float(np.mean(usable))
    spectrum = np.fft.rfft(centered) / len(centered)
    rms_bins = np.abs(spectrum) * math.sqrt(2.0)
    if len(rms_bins) > 0:
        rms_bins[0] = 0.0
    sample_frequency_hz = samples_per_period * switching_frequency_hz
    frequency_resolution_hz = sample_frequency_hz / len(usable)
    if fundamental_frequency_hz is None:
        fundamental_index = int(np.argmax(rms_bins[1:]) + 1) if len(rms_bins) > 1 else 0
    else:
        fundamental_index = int(round(fundamental_frequency_hz / frequency_resolution_hz))
        fundamental_index = min(max(fundamental_index, 1), len(rms_bins) - 1)
    fundamental = float(rms_bins[fundamental_index]) if fundamental_index < len(rms_bins) else 0.0
    harmonic_indices = range(2 * fundamental_index, len(rms_bins), fundamental_index)
    harmonic_sq = sum(float(rms_bins[index])**2 for index in harmonic_indices)
    thd = 100.0 * math.sqrt(harmonic_sq) / max(fundamental, 1e-15)
    dominant_index = int(np.argmax(rms_bins[1:]) + 1) if len(rms_bins) > 1 else 0
    dominant_frequency = dominant_index * switching_frequency_hz / cycles
    return fundamental, thd, float(dominant_frequency)


def signal_statistics(
    values: Iterable[float] | NDArray[np.float64],
    *,
    samples_per_period: int,
    switching_frequency_hz: float,
    fundamental_frequency_hz: float | None = None,
) -> SignalStatistics:
    array = np.asarray(tuple(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError("waveform values must be a non-empty one-dimensional array")
    average = float(np.mean(array))
    rms = float(np.sqrt(np.mean(array**2)))
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    abs_peak = max(abs(minimum), abs(maximum))
    fundamental, thd, dominant_frequency = _fft_measurements(
        array, samples_per_period, switching_frequency_hz,
        fundamental_frequency_hz=fundamental_frequency_hz)
    return SignalStatistics(
        average=average,
        rms=rms,
        minimum=minimum,
        maximum=maximum,
        peak_to_peak=maximum - minimum,
        absolute_peak=abs_peak,
        crest_factor=abs_peak / max(rms, 1e-15),
        fundamental_rms=fundamental,
        thd_percent=thd,
        frequency_hz=dominant_frequency,
    )


def _deadtime_bridge_square(
    phase: NDArray[np.float64],
    physical_level_v: float,
    deadtime_fraction: float,
) -> NDArray[np.float64]:
    phase_wrapped = np.mod(phase, 2.0 * math.pi)
    result = np.where(phase_wrapped < math.pi, physical_level_v, -physical_level_v)
    if deadtime_fraction <= 0.0:
        return result.astype(float)
    dead_angle = min(max(deadtime_fraction, 0.0), 0.45) * 2.0 * math.pi
    for edge in (0.0, math.pi, 2.0 * math.pi):
        distance = np.abs((phase_wrapped - edge + math.pi) % (2.0 * math.pi) - math.pi)
        result[distance <= dead_angle / 2.0] = 0.0
    return result.astype(float)


def _gate_waveforms(
    phase: NDArray[np.float64], deadtime_fraction: float,
    primary_topology: str = "FULL_BRIDGE",
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    p = np.mod(phase, 2.0 * math.pi)
    dead_angle = min(max(deadtime_fraction, 0.0), 0.45) * 2.0 * math.pi
    margin = dead_angle / 2.0
    pos = ((p > margin) & (p < math.pi - margin)).astype(float)
    neg = ((p > math.pi + margin) & (p < 2.0 * math.pi - margin)).astype(float)
    if str(primary_topology).upper().endswith("HALF_BRIDGE"):
        zero = np.zeros_like(pos)
        return pos, neg, zero, zero
    # Full bridge: diagonal pairs Q1/Q4 and Q2/Q3.
    return pos, neg, neg, pos


def _bridge_leg_waveforms(
    q1: NDArray[np.float64],
    q2: NDArray[np.float64],
    q3: NDArray[np.float64],
    q4: NDArray[np.float64],
    bus_voltage_v: float,
    primary_topology: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return leg A/B midpoint voltages referred to the negative bus rail."""
    midpoint = 0.5 * bus_voltage_v
    leg_a = np.where(q1 > 0.5, bus_voltage_v, np.where(q2 > 0.5, 0.0, midpoint))
    if str(primary_topology).upper().endswith("HALF_BRIDGE"):
        leg_b = np.full_like(leg_a, midpoint)
    else:
        leg_b = np.where(q3 > 0.5, bus_voltage_v, np.where(q4 > 0.5, 0.0, midpoint))
    return leg_a.astype(float), leg_b.astype(float)


def _integrate_periodic_zero_mean(current: NDArray[np.float64], dt: float, capacitance_f: float) -> NDArray[np.float64]:
    charge = np.cumsum(current) * dt
    # Remove any small numerical drift and center the periodic waveform.
    drift = np.linspace(0.0, float(charge[-1]), len(charge), endpoint=False)
    charge = charge - drift
    charge -= float(np.mean(charge))
    return charge / capacitance_f


def reconstruct_dynamic_phasor_waveforms(
    model: DynamicPhasorModel,
    steady_state: DynamicPhasorSteadyState,
    *,
    cycles: int = 2,
    samples_per_cycle: int = 1024,
    spec: "LLCDesignSpec | None" = None,
    system_analysis: "SystemAnalysis | None" = None,
) -> WaveformBundle:
    """Reconstruct key LLC node waveforms from a solved EDF state.

    This is the fast GUI mode.  Resonant states are fundamental waveforms,
    while the bridge and ideal synchronous rectifier retain their switching
    discontinuities.  The detailed switched solver uses the same signal names.
    """
    if cycles < 1 or samples_per_cycle < 64:
        raise ValueError("cycles must be >=1 and samples_per_cycle must be >=64")
    p = model.p
    inputs = steady_state.inputs
    fs = inputs.switching_frequency_hz
    total_samples = cycles * samples_per_cycle
    time_s = np.arange(total_samples, dtype=float) / (fs * samples_per_cycle)
    phase = 2.0 * math.pi * fs * time_s
    cos_phase = np.cos(phase)
    sin_phase = np.sin(phase)

    ir_c, ir_s, vcr_c, vcr_s, im_c, im_s, vco = steady_state.states
    ir = ir_c * cos_phase + ir_s * sin_phase
    vcr = vcr_c * cos_phase + vcr_s * sin_phase
    im = im_c * cos_phase + im_s * sin_phase
    primary_load = ir - im

    physical_bridge_level = p.bridge_gain * inputs.bus_voltage_v
    deadtime_fraction = p.primary_deadtime_s * fs
    v_bridge = _deadtime_bridge_square(phase, physical_bridge_level, deadtime_fraction)

    rectifier_sign = np.sign(primary_load)
    rectifier_sign[rectifier_sign == 0.0] = 1.0
    # Use the solved terminal voltage for the clamp.  The equivalent SR drop is
    # included only in the transformer clamp, not in the delivered output.
    vp = p.turns_ratio * (
        steady_state.output_voltage_v + p.rectifier_equivalent_drop_v
    ) * rectifier_sign
    vs = vp / p.turns_ratio
    secondary_current = p.turns_ratio * primary_load
    rectified_current = np.abs(secondary_current)
    output_current = steady_state.output_voltage_v / p.load_resistance_ohm
    output_cap_current = rectified_current - output_current
    dt = 1.0 / (fs * samples_per_cycle)
    output_cap_ripple = _integrate_periodic_zero_mean(
        output_cap_current, dt, p.output_capacitance_f)
    output_voltage = (
        steady_state.output_voltage_v
        + output_cap_ripple
        + p.output_cap_esr_ohm * output_cap_current
    )
    vlr = v_bridge - vcr - vp - p.series_resistance_ohm * ir
    magnetic_energy = 0.5 * p.lr_h * ir**2
    q1, q2, q3, q4 = _gate_waveforms(
        phase, deadtime_fraction, p.primary_topology)
    v_leg_a, v_leg_b = _bridge_leg_waveforms(
        q1, q2, q3, q4, inputs.bus_voltage_v, p.primary_topology)
    # Idealized device VDS traces are sufficient for node selection and timing
    # inspection.  Nonlinear Coss commutation belongs to the future Level-3
    # switching-device model and is not hidden behind these traces.
    vds_q1 = inputs.bus_voltage_v - v_leg_a
    vds_q2 = v_leg_a
    vds_q3 = inputs.bus_voltage_v - v_leg_b
    vds_q4 = v_leg_b
    i_q1 = q1 * ir
    i_q2 = q2 * ir
    i_q3 = q3 * ir
    i_q4 = q4 * ir

    positive_secondary = (secondary_current >= 0.0).astype(float)
    negative_secondary = 1.0 - positive_secondary
    gate_sr1 = positive_secondary
    gate_sr4 = positive_secondary
    gate_sr2 = negative_secondary
    gate_sr3 = negative_secondary
    i_sr1 = np.maximum(secondary_current, 0.0)
    i_sr4 = i_sr1.copy()
    i_sr2 = np.maximum(-secondary_current, 0.0)
    i_sr3 = i_sr2.copy()
    sr_block_voltage = np.maximum(np.abs(vs), steady_state.output_voltage_v)
    vds_sr1 = np.where(gate_sr1 > 0.5, 0.0, sr_block_voltage)
    vds_sr4 = vds_sr1.copy()
    vds_sr2 = np.where(gate_sr2 > 0.5, 0.0, sr_block_voltage)
    vds_sr3 = vds_sr2.copy()
    output_load_current = output_voltage / p.load_resistance_ohm
    output_cap_internal_voltage = output_voltage - p.output_cap_esr_ohm * output_cap_current

    optional_magnetic: dict[str, tuple[str, str, NDArray[np.float64], str, str, float]] = {}
    if p.primary_turns > 0 and p.transformer_core_area_m2 > 0.0:
        transformer_flux = _integrate_periodic_zero_mean(
            vp, dt, p.primary_turns * p.transformer_core_area_m2)
        if p.transformer_magnetic_path_m > 0.0:
            transformer_h = p.primary_turns * im / p.transformer_magnetic_path_m
        else:
            transformer_h = np.zeros_like(im)
        optional_magnetic.update({
            "b_transformer": (
                "变压器磁通密度 B", "T", transformer_flux, "magnetics",
                "由 Vp/(Np*Ae) 周期积分得到", fs),
            "h_transformer": (
                "变压器励磁场 H", "A/m", transformer_h, "magnetics",
                "按 Np*Im/le 的一维等效场", fs),
        })
    if p.resonant_inductor_turns > 0 and p.resonant_inductor_core_area_m2 > 0.0:
        b_lr = p.lr_h * ir / (
            p.resonant_inductor_turns * p.resonant_inductor_core_area_m2)
        if p.resonant_inductor_magnetic_path_m > 0.0:
            h_lr = p.resonant_inductor_turns * ir / p.resonant_inductor_magnetic_path_m
        else:
            h_lr = np.zeros_like(ir)
        optional_magnetic.update({
            "b_resonant_inductor": (
                "谐振电感磁通密度 B", "T", b_lr, "magnetics",
                "按 Lr*Ir/(N*Ae) 计算", fs),
            "h_resonant_inductor": (
                "谐振电感等效磁场 H", "A/m", h_lr, "magnetics",
                "按 N*Ir/le 的一维等效场；气隙局部场另由损耗模型处理", fs),
        })

    raw: dict[str, tuple[str, str, NDArray[np.float64], str, str, float]] = {
        "v_leg_a": ("A 桥臂中点电压 VA", "V", v_leg_a, "primary", "相对母线负端", fs),
        "v_leg_b": ("B 桥臂中点电压 VB", "V", v_leg_b, "primary", "全桥 B 桥臂；半桥时为母线中点", fs),
        "v_bridge": ("桥臂差模输出电压 Vab", "V", v_bridge, "primary", "一次全桥/半桥施加到谐振腔的电压", fs),
        "i_resonant": ("谐振电流 Ir", "A", ir, "primary", "Lr、Cr 与变压器原边串联电流", fs),
        "v_resonant_cap": ("谐振电容电压 VCr", "V", vcr, "primary", "谐振电容两端电压", fs),
        "v_resonant_inductor": ("谐振电感电压 VLr", "V", vlr, "primary", "谐振电感有效端电压", fs),
        "v_transformer_primary": ("变压器原边电压 Vp", "V", vp, "transformer", "理想整流钳位下的原边电压", fs),
        "i_transformer_primary": ("变压器原边电流 Ip", "A", ir, "transformer", "流入变压器端口的总原边电流", fs),
        "i_magnetizing": ("励磁电流 Im", "A", im, "transformer", "变压器励磁支路电流", fs),
        "i_primary_load": ("原边负载分量 Iload,p", "A", primary_load, "transformer", "Ir-Im，折算到原边的负载电流", fs),
        "v_transformer_secondary": ("变压器次级电压 Vs", "V", vs, "secondary", "未整流的次级绕组电压", fs),
        "i_transformer_secondary": ("变压器次级电流 Is", "A", secondary_current, "secondary", "未整流的次级绕组电流", fs),
        "v_rectified": ("SR 整流后电压 Vrect", "V", np.abs(vs), "secondary", "全桥 SR 理想整流电压", 2.0*fs),
        "i_rectified": ("SR 整流输出电流 Irect", "A", rectified_current, "secondary", "全桥同步整流后的脉动电流", 2.0*fs),
        "i_load_output": ("输出负载电流 Io", "A", output_load_current, "output", "端口电压除以等效负载", 2.0*fs),
        "i_output_cap": ("输出电容电流 ICo", "A", output_cap_current, "output", "整流电流减去负载电流", 2.0*fs),
        "v_output_cap_internal": ("输出电容内部电压 VCo", "V", output_cap_internal_voltage, "output", "不含 ESR 瞬时压降", 2.0*fs),
        "v_output": ("输出端电压 Vo", "V", output_voltage, "output", "包含容量纹波与 ESR 纹波", 2.0*fs),
        "v_output_ripple": ("输出纹波 ΔVo", "V", output_voltage - np.mean(output_voltage), "output", "去除直流量后的输出纹波", 2.0*fs),
        "energy_lr": ("谐振电感储能 ELr", "J", magnetic_energy, "magnetics", "0.5*Lr*Ir^2", 2.0*fs),
        "gate_q1": ("Q1 门极逻辑", "pu", q1, "switching", "理想化一次侧门极逻辑", fs),
        "gate_q2": ("Q2 门极逻辑", "pu", q2, "switching", "理想化一次侧门极逻辑", fs),
        "gate_q3": ("Q3 门极逻辑", "pu", q3, "switching", "理想化一次侧门极逻辑", fs),
        "gate_q4": ("Q4 门极逻辑", "pu", q4, "switching", "理想化一次侧门极逻辑", fs),
        "vds_q1": ("Q1 VDS", "V", vds_q1, "switching", "理想开关 VDS；死区中点取 Vbus/2", fs),
        "vds_q2": ("Q2 VDS", "V", vds_q2, "switching", "理想开关 VDS；死区中点取 Vbus/2", fs),
        "vds_q3": ("Q3 VDS", "V", vds_q3, "switching", "全桥 B 臂上管 VDS", fs),
        "vds_q4": ("Q4 VDS", "V", vds_q4, "switching", "全桥 B 臂下管 VDS", fs),
        "ids_q1": ("Q1 支路电流", "A", i_q1, "switching", "理想导通窗口内的有符号谐振电流", fs),
        "ids_q2": ("Q2 支路电流", "A", i_q2, "switching", "理想导通窗口内的有符号谐振电流", fs),
        "ids_q3": ("Q3 支路电流", "A", i_q3, "switching", "理想导通窗口内的有符号谐振电流", fs),
        "ids_q4": ("Q4 支路电流", "A", i_q4, "switching", "理想导通窗口内的有符号谐振电流", fs),
        "gate_sr1": ("SR1 门极逻辑", "pu", gate_sr1, "sr_switching", "理想电流极性驱动", fs),
        "gate_sr2": ("SR2 门极逻辑", "pu", gate_sr2, "sr_switching", "理想电流极性驱动", fs),
        "gate_sr3": ("SR3 门极逻辑", "pu", gate_sr3, "sr_switching", "理想电流极性驱动", fs),
        "gate_sr4": ("SR4 门极逻辑", "pu", gate_sr4, "sr_switching", "理想电流极性驱动", fs),
        "ids_sr1": ("SR1 电流", "A", i_sr1, "sr_switching", "全桥 SR 正向对角器件电流", fs),
        "ids_sr2": ("SR2 电流", "A", i_sr2, "sr_switching", "全桥 SR 反向对角器件电流", fs),
        "ids_sr3": ("SR3 电流", "A", i_sr3, "sr_switching", "全桥 SR 反向对角器件电流", fs),
        "ids_sr4": ("SR4 电流", "A", i_sr4, "sr_switching", "全桥 SR 正向对角器件电流", fs),
        "vds_sr1": ("SR1 VDS", "V", vds_sr1, "sr_switching", "理想全桥 SR 阻断电压", fs),
        "vds_sr2": ("SR2 VDS", "V", vds_sr2, "sr_switching", "理想全桥 SR 阻断电压", fs),
        "vds_sr3": ("SR3 VDS", "V", vds_sr3, "sr_switching", "理想全桥 SR 阻断电压", fs),
        "vds_sr4": ("SR4 VDS", "V", vds_sr4, "sr_switching", "理想全桥 SR 阻断电压", fs),
    }
    raw.update(optional_magnetic)
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
                switching_frequency_hz=fs,
                fundamental_frequency_hz=fundamental_hz,
            ),
            group=group,
            description=description,
        )

    return WaveformBundle(
        time_s=time_s,
        switching_frequency_hz=fs,
        model_name="dynamic_phasor_edf_fast",
        signals=signals,
        warnings=(
            "Fast waveform mode retains only fundamental resonant-state components.",
            "Primary VDS traces are ideal switch states; nonlinear Coss commutation is not included.",
        ),
        metadata={
            "output_voltage_v": steady_state.output_voltage_v,
            "series_resistance_ohm": p.series_resistance_ohm,
            "cycles": cycles,
            "samples_per_cycle": samples_per_cycle,
        },
    )
