"""Piecewise nonlinear LLC steady-state waveform solver.

This Level-2 model integrates the actual switching waveform while retaining an
ideal transformer and ideal full-bridge synchronous rectifier clamp.  It is
more detailed than the fast dynamic-phasor reconstruction and is intended for
waveform/stress verification, not for large optimization sweeps.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root

from .plant import DynamicPhasorModel, DynamicPhasorSteadyState, LLCPlantInputs
from .waveforms import (
    WaveformBundle,
    WaveformSignal,
    _bridge_leg_waveforms,
    _deadtime_bridge_square,
    _gate_waveforms,
    _integrate_periodic_zero_mean,
    signal_statistics,
)


@dataclass(frozen=True)
class SwitchedSimulationConfig:
    samples_per_cycle: int = 512
    output_cycles: int = 2
    minimum_settling_cycles: int = 20
    maximum_settling_cycles: int = 500
    convergence_tolerance: float = 1e-8
    rectifier_smoothing_current_a: float = 0.02
    use_periodic_shooting: bool = True
    shooting_max_evaluations: int = 80

    def validate(self) -> None:
        if self.samples_per_cycle < 128:
            raise ValueError("switched solver requires at least 128 samples per cycle")
        if self.output_cycles < 1:
            raise ValueError("output_cycles must be >= 1")
        if self.minimum_settling_cycles < 1:
            raise ValueError("minimum_settling_cycles must be >= 1")
        if self.maximum_settling_cycles < self.minimum_settling_cycles:
            raise ValueError("maximum_settling_cycles must exceed minimum_settling_cycles")
        if self.convergence_tolerance <= 0.0:
            raise ValueError("convergence tolerance must be positive")
        if self.shooting_max_evaluations < 10:
            raise ValueError("shooting_max_evaluations must be >= 10")


@dataclass(frozen=True)
class _Instantaneous:
    bridge_voltage_v: float
    output_voltage_v: float
    primary_voltage_v: float
    secondary_voltage_v: float
    secondary_current_a: float
    rectified_current_a: float
    output_cap_current_a: float


def _bridge_value(phase: float, level: float, deadtime_fraction: float) -> float:
    wrapped = phase % (2.0 * math.pi)
    dead_angle = min(max(deadtime_fraction, 0.0), 0.45) * 2.0 * math.pi
    if dead_angle > 0.0:
        for edge in (0.0, math.pi, 2.0 * math.pi):
            distance = abs((wrapped - edge + math.pi) % (2.0 * math.pi) - math.pi)
            if distance <= dead_angle / 2.0:
                return 0.0
    return level if wrapped < math.pi else -level


def _instantaneous(
    model: DynamicPhasorModel,
    inputs: LLCPlantInputs,
    phase: float,
    state: NDArray[np.float64],
    smoothing_current_a: float,
) -> _Instantaneous:
    p = model.p
    ir, _, im, vco = map(float, state)
    primary_load = ir - im
    secondary_current = p.turns_ratio * primary_load
    rectified_current = abs(secondary_current)
    rload = p.load_resistance_ohm
    esr = p.output_cap_esr_ohm
    i_dist = inputs.load_current_disturbance_a
    vout = (
        rload * vco + rload * esr * (rectified_current - i_dist)
    ) / (rload + esr)
    smoothing = max(smoothing_current_a, 1e-9)
    polarity = math.tanh(primary_load / smoothing)
    primary_voltage = p.turns_ratio * max(
        vout + p.rectifier_equivalent_drop_v, 0.0
    ) * polarity
    bridge = _bridge_value(
        phase,
        p.bridge_gain * inputs.bus_voltage_v,
        p.primary_deadtime_s * inputs.switching_frequency_hz,
    )
    cap_current = rectified_current - vout / rload - i_dist
    return _Instantaneous(
        bridge_voltage_v=bridge,
        output_voltage_v=vout,
        primary_voltage_v=primary_voltage,
        secondary_voltage_v=primary_voltage / p.turns_ratio,
        secondary_current_a=secondary_current,
        rectified_current_a=rectified_current,
        output_cap_current_a=cap_current,
    )


def _rhs(
    model: DynamicPhasorModel,
    inputs: LLCPlantInputs,
    phase: float,
    state: NDArray[np.float64],
    smoothing_current_a: float,
) -> NDArray[np.float64]:
    p = model.p
    ir, vcr, im, _ = map(float, state)
    alg = _instantaneous(model, inputs, phase, state, smoothing_current_a)
    return np.asarray(
        [
            (
                alg.bridge_voltage_v
                - vcr
                - alg.primary_voltage_v
                - p.series_resistance_ohm * ir
            ) / p.lr_h,
            ir / p.cr_f,
            (
                alg.primary_voltage_v
                - p.magnetizing_series_resistance_ohm * im
            ) / p.lm_h,
            alg.output_cap_current_a / p.output_capacitance_f,
        ],
        dtype=float,
    )


def _rk4_step(
    model: DynamicPhasorModel,
    inputs: LLCPlantInputs,
    phase: float,
    state: NDArray[np.float64],
    dt: float,
    smoothing_current_a: float,
) -> NDArray[np.float64]:
    omega = 2.0 * math.pi * inputs.switching_frequency_hz
    k1 = _rhs(model, inputs, phase, state, smoothing_current_a)
    k2 = _rhs(
        model, inputs, phase + omega * dt / 2.0,
        state + dt * k1 / 2.0, smoothing_current_a)
    k3 = _rhs(
        model, inputs, phase + omega * dt / 2.0,
        state + dt * k2 / 2.0, smoothing_current_a)
    k4 = _rhs(
        model, inputs, phase + omega * dt,
        state + dt * k3, smoothing_current_a)
    return state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def _advance_one_cycle(
    model: DynamicPhasorModel,
    inputs: LLCPlantInputs,
    state: NDArray[np.float64],
    dt: float,
    samples_per_cycle: int,
    smoothing_current_a: float,
) -> NDArray[np.float64]:
    phase = 0.0
    omega = 2.0 * math.pi * inputs.switching_frequency_hz
    result = np.asarray(state, dtype=float).copy()
    for _ in range(samples_per_cycle):
        result = _rk4_step(
            model, inputs, phase, result, dt, smoothing_current_a)
        phase += omega * dt
    return result


def simulate_switched_steady_state(
    model: DynamicPhasorModel,
    steady_state: DynamicPhasorSteadyState,
    config: SwitchedSimulationConfig | None = None,
) -> WaveformBundle:
    """Settle and record the nonlinear piecewise LLC waveform."""
    cfg = config or SwitchedSimulationConfig()
    cfg.validate()
    inputs = steady_state.inputs
    fs = inputs.switching_frequency_hz
    period = 1.0 / fs
    dt = period / cfg.samples_per_cycle
    omega = 2.0 * math.pi * fs
    # Dynamic-phasor states evaluated at theta=0 provide a close initial state.
    state = np.asarray(
        [
            steady_state.states[0],
            steady_state.states[2],
            steady_state.states[4],
            steady_state.states[6],
        ],
        dtype=float,
    )
    phase = 0.0
    scales = np.asarray(
        [
            max(steady_state.resonant_current_peak_a, 1.0),
            max(inputs.bus_voltage_v, 10.0),
            max(steady_state.magnetizing_current_peak_a, 1.0),
            max(steady_state.output_voltage_v, 1.0),
        ],
        dtype=float,
    )
    convergence = math.inf
    settled_cycles = 0
    # Short transient preconditioning keeps the periodic shooting solve on the
    # desired physical orbit without waiting for the slow output-capacitor pole.
    precondition_cycles = (
        cfg.minimum_settling_cycles
        if cfg.use_periodic_shooting else cfg.maximum_settling_cycles
    )
    for cycle in range(precondition_cycles):
        start = state.copy()
        state = _advance_one_cycle(
            model, inputs, state, dt, cfg.samples_per_cycle,
            cfg.rectifier_smoothing_current_a)
        convergence = float(np.linalg.norm((state - start) / scales))
        settled_cycles = cycle + 1
        if (
            not cfg.use_periodic_shooting
            and settled_cycles >= cfg.minimum_settling_cycles
            and convergence <= cfg.convergence_tolerance
        ):
            break

    shooting_evaluations = 0
    shooting_success = False
    if cfg.use_periodic_shooting:
        def residual(candidate: NDArray[np.float64]) -> NDArray[np.float64]:
            nonlocal shooting_evaluations
            shooting_evaluations += 1
            end = _advance_one_cycle(
                model, inputs, candidate, dt, cfg.samples_per_cycle,
                cfg.rectifier_smoothing_current_a)
            return (end - candidate) / scales

        solution = root(
            residual,
            state,
            method="hybr",
            options={
                "xtol": cfg.convergence_tolerance,
                "maxfev": cfg.shooting_max_evaluations,
            },
        )
        state = np.asarray(solution.x, dtype=float)
        convergence = float(np.linalg.norm(residual(state)))
        shooting_success = bool(
            solution.success and convergence <= max(cfg.convergence_tolerance * 10.0, 1e-9)
        )
    converged = (
        shooting_success if cfg.use_periodic_shooting
        else convergence <= cfg.convergence_tolerance
    )
    phase = 0.0

    total_samples = cfg.output_cycles * cfg.samples_per_cycle
    states = np.zeros((total_samples, 4), dtype=float)
    bridge = np.zeros(total_samples, dtype=float)
    vout = np.zeros(total_samples, dtype=float)
    vp = np.zeros(total_samples, dtype=float)
    vs = np.zeros(total_samples, dtype=float)
    isec = np.zeros(total_samples, dtype=float)
    irect = np.zeros(total_samples, dtype=float)
    ico = np.zeros(total_samples, dtype=float)
    phase_values = np.zeros(total_samples, dtype=float)
    for index in range(total_samples):
        states[index] = state
        phase_values[index] = phase
        alg = _instantaneous(
            model, inputs, phase, state, cfg.rectifier_smoothing_current_a)
        bridge[index] = alg.bridge_voltage_v
        vout[index] = alg.output_voltage_v
        vp[index] = alg.primary_voltage_v
        vs[index] = alg.secondary_voltage_v
        isec[index] = alg.secondary_current_a
        irect[index] = alg.rectified_current_a
        ico[index] = alg.output_cap_current_a
        state = _rk4_step(
            model, inputs, phase, state, dt,
            cfg.rectifier_smoothing_current_a)
        phase += omega * dt

    time_s = np.arange(total_samples, dtype=float) * dt
    ir = states[:, 0]
    vcr = states[:, 1]
    im = states[:, 2]
    primary_load = ir - im
    vlr = bridge - vcr - vp - model.p.series_resistance_ohm * ir
    energy = 0.5 * model.p.lr_h * ir**2
    phase_local = phase_values - phase_values[0]
    q1, q2, q3, q4 = _gate_waveforms(
        phase_local, model.p.primary_deadtime_s * fs, model.p.primary_topology)
    v_leg_a, v_leg_b = _bridge_leg_waveforms(
        q1, q2, q3, q4, inputs.bus_voltage_v, model.p.primary_topology)
    vds_q1 = inputs.bus_voltage_v - v_leg_a
    vds_q2 = v_leg_a
    vds_q3 = inputs.bus_voltage_v - v_leg_b
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
    output_load_current = vout / model.p.load_resistance_ohm
    output_cap_internal_voltage = states[:, 3]

    optional_magnetic: dict[str, tuple[str, str, NDArray[np.float64], str, str, float]] = {}
    if model.p.primary_turns > 0 and model.p.transformer_core_area_m2 > 0.0:
        b_transformer = _integrate_periodic_zero_mean(
            vp, dt, model.p.primary_turns * model.p.transformer_core_area_m2)
        h_transformer = (
            model.p.primary_turns * im / model.p.transformer_magnetic_path_m
            if model.p.transformer_magnetic_path_m > 0.0 else np.zeros_like(im)
        )
        optional_magnetic.update({
            "b_transformer": (
                "变压器磁通密度 B", "T", b_transformer, "magnetics",
                "由详细 Vp 周期积分得到", fs),
            "h_transformer": (
                "变压器励磁场 H", "A/m", h_transformer, "magnetics",
                "按 Np*Im/le 的一维等效场", fs),
        })
    if (
        model.p.resonant_inductor_turns > 0
        and model.p.resonant_inductor_core_area_m2 > 0.0
    ):
        b_lr = model.p.lr_h * ir / (
            model.p.resonant_inductor_turns * model.p.resonant_inductor_core_area_m2)
        h_lr = (
            model.p.resonant_inductor_turns * ir
            / model.p.resonant_inductor_magnetic_path_m
            if model.p.resonant_inductor_magnetic_path_m > 0.0 else np.zeros_like(ir)
        )
        optional_magnetic.update({
            "b_resonant_inductor": (
                "谐振电感磁通密度 B", "T", b_lr, "magnetics",
                "按 Lr*Ir/(N*Ae) 计算", fs),
            "h_resonant_inductor": (
                "谐振电感等效磁场 H", "A/m", h_lr, "magnetics",
                "按 N*Ir/le 的一维等效场", fs),
        })

    raw: dict[str, tuple[str, str, NDArray[np.float64], str, str, float]] = {
        "v_leg_a": ("A 桥臂中点电压 VA", "V", v_leg_a, "primary", "相对母线负端", fs),
        "v_leg_b": ("B 桥臂中点电压 VB", "V", v_leg_b, "primary", "全桥 B 桥臂；半桥时为母线中点", fs),
        "v_bridge": ("桥臂差模输出电压 Vab", "V", bridge, "primary", "含死区的理想桥臂电压", fs),
        "i_resonant": ("谐振电流 Ir", "A", ir, "primary", "分段非线性模型的谐振电流", fs),
        "v_resonant_cap": ("谐振电容电压 VCr", "V", vcr, "primary", "分段积分得到的谐振电容电压", fs),
        "v_resonant_inductor": ("谐振电感电压 VLr", "V", vlr, "primary", "Lr 端电压", fs),
        "v_transformer_primary": ("变压器原边电压 Vp", "V", vp, "transformer", "SR 整流钳位原边电压", fs),
        "i_transformer_primary": ("变压器原边电流 Ip", "A", ir, "transformer", "变压器输入端总电流", fs),
        "i_magnetizing": ("励磁电流 Im", "A", im, "transformer", "励磁电感电流", fs),
        "i_primary_load": ("原边负载分量 Iload,p", "A", primary_load, "transformer", "Ir-Im", fs),
        "v_transformer_secondary": ("变压器次级电压 Vs", "V", vs, "secondary", "未整流次级电压", fs),
        "i_transformer_secondary": ("变压器次级电流 Is", "A", isec, "secondary", "未整流次级电流", fs),
        "v_rectified": ("SR 整流后电压 Vrect", "V", np.abs(vs), "secondary", "全桥 SR 理想整流电压", 2.0*fs),
        "i_rectified": ("SR 整流输出电流 Irect", "A", irect, "secondary", "全桥 SR 整流电流", 2.0*fs),
        "i_load_output": ("输出负载电流 Io", "A", output_load_current, "output", "端口电压除以等效负载", 2.0*fs),
        "i_output_cap": ("输出电容电流 ICo", "A", ico, "output", "整流电流减去负载电流", 2.0*fs),
        "v_output_cap_internal": ("输出电容内部电压 VCo", "V", output_cap_internal_voltage, "output", "不含 ESR 瞬时压降", 2.0*fs),
        "v_output": ("输出端电压 Vo", "V", vout, "output", "包含 Co ESR 的端口电压", 2.0*fs),
        "v_output_ripple": ("输出纹波 ΔVo", "V", vout - np.mean(vout), "output", "去直流输出纹波", 2.0*fs),
        "energy_lr": ("谐振电感储能 ELr", "J", energy, "magnetics", "0.5*Lr*Ir^2", 2.0*fs),
        "gate_q1": ("Q1 门极逻辑", "pu", q1, "switching", "理想化门极逻辑", fs),
        "gate_q2": ("Q2 门极逻辑", "pu", q2, "switching", "理想化门极逻辑", fs),
        "gate_q3": ("Q3 门极逻辑", "pu", q3, "switching", "理想化门极逻辑", fs),
        "gate_q4": ("Q4 门极逻辑", "pu", q4, "switching", "理想化门极逻辑", fs),
        "vds_q1": ("Q1 VDS", "V", vds_q1, "switching", "理想 VDS 状态，不含非线性 Coss", fs),
        "vds_q2": ("Q2 VDS", "V", vds_q2, "switching", "理想 VDS 状态，不含非线性 Coss", fs),
        "vds_q3": ("Q3 VDS", "V", vds_q3, "switching", "全桥 B 臂上管 VDS", fs),
        "vds_q4": ("Q4 VDS", "V", vds_q4, "switching", "全桥 B 臂下管 VDS", fs),
        "ids_q1": ("Q1 支路电流", "A", i_q1, "switching", "理想导通窗口有符号电流", fs),
        "ids_q2": ("Q2 支路电流", "A", i_q2, "switching", "理想导通窗口有符号电流", fs),
        "ids_q3": ("Q3 支路电流", "A", i_q3, "switching", "理想导通窗口有符号电流", fs),
        "ids_q4": ("Q4 支路电流", "A", i_q4, "switching", "理想导通窗口有符号电流", fs),
        "gate_sr1": ("SR1 门极逻辑", "pu", gate_sr1, "sr_switching", "理想电流极性驱动", fs),
        "gate_sr2": ("SR2 门极逻辑", "pu", gate_sr2, "sr_switching", "理想电流极性驱动", fs),
        "gate_sr3": ("SR3 门极逻辑", "pu", gate_sr3, "sr_switching", "理想电流极性驱动", fs),
        "gate_sr4": ("SR4 门极逻辑", "pu", gate_sr4, "sr_switching", "理想电流极性驱动", fs),
        "ids_sr1": ("SR1 电流", "A", i_sr1, "sr_switching", "正向对角器件电流", fs),
        "ids_sr2": ("SR2 电流", "A", i_sr2, "sr_switching", "反向对角器件电流", fs),
        "ids_sr3": ("SR3 电流", "A", i_sr3, "sr_switching", "反向对角器件电流", fs),
        "ids_sr4": ("SR4 电流", "A", i_sr4, "sr_switching", "正向对角器件电流", fs),
        "vds_sr1": ("SR1 VDS", "V", vds_sr1, "sr_switching", "理想 SR 阻断电压", fs),
        "vds_sr2": ("SR2 VDS", "V", vds_sr2, "sr_switching", "理想 SR 阻断电压", fs),
        "vds_sr3": ("SR3 VDS", "V", vds_sr3, "sr_switching", "理想 SR 阻断电压", fs),
        "vds_sr4": ("SR4 VDS", "V", vds_sr4, "sr_switching", "理想 SR 阻断电压", fs),
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
                samples_per_period=cfg.samples_per_cycle,
                switching_frequency_hz=fs,
                fundamental_frequency_hz=fundamental_hz,
            ),
            group=group,
            description=description,
        )

    warnings = [
        "Switched model uses an ideal transformer and ideal full-bridge SR clamp.",
        "MOSFET nonlinear Coss/Qoss, leakage inductance and winding capacitance are not yet included.",
    ]
    if not converged:
        warnings.append(
            f"Periodic settling did not meet tolerance after {settled_cycles} cycles; "
            f"normalized mismatch={convergence:.3e}."
        )
    return WaveformBundle(
        time_s=time_s,
        switching_frequency_hz=fs,
        model_name="piecewise_switched_level2",
        signals=signals,
        warnings=tuple(warnings),
        metadata={
            "settling_cycles": settled_cycles,
            "periodic_mismatch": convergence,
            "converged": str(converged),
            "periodic_shooting": str(cfg.use_periodic_shooting),
            "shooting_evaluations": shooting_evaluations,
            "samples_per_cycle": cfg.samples_per_cycle,
            "output_cycles": cfg.output_cycles,
        },
    )
