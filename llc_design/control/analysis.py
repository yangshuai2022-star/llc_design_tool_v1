"""High-level LLC small-signal analysis workflow."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import json
import math
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..core.operating_point import LLCOperatingPoint, solve_operating_point
from ..core.spec import LLCDesignSpec
from ..core.tank import TankDesign, design_tank
from ..core.tank import bridge_fundamental_rms_v, tank_state
from ..dynamics.plant import (
    DynamicPhasorModel,
    DynamicPhasorSteadyState,
    LLCPlantInputs,
    LLCPlantParameters,
)
from ..models.system import LLCSystemAnalyzer, SystemAnalysis
from .discretize import DiscretePlant, discretize_zoh
from .export_c import export_c99_plant_model, export_c99_state_space_model
from .linearize import (
    ControlInputKind,
    LinearizedPlant,
    SISOTransferFunction,
    linearize_dynamic_phasor,
)


@dataclass(frozen=True)
class SmallSignalAnalysis:
    spec: LLCDesignSpec
    tank: TankDesign
    operating_point: LLCOperatingPoint
    parameters: LLCPlantParameters
    steady_state: DynamicPhasorSteadyState
    continuous_plant: LinearizedPlant
    continuous_transfer: SISOTransferFunction
    discrete_plant: DiscretePlant
    sample_time_s: float
    control_input_kind: ControlInputKind
    timer_clock_hz: float | None
    line_to_output_transfer: SISOTransferFunction
    output_impedance_transfer: SISOTransferFunction
    resonant_current_transfer: SISOTransferFunction
    magnetizing_current_transfer: SISOTransferFunction

    @property
    def stable(self) -> bool:
        return self.continuous_plant.stable and self.discrete_plant.stable


def _nearest_operating_point(
    analysis: SystemAnalysis,
    vbus_v: float,
    load_fraction: float,
) -> LLCOperatingPoint | None:
    exact = [
        item.operating_point for item in analysis.operating_points
        if abs(item.operating_point.vbus_v - vbus_v) < 1e-9
        and abs(item.operating_point.load_fraction - load_fraction) < 1e-9
    ]
    return exact[0] if exact else None


def build_small_signal_analysis(
    spec: LLCDesignSpec,
    *,
    vbus_v: float | None = None,
    load_fraction: float = 1.0,
    sample_time_s: float = 20e-6,
    control_input_kind: ControlInputKind = ControlInputKind.FREQUENCY_HZ,
    timer_clock_hz: float | None = None,
    input_delay_samples: int = 0,
    system_analysis: SystemAnalysis | None = None,
    series_resistance_ohm: float | None = None,
    trim_frequency_to_output: bool = True,
) -> SmallSignalAnalysis:
    """Solve, linearize and ZOH-discretize the LLC plant at one work point."""
    spec.validate()
    if not (0.0 < load_fraction <= 1.5):
        raise ValueError("load fraction must be within 0..1.5")
    tank = design_tank(spec)
    bus = spec.vbus_nom_v if vbus_v is None else float(vbus_v)
    analysis = system_analysis
    operating_point = (
        _nearest_operating_point(analysis, bus, load_fraction)
        if analysis is not None else None
    )
    if operating_point is None:
        operating_point = solve_operating_point(spec, tank, bus, load_fraction)
    params = LLCPlantParameters.from_design(
        spec, tank, operating_point, analysis,
        series_resistance_ohm=series_resistance_ohm)
    model = DynamicPhasorModel(params)
    inputs = LLCPlantInputs(
        operating_point.switching_frequency_hz,
        operating_point.vbus_v,
        0.0,
    )
    if trim_frequency_to_output:
        steady = model.solve_regulated_steady_state(
            bus_voltage_v=operating_point.vbus_v,
            target_output_voltage_v=spec.vout_v,
            frequency_guess_hz=operating_point.switching_frequency_hz,
            minimum_frequency_hz=spec.minimum_frequency_hz,
            maximum_frequency_hz=spec.maximum_frequency_hz,
            operating_point=operating_point,
        )
        state = tank_state(tank, steady.inputs.switching_frequency_hz, operating_point.rac_ohm)
        operating_point = replace(
            operating_point,
            switching_frequency_hz=steady.inputs.switching_frequency_hz,
            normalized_frequency=steady.inputs.switching_frequency_hz / tank.fr_hz,
            achieved_gain=operating_point.required_gain,
            branch=f"{operating_point.branch}+EDF_TRIM",
            input_impedance_ohm=state.z_input_ohm,
            input_phase_deg=state.input_phase_deg,
            bridge_fundamental_rms_v=bridge_fundamental_rms_v(
                spec, operating_point.vbus_v),
            resonant_current_rms_a=steady.resonant_current_rms_a,
            resonant_current_peak_a=steady.resonant_current_peak_a,
            magnetizing_current_rms_a=steady.magnetizing_current_rms_a,
            magnetizing_current_peak_a=steady.magnetizing_current_peak_a,
            reflected_load_current_rms_a=(
                steady.primary_load_current_peak_a / math.sqrt(2.0)),
            secondary_current_rms_a=steady.secondary_current_rms_a,
            secondary_current_peak_a=steady.secondary_current_rms_a * math.sqrt(2.0),
        )
    else:
        steady = model.solve_steady_state(inputs, operating_point=operating_point)
    continuous = linearize_dynamic_phasor(model, steady)
    continuous = continuous.with_control_input(
        control_input_kind, timer_clock_hz=timer_clock_hz)
    transfer = continuous.siso(output_name="output_voltage_v")
    line_transfer = continuous.siso(
        input_name="bus_voltage_v", output_name="output_voltage_v")
    output_impedance = continuous.siso(
        input_name="load_current_disturbance_a", output_name="output_voltage_v"
    ).scaled(-1.0, input_name="load_current_a", input_unit="A")
    resonant_current_transfer = continuous.siso(
        output_name="resonant_current_rms_a")
    magnetizing_current_transfer = continuous.siso(
        output_name="magnetizing_current_rms_a")
    discrete = discretize_zoh(
        continuous,
        sample_time_s,
        output_name="output_voltage_v",
        input_delay_samples=input_delay_samples,
    )
    return SmallSignalAnalysis(
        spec=spec,
        tank=tank,
        operating_point=operating_point,
        parameters=params,
        steady_state=steady,
        continuous_plant=continuous,
        continuous_transfer=transfer,
        discrete_plant=discrete,
        sample_time_s=sample_time_s,
        control_input_kind=control_input_kind,
        timer_clock_hz=timer_clock_hz,
        line_to_output_transfer=line_transfer,
        output_impedance_transfer=output_impedance,
        resonant_current_transfer=resonant_current_transfer,
        magnetizing_current_transfer=magnetizing_current_transfer,
    )


def _complex_rows(values: np.ndarray, kind: str) -> list[dict[str, float | str]]:
    rows = []
    for index, value in enumerate(values):
        rows.append({
            "kind": kind,
            "index": index,
            "real": float(np.real(value)),
            "imag": float(np.imag(value)),
            "magnitude": float(abs(value)),
            "frequency_hz": float(abs(np.imag(value)) / (2.0 * math.pi)),
        })
    return rows


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return [{"real": float(v.real), "imag": float(v.imag)} for v in value]
        return value.tolist()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, ControlInputKind):
        return value.value
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _format_polynomial(coefficients: np.ndarray, variable: str) -> str:
    degree = len(coefficients) - 1
    terms: list[str] = []
    for index, coefficient in enumerate(coefficients):
        power = degree - index
        if abs(coefficient) < 1e-16:
            continue
        magnitude = abs(float(coefficient))
        if power == 0:
            core = f"{magnitude:.8e}"
        elif power == 1:
            core = f"{magnitude:.8e}{variable}"
        else:
            core = f"{magnitude:.8e}{variable}^{power}"
        if not terms:
            terms.append(("-" if coefficient < 0 else "") + core)
        else:
            terms.append((" - " if coefficient < 0 else " + ") + core)
    return "".join(terms) or "0"


def write_small_signal_markdown(result: SmallSignalAnalysis, path: str | Path) -> Path:
    output = Path(path)
    tf = result.continuous_transfer
    zd = result.discrete_plant
    op = result.operating_point
    steady = result.steady_state
    lines = [
        "# LLC 功率级小信号与数字对象计算书",
        "",
        "> 版权说明：工具设计人 **杨帅锅** · 开关电源仿真与实用设计",
        "",
        "> 模型：七状态动态相量/扩展描述函数数值模型。谐振状态保留正交基波分量，整流器使用正弦负载电流的全波描述函数；模型在选定工作点求稳态后进行 Jacobian 线性化。",
        "",
        "## 1. 工作点",
        "",
        "| 项目 | 数值 |",
        "|---|---:|",
        f"| 母线电压 | {op.vbus_v:.6g} V |",
        f"| 负载 | {op.load_fraction*100:.3f}% / {op.pout_w:.3f} W |",
        f"| 开关频率 | {op.switching_frequency_hz/1e3:.6f} kHz |",
        f"| Lr / Cr / Lm | {result.tank.lr_h*1e6:.6f} µH / {result.tank.cr_f*1e9:.6f} nF / {result.tank.lm_h*1e6:.6f} µH |",
        f"| 等效串联阻尼 | {result.parameters.series_resistance_ohm:.6f} Ω |",
        f"| EDF 稳态输出 | {steady.output_voltage_v:.6f} V |",
        f"| 输出误差 | {steady.output_voltage_error_v:.9f} V |",
        f"| 频率二次校准 | {'是' if steady.frequency_trimmed else '否'} |",
        f"| EDF Ir RMS / Im RMS | {steady.resonant_current_rms_a:.6f} / {steady.magnetizing_current_rms_a:.6f} A |",
        "",
        "## 2. 连续域状态空间",
        "",
        f"- 状态：`{', '.join(result.continuous_plant.state_names)}`",
        f"- 输入：`{', '.join(result.continuous_plant.input_names)}`",
        f"- 输出：`{', '.join(result.continuous_plant.output_names)}`",
        f"- 连续极点稳定：**{'是' if result.continuous_plant.stable else '否'}**",
        "",
        "矩阵保存在 `continuous_state_space.json`。",
        "",
        "## 3. 频率/周期/计数到输出电压传递函数",
        "",
        f"控制输入：`{tf.input_name}` [{tf.input_unit}]",
        "",
        "```text",
        f"G(s) = ({_format_polynomial(tf.numerator, 's')})",
        f"       / ({_format_polynomial(tf.denominator, 's')})",
        "```",
        "",
        f"- 直流增益：{tf.dc_gain:.12g} {tf.output_unit}/{tf.input_unit}",
        f"- 阶数：{tf.order}",
        f"- 母线到输出 Gvg(0)：{result.line_to_output_transfer.dc_gain:.12g} V/V",
        f"- 输出阻抗 Zout(0)：{result.output_impedance_transfer.dc_gain:.12g} Ω",
        f"- 频率到谐振电流 Girf(0)：{result.resonant_current_transfer.dc_gain:.12g} A/{tf.input_unit}",
        f"- 频率到励磁电流 Gimf(0)：{result.magnetizing_current_transfer.dc_gain:.12g} A/{tf.input_unit}",
        "",
        "## 4. ZOH 离散对象",
        "",
        f"- 控制采样周期：{zd.sample_time_s*1e6:.6f} µs",
        f"- 采样频率：{1.0/zd.sample_time_s/1e3:.6f} kHz",
        f"- 计算/PWM 更新延迟：{zd.input_delay_samples} 个采样",
        f"- 离散极点稳定：**{'是' if zd.stable else '否'}**",
        "",
        "```text",
        "G(z) = (" + " + ".join(f"{b:.10e} z^-{i}" for i, b in enumerate(zd.numerator)) + ")",
        "       / (" + " + ".join(f"{a:.10e} z^-{i}" for i, a in enumerate(zd.denominator)) + ")",
        "```",
        "",
        "```text",
        zd.difference_equation.text(precision=12),
        "```",
        "",
        "## 5. 输出文件",
        "",
        "- `continuous_state_space.json`",
        "- `discrete_state_space.json`",
        "- `continuous_poles_zeros.csv`",
        "- `discrete_poles_zeros.csv`",
        "- `bode_continuous_discrete.png`",
        "- `pole_zero_map.png`",
        "- `llc_plant_model.h/.c`",
        "",
        "## 6. 精度边界",
        "",
        "- 该对象不是静态 FHA 增益斜率拼接模型；谐振腔正交状态和输出电容动态均参与 Jacobian。",
        "- 默认先用 EDF 外层频率求解把输出电压校准到设定值，再在该受控平衡点线性化。",
        "- 整流器假定负载电流以基波为主，未包含次级漏感、换流振铃及非线性 Coss。",
        "- 接近轻载 Burst、跳频、打嗝和模式切换时，单工作点 LTI 模型不再成立。",
        "- 逐周期同步采样的最终控制验证应继续使用 sampled-data/Poincaré 模型或频率注入辨识。",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _plot_bode(result: SmallSignalAnalysis, path: Path) -> Path:
    fsw = result.operating_point.switching_frequency_hz
    nyquist = 0.5 / result.sample_time_s
    fmax = max(100.0, min(0.45 * fsw, 0.90 * nyquist))
    fmin = max(0.5, fmax / 1e5)
    frequencies = np.geomspace(fmin, fmax, 1200)
    hc = result.continuous_transfer.frequency_response(frequencies)
    hd = result.discrete_plant.frequency_response(frequencies)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].semilogx(frequencies, 20.0 * np.log10(np.maximum(np.abs(hc), 1e-18)), label="continuous EDF")
    axes[0].semilogx(frequencies, 20.0 * np.log10(np.maximum(np.abs(hd), 1e-18)), linestyle="--", label="ZOH discrete")
    axes[0].set_ylabel("Magnitude (dB)")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()
    axes[1].semilogx(frequencies, np.unwrap(np.angle(hc)) * 180.0 / math.pi, label="continuous EDF")
    axes[1].semilogx(frequencies, np.unwrap(np.angle(hd)) * 180.0 / math.pi, linestyle="--", label="ZOH discrete")
    axes[1].set_xlabel("Perturbation frequency (Hz)")
    axes[1].set_ylabel("Phase (deg)")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()
    fig.suptitle(
        f"LLC control-to-output plant @ {result.operating_point.vbus_v:.0f} V, "
        f"{result.operating_point.load_fraction*100:.0f}% load")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_poles_zeros(result: SmallSignalAnalysis, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    cp = result.continuous_transfer.poles
    cz = result.continuous_transfer.zeros
    axes[0].scatter(cp.real, cp.imag, marker="x", label="poles")
    if len(cz):
        axes[0].scatter(cz.real, cz.imag, facecolors="none", edgecolors="C1", label="zeros")
    axes[0].axvline(0.0, linewidth=0.8)
    axes[0].set_title("Continuous s-plane")
    axes[0].set_xlabel("Real (rad/s)")
    axes[0].set_ylabel("Imaginary (rad/s)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    theta = np.linspace(0.0, 2.0 * math.pi, 500)
    axes[1].plot(np.cos(theta), np.sin(theta), linestyle=":")
    dp = result.discrete_plant.poles
    dz = result.discrete_plant.zeros
    axes[1].scatter(dp.real, dp.imag, marker="x", label="poles")
    if len(dz):
        axes[1].scatter(dz.real, dz.imag, facecolors="none", edgecolors="C1", label="zeros")
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_title("Discrete z-plane")
    axes[1].set_xlabel("Real")
    axes[1].set_ylabel("Imaginary")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def export_small_signal_analysis(
    result: SmallSignalAnalysis,
    directory: str | Path,
) -> dict[str, Path]:
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    paths["report"] = write_small_signal_markdown(result, output / "LLC_small_signal_model.md")

    continuous_json = {
        "model_name": result.continuous_plant.model_name,
        "state_names": result.continuous_plant.state_names,
        "input_names": result.continuous_plant.input_names,
        "input_units": result.continuous_plant.input_units,
        "output_names": result.continuous_plant.output_names,
        "output_units": result.continuous_plant.output_units,
        "A": result.continuous_plant.a,
        "B": result.continuous_plant.b,
        "C": result.continuous_plant.c,
        "D": result.continuous_plant.d,
        "steady_states": result.continuous_plant.steady_states,
        "steady_inputs": result.continuous_plant.steady_inputs,
        "steady_outputs": result.continuous_plant.steady_outputs,
        "poles": result.continuous_plant.poles,
        "transfer_numerator": result.continuous_transfer.numerator,
        "transfer_denominator": result.continuous_transfer.denominator,
        "dc_gain": result.continuous_transfer.dc_gain,
        "transfers": {
            "Gvf": {
                "numerator": result.continuous_transfer.numerator,
                "denominator": result.continuous_transfer.denominator,
                "dc_gain": result.continuous_transfer.dc_gain,
            },
            "Gvg": {
                "numerator": result.line_to_output_transfer.numerator,
                "denominator": result.line_to_output_transfer.denominator,
                "dc_gain": result.line_to_output_transfer.dc_gain,
            },
            "Zout": {
                "numerator": result.output_impedance_transfer.numerator,
                "denominator": result.output_impedance_transfer.denominator,
                "dc_gain": result.output_impedance_transfer.dc_gain,
            },
            "Girf": {
                "numerator": result.resonant_current_transfer.numerator,
                "denominator": result.resonant_current_transfer.denominator,
                "dc_gain": result.resonant_current_transfer.dc_gain,
            },
            "Gimf": {
                "numerator": result.magnetizing_current_transfer.numerator,
                "denominator": result.magnetizing_current_transfer.denominator,
                "dc_gain": result.magnetizing_current_transfer.dc_gain,
            },
        },
    }
    paths["continuous_json"] = output / "continuous_state_space.json"
    paths["continuous_json"].write_text(
        json.dumps(_jsonable(continuous_json), ensure_ascii=False, indent=2), encoding="utf-8")

    discrete_json = {
        "sample_time_s": result.discrete_plant.sample_time_s,
        "input_delay_samples": result.discrete_plant.input_delay_samples,
        "Ad": result.discrete_plant.ad,
        "Bd": result.discrete_plant.bd,
        "Cd": result.discrete_plant.cd,
        "Dd": result.discrete_plant.dd,
        "numerator_z_inverse": result.discrete_plant.numerator,
        "denominator_z_inverse": result.discrete_plant.denominator,
        "poles": result.discrete_plant.poles,
        "zeros": result.discrete_plant.zeros,
        "difference_equation": result.discrete_plant.difference_equation.text(precision=16),
    }
    paths["discrete_json"] = output / "discrete_state_space.json"
    paths["discrete_json"].write_text(
        json.dumps(_jsonable(discrete_json), ensure_ascii=False, indent=2), encoding="utf-8")

    continuous_rows = _complex_rows(result.continuous_transfer.poles, "pole")
    continuous_rows.extend(_complex_rows(result.continuous_transfer.zeros, "zero"))
    paths["continuous_pz"] = output / "continuous_poles_zeros.csv"
    pd.DataFrame(continuous_rows).to_csv(paths["continuous_pz"], index=False)
    discrete_rows = _complex_rows(result.discrete_plant.poles, "pole")
    discrete_rows.extend(_complex_rows(result.discrete_plant.zeros, "zero"))
    paths["discrete_pz"] = output / "discrete_poles_zeros.csv"
    pd.DataFrame(discrete_rows).to_csv(paths["discrete_pz"], index=False)

    paths["bode"] = _plot_bode(result, output / "bode_continuous_discrete.png")
    paths["pole_zero"] = _plot_poles_zeros(result, output / "pole_zero_map.png")
    paths.update({f"c99_direct_{key}": value for key, value in export_c99_plant_model(
        result.discrete_plant, output, basename="llc_plant_model").items()})
    paths.update({f"c99_state_{key}": value for key, value in export_c99_state_space_model(
        result.discrete_plant, output, basename="llc_plant_state_space").items()})
    return paths
