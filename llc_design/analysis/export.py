"""Export V8 multi-fidelity LLC analysis results."""

from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from .types import FidelityLevel, MultiFidelityAnalysis
from ..dynamics.export import export_waveform_bundle


def _jsonable(value: Any) -> Any:
    if isinstance(value, FidelityLevel):
        return value.value
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return [
                {"real": float(item.real), "imag": float(item.imag)}
                for item in value
            ]
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _comparison_header() -> tuple[str, ...]:
    return (
        "model",
        "reference",
        "converged",
        "residual_norm",
        "switching_frequency_hz",
        "output_voltage_v",
        "output_power_w",
        "normalized_gain",
        "input_phase_deg",
        "resonant_current_rms_a",
        "resonant_current_peak_a",
        "magnetizing_current_rms_a",
        "secondary_current_rms_a",
        "resonant_capacitor_peak_v",
        "power_balance_error_percent",
        "frequency_error_percent",
        "gain_error_percent",
        "resonant_rms_error_percent",
        "resonant_peak_error_percent",
        "magnetizing_rms_error_percent",
        "secondary_rms_error_percent",
        "resonant_cap_peak_error_percent",
    )


def _comparison_record(
    analysis: MultiFidelityAnalysis,
    row,
) -> dict[str, Any]:
    metrics = row.metrics
    return {
        "model": row.fidelity.value,
        "reference": row.fidelity is analysis.reference_level,
        "converged": row.converged,
        "residual_norm": row.residual_norm,
        "switching_frequency_hz": metrics.switching_frequency_hz,
        "output_voltage_v": metrics.output_voltage_v,
        "output_power_w": metrics.output_power_w,
        "normalized_gain": metrics.normalized_gain,
        "input_phase_deg": metrics.input_phase_deg,
        "resonant_current_rms_a": metrics.resonant_current_rms_a,
        "resonant_current_peak_a": metrics.resonant_current_peak_a,
        "magnetizing_current_rms_a": metrics.magnetizing_current_rms_a,
        "secondary_current_rms_a": metrics.secondary_current_rms_a,
        "resonant_capacitor_peak_v": metrics.resonant_capacitor_peak_v,
        "power_balance_error_percent": metrics.power_balance_error_percent,
        "frequency_error_percent": row.frequency_error_percent,
        "gain_error_percent": row.gain_error_percent,
        "resonant_rms_error_percent": row.resonant_rms_error_percent,
        "resonant_peak_error_percent": row.resonant_peak_error_percent,
        "magnetizing_rms_error_percent": row.magnetizing_rms_error_percent,
        "secondary_rms_error_percent": row.secondary_rms_error_percent,
        "resonant_cap_peak_error_percent": row.resonant_cap_peak_error_percent,
    }


def write_multifidelity_markdown(
    analysis: MultiFidelityAnalysis,
    path: str | Path,
) -> Path:
    output = Path(path)
    request = analysis.request
    lines = [
        "# LLC V8 多保真模型对比计算书",
        "",
        "> 模型层级：FHA 快速设计、多谐波谐波平衡、理想开关分段时域。",
        "",
        "## 工作点",
        "",
        f"- 母线电压：{request.bus_voltage_v:.6g} V",
        f"- 目标输出：{request.output_voltage_target_v:.6g} V",
        f"- 负载：{request.load_fraction*100:.3f}% / {request.requested_output_power_w:.6g} W",
        f"- 比较参考：`{analysis.reference_level.value}`",
        "",
        "## 模型结果",
        "",
        "| 模型 | 收敛 | Fsw/kHz | Vo/V | Gain | Ir RMS/A | Ir PK/A | Im RMS/A | Is RMS/A | VCr PK/V | ΔIr RMS/% | ΔVCr PK/% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in analysis.comparison_rows:
        metrics = row.metrics
        marker = " **REF**" if row.fidelity is analysis.reference_level else ""
        lines.append(
            f"| {row.fidelity.value}{marker} | {'PASS' if row.converged else 'FAIL'} | "
            f"{metrics.switching_frequency_hz/1e3:.6f} | {metrics.output_voltage_v:.6f} | "
            f"{metrics.normalized_gain:.8f} | {metrics.resonant_current_rms_a:.6f} | "
            f"{metrics.resonant_current_peak_a:.6f} | {metrics.magnetizing_current_rms_a:.6f} | "
            f"{metrics.secondary_current_rms_a:.6f} | {metrics.resonant_capacitor_peak_v:.6f} | "
            f"{row.resonant_rms_error_percent:.4f} | {row.resonant_cap_peak_error_percent:.4f} |"
        )

    lines.extend(["", "## 收敛与模型边界", ""])
    for level, result in analysis.results.items():
        lines.append(
            f"- `{level.value}`：converged={result.convergence.converged}，"
            f"residual={result.convergence.residual_norm:.6e}，"
            f"method=`{result.convergence.method}`"
        )
        if result.harmonic_orders:
            lines.append(
                "  - 谐波阶次：" + ", ".join(str(item) for item in result.harmonic_orders)
            )
        for warning in result.warnings:
            lines.append(f"  - {warning}")
    for warning in analysis.warnings:
        lines.append(f"- 系统提示：{warning}")

    lines.extend(["", "## 文件结构", ""])
    for level in analysis.results:
        lines.append(f"- `{level.value}/`：该模型的同步波形 CSV、统计、图表和单模型报告。")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def export_multifidelity_analysis(
    analysis: MultiFidelityAnalysis,
    directory: str | Path,
    *,
    export_model_waveforms: bool = True,
) -> dict[str, Path]:
    """Export comparison CSV/JSON/Markdown and optional per-model waveforms."""

    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    records = [_comparison_record(analysis, row) for row in analysis.comparison_rows]
    comparison_csv = output / "model_comparison.csv"
    with comparison_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_comparison_header())
        writer.writeheader()
        writer.writerows(records)
    paths["comparison_csv"] = comparison_csv

    payload = {
        "request": {
            "vbus_v": analysis.request.bus_voltage_v,
            "target_output_voltage_v": analysis.request.output_voltage_target_v,
            "load_fraction": analysis.request.load_fraction,
            "requested_output_power_w": analysis.request.requested_output_power_w,
            "frequency_hz": analysis.request.frequency_hz,
            "regulate_output": analysis.request.regulate_output,
        },
        "reference_level": analysis.reference_level.value,
        "warnings": list(analysis.warnings),
        "comparison": records,
        "models": {
            level.value: {
                "metrics": result.metrics.as_dict(),
                "convergence": asdict(result.convergence),
                "harmonic_orders": list(result.harmonic_orders),
                "warnings": list(result.warnings),
                "diagnostics": _jsonable(dict(result.diagnostics)),
            }
            for level, result in analysis.results.items()
        },
    }
    summary_json = output / "model_comparison.json"
    summary_json.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths["summary_json"] = summary_json
    paths["report"] = write_multifidelity_markdown(
        analysis, output / "LLC_V8_model_comparison.md")

    if export_model_waveforms:
        for level, result in analysis.results.items():
            model_paths = export_waveform_bundle(
                result.waveform, output / level.value)
            for key, path in model_paths.items():
                paths[f"{level.value}_{key}"] = path
    return paths
