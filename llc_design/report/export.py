"""Markdown/CSV/JSON calculation-book export."""

from __future__ import annotations

import math
from dataclasses import asdict
from enum import Enum
import json
from pathlib import Path

import pandas as pd

from ..models.system import SystemAnalysis
from ..magnetics.core import CoreDatabase
from ..plotting.plots import create_all_plots


def _jsonable(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, float) and not math.isfinite(value):
        # Keep output.json strict-JSON compatible; non-finite values encode as null.
        return None
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def operating_point_rows(analysis: SystemAnalysis) -> list[dict]:
    rows = []
    for point in analysis.operating_points:
        op = point.operating_point
        row = {
            "label": point.label,
            "vbus_v": op.vbus_v,
            "load_pct": op.load_fraction * 100.0,
            "pout_w": op.pout_w,
            "frequency_khz": op.switching_frequency_hz / 1e3,
            "normalized_frequency": op.normalized_frequency,
            "required_gain": op.required_gain,
            "phase_deg": op.input_phase_deg,
            "q_effective": op.q_effective,
            "resonant_current_rms_a": op.resonant_current_rms_a,
            "resonant_current_peak_a": op.resonant_current_peak_a,
            "magnetizing_current_peak_a": op.magnetizing_current_peak_a,
            "secondary_current_rms_a": op.secondary_current_rms_a,
            "commutation_current_a": op.commutation_current_a,
            "zvs_charge_margin": point.primary.zvs_charge_margin,
            "zvs_energy_margin": point.primary.zvs_energy_margin,
            "output_ripple_vpp": point.output_capacitor.total_ripple_vpp,
            "total_loss_w": point.total_loss_w,
            "efficiency_pct": point.efficiency * 100.0,
        }
        row.update(point.breakdown())
        row.update(point.detailed_magnetics_breakdown())
        rows.append(row)
    return rows


def write_markdown_report(analysis: SystemAnalysis, path: str | Path) -> Path:
    output = Path(path)
    nominal = analysis.nominal
    spec, tank = analysis.spec, analysis.tank
    t = analysis.transformer
    ind = analysis.resonant_inductor
    bus = analysis.bus_capacitor
    lines = [
        "# LLC 全桥 + 全桥同步整流 V2 磁性损耗计算书",
        "",
        "> 版权说明：工具设计人 **杨帅锅** · 开关电源仿真与实用设计",
        "",
        "> **磁性模型等级：高级工程模型。** 磁芯使用温度插值 iGSE 对实际重构 B(t) 积分；Litz 铜损按逐谐波、圆股精确集肤、层间 MMF、束内换位、端接和气隙边缘场分解。磁材系数和标准磁芯几何仍需在硬件定版前用准确厂家曲线与样件校准。",
        "",
        "## 1. 设计输入",
        "",
        "| 项目 | 数值 |",
        "|---|---:|",
        f"| 母线额定/范围 | {spec.vbus_nom_v:.0f} V / {spec.vbus_hold_end_v:.0f}–{spec.vbus_max_v:.0f} V |",
        f"| 输出 | {spec.vout_v:.1f} V / {spec.pout_w/1000:.2f} kW / {spec.output_current_a:.2f} A |",
        f"| 一次侧 | {spec.primary_topology.value} |",
        f"| 二次侧 | {spec.secondary_topology.value} |",
        f"| 频率 | {spec.minimum_frequency_hz/1e3:.1f}–{spec.maximum_frequency_hz/1e3:.1f} kHz，fr={spec.resonant_frequency_hz/1e3:.1f} kHz |",
        f"| 匝数/匝比 | {spec.primary_turns}:{spec.secondary_turns} / {spec.turns_ratio:.4f} |",
        f"| Litz 单丝裸铜直径 | {spec.litz_strand_copper_diameter_m*1e3:.3f} mm |",
        f"| 母线电容/维持时间 | {spec.bus_capacitance_f*1e6:.0f} µF / {spec.requested_hold_time_s*1e3:.1f} ms |",
        "",
        "## 2. 谐振腔结果",
        "",
        "| 参数 | 结果 |",
        "|---|---:|",
        f"| Ln = Lm/Lr | {tank.ln_ratio:.4f} |",
        f"| 满载 Qe | {tank.q_full_load:.4f} |",
        f"| Rac（额定满载、一次侧折算） | {tank.rac_nom_ohm:.3f} Ω |",
        f"| Zr | {tank.zr_ohm:.3f} Ω |",
        f"| Lr | {tank.lr_h*1e6:.3f} µH |",
        f"| Cr | {tank.cr_f*1e9:.3f} nF |",
        f"| Lm | {tank.lm_h*1e6:.3f} µH |",
        f"| fm | {tank.fm_hz/1e3:.3f} kHz |",
        "",
        "## 3. 标称点结果",
        "",
        "| 项目 | 结果 |",
        "|---|---:|",
        f"| 开关频率 | {nominal.operating_point.switching_frequency_hz/1e3:.3f} kHz |",
        f"| 输入阻抗相角 | {nominal.operating_point.input_phase_deg:.3f}° |",
        f"| 谐振电流 RMS/峰值 | {nominal.operating_point.resonant_current_rms_a:.3f} / {nominal.operating_point.resonant_current_peak_a:.3f} A |",
        f"| 二次电流 RMS/峰值 | {nominal.operating_point.secondary_current_rms_a:.3f} / {nominal.operating_point.secondary_current_peak_a:.3f} A |",
        f"| ZVS 电荷/能量裕量 | {nominal.primary.zvs_charge_margin:.2f} / {nominal.primary.zvs_energy_margin:.2f} |",
        f"| 总损耗 | {nominal.total_loss_w:.3f} W |",
        f"| 估算效率 | {nominal.efficiency*100:.3f}% |",
        f"| 输出电容纹波电流 RMS | {nominal.output_capacitor.capacitor_current_rms_a:.3f} A |",
        f"| 输出纹波 | {nominal.output_capacitor.total_ripple_vpp:.3f} Vpp |",
        "",
        "## 4. 变压器初算",
        "",
        f"- 磁芯：`{t.core.part_number}`，{t.core.family}/{t.core.shape}，{t.core.manufacturer}，材料 {t.core.material}",
        f"- 有效参数：Ae={t.core.ae_mm2:.1f} mm²，Amin={t.core.amin_mm2:.1f} mm²，Aw={t.core.aw_mm2:.1f} mm²，Ve={t.core.ve_mm3/1000:.1f} cm³",
        f"- 绕组结构：`P{spec.primary_turns//2}-S{spec.secondary_turns}-P{spec.primary_turns-spec.primary_turns//2}`",
        f"- 一次线：{t.primary_wire.description}；每个半绕组 {t.primary_layers_per_half} 层",
        f"- 二次线：{t.secondary_wire.description}；{t.secondary_layers} 层",
        f"- 窗口利用率：{t.fill_factor*100:.2f}%",
        f"- 径向构建：{t.radial_build_mm:.2f} mm / 窗口高度 {t.core.window_height_mm:.2f} mm",
        f"- Lm 所需总气隙：{t.gap_total_mm:.3f} mm",
        f"- 最差 Bpk（Amin）：{t.worst_b_peak_t:.4f} T",
        f"- 标称热点估算：{nominal.transformer.estimated_hotspot_c:.1f} °C",
        f"- 标称磁芯/一次铜/二次铜损：{nominal.transformer.core_w:.3f} / {nominal.transformer.primary_copper_w:.3f} / {nominal.transformer.secondary_copper_w:.3f} W",
        "",
        "## 5. 外置谐振电感初算",
        "",
        f"- 磁芯：`{ind.core.part_number}`，{ind.core.family}/{ind.core.shape}，材料 {ind.core.material}",
        f"- 有效参数：Ae={ind.core.ae_mm2:.1f} mm²，Amin={ind.core.amin_mm2:.1f} mm²，Aw={ind.core.aw_mm2:.1f} mm²，Ve={ind.core.ve_mm3/1000:.1f} cm³",
        f"- 匝数：{ind.turns} T",
        f"- 绕线：{ind.wire.description}",
        f"- 层数：{ind.layers}（限制 ≤ {spec.resonant_inductor_max_layers}）",
        f"- 总气隙：{ind.gap_total_mm:.3f} mm",
        f"- 窗口利用率：{ind.fill_factor*100:.2f}%",
        f"- 最差 Bpk（Amin）：{ind.worst_b_peak_t:.4f} T",
        f"- 标称热点估算：{nominal.resonant_inductor.estimated_hotspot_c:.1f} °C",
        f"- 标称磁芯/铜损：{nominal.resonant_inductor.core_w:.3f} / {nominal.resonant_inductor.copper_w:.3f} W",
        "",
        "### 自动磁芯选型前列候选",
        "",
        "| 部件 | 排名 | 磁芯 | 家族 | 匝数 | 标称损耗 | 热点 | Bpk | 气隙 |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|",
        *[f"| 变压器 | {x.rank} | {x.part_number} | {x.family} | {spec.primary_turns}:{spec.secondary_turns} | {x.nominal_total_loss_w:.3f} W | {x.nominal_hotspot_c:.1f} °C | {x.worst_b_peak_t:.3f} T | {x.gap_total_mm:.2f} mm |" for x in t.alternatives[:5]],
        *[f"| 谐振电感 | {x.rank} | {x.part_number} | {x.family} | {x.turns} | {x.nominal_total_loss_w:.3f} W | {x.nominal_hotspot_c:.1f} °C | {x.worst_b_peak_t:.3f} T | {x.gap_total_mm:.2f} mm |" for x in ind.alternatives[:5]],
        "",
        "## 6. 标称磁性损耗深度分解",
        "",
        "| 损耗机制 | W |",
        "|---|---:|",
        *[f"| {key} | {value:.5f} |" for key, value in nominal.detailed_magnetics_breakdown().items()],
        "",
        "## 7. 母线维持时间",
        "",
        f"- 安装容量：{bus.installed_capacitance_f*1e6:.1f} µF",
        f"- 400 V 放电到 {spec.vbus_hold_end_v:.0f} V 的估算时间：{bus.hold_time_to_limit_s*1e3:.3f} ms",
        f"- 指定 {spec.requested_hold_time_s*1e3:.1f} ms 所需容量：{bus.required_capacitance_f*1e6:.1f} µF",
        f"- 指定时间结束时母线电压：{bus.predicted_end_voltage_v:.2f} V",
        "",
        "## 8. 工作点表",
        "",
        "| 母线 | 负载 | fs | 增益 | 相角 | Ir RMS | Is RMS | 损耗 | 效率 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for point in analysis.operating_points:
        op = point.operating_point
        lines.append(
            f"| {op.vbus_v:.0f} V | {op.load_fraction*100:.0f}% | "
            f"{op.switching_frequency_hz/1e3:.2f} kHz | {op.required_gain:.3f} | "
            f"{op.input_phase_deg:.2f}° | {op.resonant_current_rms_a:.2f} A | "
            f"{op.secondary_current_rms_a:.2f} A | {point.total_loss_w:.2f} W | "
            f"{point.efficiency*100:.2f}% |")
    lines.extend(["", "## 9. 可行性", "",
                  f"**结论：{'通过 V2 磁性初筛' if analysis.feasible else '未通过 V2 磁性初筛'}**", ""])
    if analysis.feasibility_reasons:
        lines.extend(f"- 失败：{reason}" for reason in analysis.feasibility_reasons)
    else:
        lines.append("- 当前硬约束未发现失败项。")
    lines.extend(["", "## 10. 模型警告", ""])
    lines.extend(f"- {warning}" for warning in analysis.warnings)
    lines.extend(["", "## 11. 图表", "",
                  "![增益曲线](gain_curves.png)", "",
                  "![损耗分解](loss_breakdown.png)", "",
                  "![磁性损耗深度分解](magnetics_loss_breakdown.png)", "",
                  "![效率工作点](efficiency_workpoints.png)", ""])
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def export_calculation_book(analysis: SystemAnalysis,
                            output_directory: str | Path) -> dict[str, Path]:
    out = Path(output_directory)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    paths["report"] = write_markdown_report(analysis, out / "LLC_calculation_book.md")
    rows = operating_point_rows(analysis)
    paths["operating_points"] = out / "operating_points.csv"
    pd.DataFrame(rows).to_csv(paths["operating_points"], index=False)
    paths["loss_breakdown_csv"] = out / "nominal_loss_breakdown.csv"
    pd.DataFrame([{"loss_item": key, "loss_w": value}
                  for key, value in analysis.nominal.breakdown().items()]).to_csv(
                      paths["loss_breakdown_csv"], index=False)
    paths["magnetics_breakdown_csv"] = out / "nominal_magnetics_breakdown.csv"
    pd.DataFrame([{"loss_item": key, "loss_w": value}
                  for key, value in analysis.nominal.detailed_magnetics_breakdown().items()]).to_csv(
                      paths["magnetics_breakdown_csv"], index=False)
    paths["core_library_csv"] = out / "magnetic_core_library.csv"
    core_rows = []
    for core in CoreDatabase().cores:
        core_rows.append({
            "part_number": core.part_number, "family": core.family,
            "shape": core.shape, "manufacturer": core.manufacturer,
            "material": core.material, "purposes": ",".join(core.purposes),
            "ae_mm2": core.ae_mm2, "amin_mm2": core.amin_mm2,
            "aw_mm2": core.aw_mm2, "ve_mm3": core.ve_mm3,
            "le_mm": core.le_mm, "window_width_mm": core.window_width_mm,
            "window_height_mm": core.window_height_mm,
            "mlt_primary_mm": core.mlt_primary_mm,
            "thermal_resistance_k_per_w": core.thermal_resistance_k_per_w,
            "core_mass_g": core.core_mass_g, "cost_usd": core.cost_usd,
        })
    pd.DataFrame(core_rows).to_csv(paths["core_library_csv"], index=False)
    paths["transformer_candidates_csv"] = out / "transformer_core_candidates.csv"
    pd.DataFrame([asdict(x) for x in analysis.transformer.alternatives]).to_csv(
        paths["transformer_candidates_csv"], index=False)
    paths["inductor_candidates_csv"] = out / "resonant_inductor_core_candidates.csv"
    pd.DataFrame([asdict(x) for x in analysis.resonant_inductor.alternatives]).to_csv(
        paths["inductor_candidates_csv"], index=False)
    paths["result_json"] = out / "analysis.json"
    payload = {
        "spec": _jsonable(asdict(analysis.spec)),
        "tank": _jsonable(asdict(analysis.tank)),
        "transformer": _jsonable(asdict(analysis.transformer)),
        "resonant_inductor": _jsonable(asdict(analysis.resonant_inductor)),
        "bus_capacitor": _jsonable(asdict(analysis.bus_capacitor)),
        "feasible": analysis.feasible,
        "feasibility_reasons": list(analysis.feasibility_reasons),
        "warnings": list(analysis.warnings),
        "operating_points": rows,
    }
    paths["result_json"].write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
    for plot_path in create_all_plots(analysis, out):
        paths[f"plot_{plot_path.stem}"] = plot_path
    return paths
