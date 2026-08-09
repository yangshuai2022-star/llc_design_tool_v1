#!/usr/bin/env python3
"""Print a compact engineering summary of a design run (analysis.json)."""
import json
import sys
from pathlib import Path


def fmt(v, nd=3):
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def core_material(c: dict) -> str:
    material = c.get("material")
    if material:
        return material
    return c.get("material_spec", {}).get("grade", "?")


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: show_params.py <output目录> [计算书名.md]")
        return 1
    out = Path(sys.argv[1])
    analysis_path = out / "analysis.json"
    if not analysis_path.exists():
        print(f"错误: 找不到 {analysis_path}")
        return 1
    d = json.loads(analysis_path.read_text())
    spec = d["spec"]
    tank = d["tank"]
    tr = d["transformer"]
    li = d["resonant_inductor"]
    ops = d["operating_points"]

    W = 76
    line = "=" * W

    print(line)
    print("  LLC 设计摘要")
    print(line)

    print("\n【一次/二次侧】")
    print(f"  拓扑: {spec['primary_topology'].replace('_',' ')}  +  {spec['secondary_topology'].replace('_',' ')}")
    print(f"  输入: {spec['vbus_nom_v']:.0f} Vdc ({spec['vbus_min_normal_v']:.0f}-{spec['vbus_max_v']:.0f} V),  "
          f"输出: {spec['vout_v']:.0f} V / {spec['pout_w']:.0f} W")
    print(f"  匝比: {spec['primary_turns']}:{spec['secondary_turns']}   "
          f"fr={tank['fr_hz']/1e3:.0f} kHz  Ln={tank['ln_ratio']}  Qfull={tank['q_full_load']}")
    print(f"  母线电容: {spec['bus_capacitance_f']*1e3:.2f} mF (保压至 {d['bus_capacitor']['predicted_end_voltage_v']:.0f} V, "
          f"保压时间 {d['bus_capacitor']['hold_time_to_limit_s']*1e3:.2f} ms)")

    print("\n【谐振槽】")
    print(f"  Lr = {tank['lr_h']*1e6:.3f} uH   Cr = {tank['cr_f']*1e9:.3f} nF   Lm = {tank['lm_h']*1e6:.3f} uH")
    print(f"  Z0 = {tank['zr_ohm']:.2f} ohm   Rac(等效负载) = {tank['rac_nom_ohm']:.2f} ohm")

    print("\n【变压器】")
    c = tr["core"]
    print(f"  磁芯: {c['part_number']} ({c['shape']}, {core_material(c)})")
    print(f"  Ae={c['ae_mm2']:.0f} mm2  Aw={c['aw_mm2']:.0f} mm2  Ve={c['ve_mm3']:.0f} mm3  le={c['le_mm']:.0f} mm")
    print(f"  匝数: 原边 {tr['primary_turns']} T / 副边 {tr['secondary_turns']} T (P/2-S-P/2)")
    pw, sw = tr["primary_wire"], tr["secondary_wire"]
    print(f"  原边线: 0.1 mm Litz x{pw['strand_count']} 股 (d包={pw['strand_outer_diameter_m']*1e3:.3f} mm, 束数 {pw['sub_bundle_count']})")
    print(f"  副边线: 0.1 mm Litz x{sw['strand_count']} 股 (束数 {sw['sub_bundle_count']})")
    print(f"  填充率 {tr['fill_factor']*100:.1f}%  总气隙 {tr['gap_total_mm']:.2f} mm  "
          f"Bpk={tr['b_t']*1e3:.0f} mT" if tr.get("b_t") else
          f"  填充率 {tr['fill_factor']*100:.1f}%  总气隙 {tr['gap_total_mm']:.2f} mm")

    print("\n【谐振电感】")
    c = li["core"]
    print(f"  磁芯: {c['part_number']} ({c['shape']}, {core_material(c)})")
    print(f"  Ae={c['ae_mm2']:.0f} mm2  Aw={c['aw_mm2']:.0f} mm2  Ve={c['ve_mm3']:.0f} mm3  le={c['le_mm']:.0f} mm")
    w = li["wire"]
    print(f"  L = {li['inductance_h']*1e6:.3f} uH, {li['turns']:.0f} T / {li['layers']:.0f} 层")
    print(f"  线: 0.1 mm Litz x{w['strand_count']} 股")
    print(f"  填充率 {li['fill_factor']*100:.1f}%  总气隙 {li['gap_total_mm']:.2f} mm  "
          f"Bpk={li['b_t']*1e3:.0f} mT" if li.get("b_t") else
          f"  填充率 {li['fill_factor']*100:.1f}%  总气隙 {li['gap_total_mm']:.2f} mm")

    print("\n【开关器件选型】")
    prim = spec["primary_device"]
    print(f"  一次侧: {prim}  x{spec['primary_parallel_devices']}  (650V 类参考器件, 每桥臂 1 并)")
    print(f"  同步整流: {spec['sr_device']}  x{spec['sr_parallel_devices_per_position']} 并联/位置 (全桥共 "
          f"{spec['sr_parallel_devices_per_position']*2} 管/每桥臂对)")
    print("  (参考数据, 硬件定版前需替换为真实器件曲线)")

    print("\n【额定工况损耗分解】")
    nom = [p for p in ops if p["vbus_v"] == spec["vbus_nom_v"] and p["load_pct"] == 100.0][0]
    items = [
        ("一次侧导通", nom["primary_conduction_w"]), ("一次侧关断", nom["primary_turnoff_w"]),
        ("一次侧死区", nom["primary_deadtime_w"]), ("一次侧驱动", nom["primary_gate_w"]),
        ("一次侧Coss", nom["primary_coss_w"]), ("SR导通", nom["sr_conduction_w"]),
        ("SR死区", nom["sr_deadtime_w"]), ("SR关断", nom["sr_turnoff_w"]),
        ("SR Coss", nom["sr_coss_w"]), ("SR驱动", nom["sr_gate_w"]),
        ("变压器磁芯", nom["transformer_core_w"]), ("变压器原边铜损", nom["transformer_primary_copper_w"]),
        ("变压器副边铜损", nom["transformer_secondary_copper_w"]),
        ("谐振电感磁芯", nom["resonant_inductor_core_w"]), ("谐振电感铜损", nom["resonant_inductor_copper_w"]),
        ("谐振电容", nom["resonant_capacitor_w"]), ("输出电容", nom["output_capacitor_w"]),
        ("辅助电源", nom["auxiliary_w"]),
    ]
    total = sum(v for _, v in items)
    half = (len(items) + 1) // 2
    for i in range(half):
        left = f"  {items[i][0]:<12}{items[i][1]:7.2f} W"
        if i + half < len(items):
            right = f"  {items[i+half][0]:<12}{items[i+half][1]:7.2f} W"
        else:
            right = ""
        print(left + right)
    print(f"  {'合计':<12}{total:7.2f} W    额定效率 {nom['efficiency_pct']:.3f}%  (fs={nom['frequency_khz']:.1f} kHz)")

    print("\n【全工况点】")
    print(f"  {'工况':<12}{'fs/kHz':>8}{'增益':>7}{'Ir RMS/A':>9}{'峰值/A':>8}{'损耗/W':>8}{'效率/%':>8}")
    for p in ops:
        print(f"  {p['label']:<12}{p['frequency_khz']:>8.1f}{p['required_gain']:>7.3f}"
              f"{p['resonant_current_rms_a']:>9.1f}{p['resonant_current_peak_a']:>8.1f}"
              f"{p['total_loss_w']:>8.2f}{p['efficiency_pct']:>8.2f}")

    feasible = d.get("feasible", False)
    print(f"\n  可行性: {'PASS' if feasible else 'FAIL'}   "
          + ("; ".join(d.get("feasibility_reasons", [])) if not feasible else ""))

    book = out / "LLC_calculation_book.md"
    if book.exists():
        print(f"  计算书: {book}")
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
