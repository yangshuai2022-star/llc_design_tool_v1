#!/usr/bin/env python3
"""交互式自定义 LLC 设计参数, 保存到 llc_custom.json。

用法:
  python custom_input.py                 # 交互式输入 (直接回车使用默认值)
  python custom_input.py --reuse         # 直接沿用上次保存的参数
  python custom_input.py vbus=200 ...    # 命令行指定, 未指定项使用默认值

说明:
  - 修改输入电压 vbus 时, 母线最低/最高/保压结束电压按同一比例自动缩放。
  - 其余未提示的参数 (器件、死区、Litz 线规等) 可直接编辑 llc_custom.json。
"""

from __future__ import annotations

import sys
from pathlib import Path

from llc_design.core.config import load_spec, save_spec
from llc_design.core.spec import LLCDesignSpec

ROOT = Path(__file__).resolve().parent
CUSTOM = ROOT / "llc_custom.json"

PROMPTS = [
    ("vbus", "输入母线电压 vbus (V)", "400"),
    ("vout", "输出电压 vout (V)", "53"),
    ("pout", "输出功率 pout (W)", "3000"),
    ("fr", "谐振频率 fr (kHz)", "100"),
    ("ln", "Ln = Lm/Lr", "5.0"),
    ("q", "满载 FHA Q", "0.35"),
    ("np", "原边匝数", "30"),
    ("ns", "副边匝数", "4"),
    ("hold_ms", "保压时间 (ms)", "20"),
    ("bus_mF", "母线电容 (mF)", "1.8"),
]

BOUND_KEYS = ("vbus", "vout", "pout", "fr", "ln", "q", "np", "ns", "hold_ms", "bus_mF")


def _load_previous() -> LLCDesignSpec | None:
    if not CUSTOM.exists():
        return None
    try:
        return load_spec(CUSTOM)
    except Exception:
        return None


def _fmt_default(spec: LLCDesignSpec) -> dict[str, str]:
    return {
        "vbus": f"{spec.vbus_nom_v:g}",
        "vout": f"{spec.vout_v:g}",
        "pout": f"{spec.pout_w:g}",
        "fr": f"{spec.resonant_frequency_hz / 1e3:g}",
        "ln": f"{spec.ln_ratio:g}",
        "q": f"{spec.q_full_load:g}",
        "np": f"{spec.primary_turns:g}",
        "ns": f"{spec.secondary_turns:g}",
        "hold_ms": f"{spec.requested_hold_time_s * 1e3:g}",
        "bus_mF": f"{spec.bus_capacitance_f * 1e3:g}",
    }


def _ask(prompt: str, default: str) -> str:
    try:
        raw = input(f"  {prompt:<26}[{default}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise SystemExit(0)
    return raw if raw else default


def _ask_yesno(prompt: str, default: bool) -> bool:
    hint = "(Y/n)" if default else "(y/N)"
    ans = _ask(f"{prompt} {hint}", "").strip().lower()
    if not ans:
        return default
    return ans in ("y", "yes", "是")


def _build_spec(base: LLCDesignSpec, values: dict[str, str]) -> LLCDesignSpec:
    spec = base.clone(
        vbus_nom_v=float(values["vbus"]),
        vout_v=float(values["vout"]),
        pout_w=float(values["pout"]),
        resonant_frequency_hz=float(values["fr"]) * 1e3,
        ln_ratio=float(values["ln"]),
        q_full_load=float(values["q"]),
        primary_turns=int(float(values["np"])),
        secondary_turns=int(float(values["ns"])),
        requested_hold_time_s=float(values["hold_ms"]) * 1e-3,
        bus_capacitance_f=float(values["bus_mF"]) * 1e-3,
    )
    if spec.vbus_nom_v != base.vbus_nom_v:
        ratio = spec.vbus_nom_v / base.vbus_nom_v
        spec = spec.clone(
            vbus_min_normal_v=base.vbus_min_normal_v * ratio,
            vbus_max_v=base.vbus_max_v * ratio,
            vbus_hold_end_v=base.vbus_hold_end_v * ratio,
        )
    return spec


def _print_summary(spec: LLCDesignSpec) -> None:
    print("\n  本次设计参数:")
    print(f"    输入: {spec.vbus_nom_v:.0f} Vdc ({spec.vbus_min_normal_v:.0f}-{spec.vbus_max_v:.0f} V, "
          f"保压至 {spec.vbus_hold_end_v:.0f} V)")
    print(f"    输出: {spec.vout_v:.1f} V / {spec.pout_w:.0f} W")
    print(f"    谐振: fr={spec.resonant_frequency_hz / 1e3:.1f} kHz, Ln={spec.ln_ratio:.2f}, Qfull={spec.q_full_load:.2f}")
    print(f"    匝比: {spec.primary_turns}:{spec.secondary_turns}")
    print(f"    母线: {spec.bus_capacitance_f * 1e3:.2f} mF, 保压 {spec.requested_hold_time_s * 1e3:.1f} ms")


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--reuse"]
    reuse_prev = "--reuse" in sys.argv
    cli_pairs: dict[str, str] = {}
    for arg in args:
        if "=" not in arg:
            print(f"错误: 无法识别的参数 '{arg}', 应为 名称=值 或 --reuse")
            return 1
        key, value = arg.split("=", 1)
        cli_pairs[key.strip()] = value.strip()

    unknown = set(cli_pairs) - set(BOUND_KEYS)
    if unknown:
        print(f"错误: 未知参数 {', '.join(sorted(unknown))}")
        return 1

    prev = _load_previous()
    print("=" * 56)
    print("  自定义 LLC 参数 (直接回车使用默认值)")
    print("=" * 56)

    if reuse_prev:
        if prev is None:
            print(f"错误: 没有上次参数文件 {CUSTOM}")
            return 1
        base = prev
        values = _fmt_default(prev)
        values.update(cli_pairs)
    elif cli_pairs:
        base = prev if prev is not None else LLCDesignSpec()
        values = _fmt_default(base)
        values.update(cli_pairs)
    else:
        if prev is not None:
            if not _ask_yesno("检测到上次参数, 直接沿用", True):
                base = prev
                values = {key: _ask(label, _fmt_default(prev).get(key, default))
                          for key, label, default in PROMPTS}
            else:
                base = prev
                values = _fmt_default(prev)
        else:
            base = LLCDesignSpec()
            values = {key: _ask(label, default) for key, label, default in PROMPTS}

    spec = _build_spec(base, values)
    try:
        spec.validate()
    except ValueError as exc:
        print(f"\n错误: 参数不合法 - {exc}")
        return 1

    _print_summary(spec)
    save_spec(spec, CUSTOM)
    print(f"  已保存: {CUSTOM}")
    print("=" * 56)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
