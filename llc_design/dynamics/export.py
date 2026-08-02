"""CSV, plot and Markdown export for LLC waveform bundles."""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .waveforms import WaveformBundle


DEFAULT_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("primary_tank", ("v_leg_a", "v_leg_b", "v_bridge", "i_resonant", "v_resonant_cap")),
    ("transformer", ("i_transformer_primary", "i_magnetizing", "i_primary_load", "v_transformer_secondary")),
    ("secondary_output", ("v_transformer_secondary", "i_transformer_secondary", "i_rectified", "i_output_cap", "v_output_ripple")),
    ("switching", ("gate_q1", "vds_q1", "ids_q1", "gate_q2", "vds_q2", "ids_q2")),
    ("sr_switching", ("gate_sr1", "vds_sr1", "ids_sr1", "gate_sr2", "vds_sr2", "ids_sr2")),
    ("magnetics", ("b_transformer", "h_transformer", "b_resonant_inductor", "h_resonant_inductor", "energy_lr")),
)


def _configure_cjk_fonts() -> None:
    """Use installed CJK fonts when available without bundling font files."""
    plt.rcParams["font.sans-serif"] = [
        "AR PL UMing CN", "Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False


def _plot_group(bundle: WaveformBundle, keys: tuple[str, ...], path: Path, title: str) -> Path:
    _configure_cjk_fonts()
    available = [key for key in keys if key in bundle.signals]
    if not available:
        raise ValueError(f"no requested waveform signals are available for {title}")
    fig, axes = plt.subplots(len(available), 1, figsize=(11, 2.25 * len(available)), sharex=True)
    if len(available) == 1:
        axes = [axes]
    time_us = bundle.time_s * 1e6
    for axis, key in zip(axes, available):
        signal = bundle.signals[key]
        axis.plot(time_us, signal.values)
        stats = signal.statistics
        axis.set_ylabel(f"{signal.label}\n({signal.unit})")
        axis.grid(True, alpha=0.3)
        axis.text(
            0.995, 0.92,
            f"RMS={stats.rms:.4g}  min={stats.minimum:.4g}  max={stats.maximum:.4g}",
            transform=axis.transAxes, ha="right", va="top", fontsize=8)
    axes[-1].set_xlabel("Time (µs)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def write_waveform_markdown(bundle: WaveformBundle, path: str | Path) -> Path:
    output = Path(path)
    lines = [
        "# LLC 关键节点波形计算书",
        "",
        "> 版权说明：工具设计人 **杨帅锅** · 开关电源仿真与实用设计",
        "",
        f"- 模型：`{bundle.model_name}`",
        f"- 开关频率：{bundle.switching_frequency_hz/1e3:.6f} kHz",
        f"- 记录周期：{bundle.time_s[-1] * bundle.switching_frequency_hz:.3f}",
        "",
        "## 波形统计",
        "",
        "| 信号 | 单位 | 频率 | 平均值 | RMS | 最小值 | 最大值 | 峰峰值 | 基波 RMS | THD |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for signal in bundle.signals.values():
        s = signal.statistics
        lines.append(
            f"| {signal.label} | {signal.unit} | {s.frequency_hz/1e3:.7g} kHz | {s.average:.7g} | {s.rms:.7g} | "
            f"{s.minimum:.7g} | {s.maximum:.7g} | {s.peak_to_peak:.7g} | "
            f"{s.fundamental_rms:.7g} | {s.thd_percent:.3f}% |")
    lines.extend(["", "## 模型警告", ""])
    lines.extend(f"- {warning}" for warning in bundle.warnings)
    lines.extend(["", "## 图表", ""])
    for group, _ in DEFAULT_GROUPS:
        lines.extend([f"![{group}]({group}.png)", ""])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def export_waveform_bundle(bundle: WaveformBundle, directory: str | Path) -> dict[str, Path]:
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = bundle.export_csv(output)
    metadata = {
        "model_name": bundle.model_name,
        "switching_frequency_hz": bundle.switching_frequency_hz,
        "warnings": list(bundle.warnings),
        "metadata": dict(bundle.metadata),
        "signals": {
            key: {
                "label": signal.label,
                "unit": signal.unit,
                "group": signal.group,
                "description": signal.description,
                "statistics": {
                    field: float(getattr(signal.statistics, field))
                    for field in signal.statistics.__dataclass_fields__
                },
            }
            for key, signal in bundle.signals.items()
        },
    }
    paths["metadata"] = output / "waveform_metadata.json"
    paths["metadata"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["report"] = write_waveform_markdown(bundle, output / "LLC_waveforms.md")
    for group, keys in DEFAULT_GROUPS:
        paths[f"plot_{group}"] = _plot_group(
            bundle, keys, output / f"{group}.png",
            f"LLC {group.replace('_', ' ')} waveforms ({bundle.model_name})")
    return paths
