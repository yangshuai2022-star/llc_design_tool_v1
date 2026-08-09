"""Plots for gain, efficiency and loss screening."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..core.tank import equivalent_ac_load_ohm, gain, target_gain
from ..models.system import SystemAnalysis


def plot_gain_curves(analysis: SystemAnalysis, path: str | Path) -> Path:
    spec, tank = analysis.spec, analysis.tank
    frequencies = np.linspace(spec.minimum_frequency_hz,
                              spec.maximum_frequency_hz, 1000)
    fig, ax = plt.subplots(figsize=(10, 6))
    for load in (0.10, 0.25, 0.50, 0.75, 1.00):
        pout = spec.pout_w * max(load, spec.minimum_modeled_load_fraction)
        transferred_power = pout * (1.0 + spec.rectifier_equivalent_drop_v / spec.vout_v)
        rac = equivalent_ac_load_ohm(
            spec.turns_ratio, spec.vout_v + spec.rectifier_equivalent_drop_v,
            transferred_power)
        values = [gain(tank, float(f), rac) for f in frequencies]
        ax.plot(frequencies / 1e3, values, label=f"{load*100:.0f}% load")

    for vbus in (spec.vbus_max_v, spec.vbus_nom_v,
                 spec.vbus_min_normal_v, spec.vbus_hold_end_v):
        required = target_gain(spec, vbus)
        ax.axhline(required, linestyle="--", linewidth=0.9,
                   label=f"required @ {vbus:.0f} V = {required:.3f}")
    ax.axvline(tank.fr_hz / 1e3, linestyle=":", linewidth=1.2, label="fr")
    ax.axvline(spec.minimum_frequency_hz / 1e3, linestyle=":", linewidth=0.8)
    ax.axvline(spec.maximum_frequency_hz / 1e3, linestyle=":", linewidth=0.8)

    for point in analysis.operating_points:
        op = point.operating_point
        ax.scatter(op.switching_frequency_hz / 1e3, op.achieved_gain, s=28)
        if op.load_fraction >= 0.999:
            ax.annotate(f"{op.vbus_v:.0f}V", (op.switching_frequency_hz / 1e3,
                                              op.achieved_gain),
                        textcoords="offset points", xytext=(4, 5), fontsize=8)
    ax.set_title("LLC FHA gain curves and solved operating points")
    ax.set_xlabel("Switching frequency (kHz)")
    ax.set_ylabel("Normalized DC gain")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_loss_breakdown(analysis: SystemAnalysis, path: str | Path) -> Path:
    breakdown = analysis.nominal.breakdown()
    grouped = {
        "Primary bridge": sum(v for k, v in breakdown.items() if k.startswith("primary_")),
        "SR bridge": sum(v for k, v in breakdown.items() if k.startswith("sr_")),
        "Transformer": sum(v for k, v in breakdown.items() if k.startswith("transformer_")),
        "Resonant inductor": sum(v for k, v in breakdown.items() if k.startswith("resonant_inductor_")),
        "Resonant capacitor": breakdown["resonant_capacitor_w"],
        "Output capacitor": breakdown["output_capacitor_w"],
        "Auxiliary": breakdown["auxiliary_w"],
    }
    labels = list(grouped)
    values = [grouped[k] for k in labels]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.barh(labels, values)
    for bar, value in zip(bars, values):
        ax.text(value + max(values) * 0.01, bar.get_y() + bar.get_height()/2,
                f"{value:.1f} W", va="center", fontsize=9)
    ax.set_xlabel("Loss (W)")
    ax.set_title("Nominal 400 V / 100% load loss breakdown")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_efficiency_map(analysis: SystemAnalysis, path: str | Path) -> Path:
    groups: dict[float, list[tuple[float, float]]] = {}
    for point in analysis.operating_points:
        op = point.operating_point
        groups.setdefault(op.vbus_v, []).append((op.load_fraction * 100.0,
                                                 point.efficiency * 100.0))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for vbus, points in sorted(groups.items()):
        points.sort()
        ax.plot([p[0] for p in points], [p[1] for p in points], marker="o",
                label=f"{vbus:.0f} V")
    ax.set_xlabel("Load (%)")
    ax.set_ylabel("Estimated efficiency (%)")
    ax.set_title("Calculated LLC efficiency work points")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output



def plot_magnetics_loss_breakdown(analysis: SystemAnalysis, path: str | Path) -> Path:
    breakdown = analysis.nominal.detailed_magnetics_breakdown()
    labels = [key.replace("_w", "").replace("_", " ") for key in breakdown]
    values = list(breakdown.values())
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(labels, values)
    max_value = max(values) if values else 1.0
    for bar, value in zip(bars, values):
        if value > max_value * 0.002:
            ax.text(value + max_value * 0.008, bar.get_y() + bar.get_height()/2,
                    f"{value:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Loss (W)")
    ax.set_title("Nominal detailed transformer and resonant-inductor losses")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output

def create_all_plots(analysis: SystemAnalysis, output_directory: str | Path) -> list[Path]:
    out = Path(output_directory)
    return [
        plot_gain_curves(analysis, out / "gain_curves.png"),
        plot_loss_breakdown(analysis, out / "loss_breakdown.png"),
        plot_magnetics_loss_breakdown(analysis, out / "magnetics_loss_breakdown.png"),
        plot_efficiency_map(analysis, out / "efficiency_workpoints.png"),
    ]
