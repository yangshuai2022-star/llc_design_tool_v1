"""CSV/JSON/plot export for PFC Control Lab."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from llc_design.control.digital_loop import DigitalTransferFunction

from .analysis import PFCControlLabAnalysis
from .waveforms import PFCLineCycleWaveforms, PFCSwitchingWaveforms


def _response_frame(frequencies_hz, responses: dict[str, np.ndarray]) -> pd.DataFrame:
    data: dict[str, np.ndarray] = {"frequency_hz": np.asarray(frequencies_hz)}
    for name, response in responses.items():
        value = np.asarray(response, dtype=complex)
        data[f"{name}_gain_db"] = 20.0 * np.log10(np.maximum(np.abs(value), 1e-300))
        data[f"{name}_phase_deg"] = np.unwrap(np.angle(value)) * 180.0 / np.pi
    return pd.DataFrame(data)


def _plot_loop(path: Path, frequencies, curves, title: str) -> Path:
    figure, (ax_mag, ax_phase) = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    for label, response in curves:
        value = np.asarray(response, dtype=complex)
        ax_mag.semilogx(frequencies, 20.0 * np.log10(np.maximum(np.abs(value), 1e-300)), label=label)
        ax_phase.semilogx(frequencies, np.unwrap(np.angle(value)) * 180.0 / np.pi, label=label)
    ax_mag.axhline(0.0, linewidth=0.8, linestyle="--")
    ax_mag.set_ylabel("Magnitude (dB)")
    ax_phase.set_ylabel("Phase (deg)")
    ax_phase.set_xlabel("Frequency (Hz)")
    for axis in (ax_mag, ax_phase):
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=8)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path



def export_pfc_controller_c99(
    controller: DigitalTransferFunction,
    path: str | Path,
    *,
    prefix: str,
    function_name: str,
    output_min: float,
    output_max: float,
) -> Path:
    """Export a uniquely-prefixed C99 Direct-Form-I controller."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    num = controller.numerator
    den = controller.denominator
    order_x = len(num)
    order_y = max(len(den) - 1, 0)
    macro = prefix.upper()
    def cf(value: float) -> str:
        text = f"{float(value):.9g}"
        if "e" not in text.lower() and "." not in text:
            text += ".0"
        return text + "f"
    lines = [
        "/* Auto-generated PFC digital controller. C99, Direct Form I. */",
        "#include <stddef.h>", "",
        f"#define {macro}_NX ({order_x}u)",
        f"#define {macro}_NY ({order_y}u)", "",
        "typedef struct {",
        f"    float x_hist[{macro}_NX];",
        f"    float y_hist[({macro}_NY > 0u) ? {macro}_NY : 1u];",
        f"}} {prefix}_state_t;", "",
        f"static const float {prefix}_b[{macro}_NX] = {{{', '.join(cf(v) for v in num)}}};",
        f"static const float {prefix}_a[({macro}_NY > 0u) ? {macro}_NY : 1u] = "
        + "{" + (", ".join(cf(v) for v in den[1:]) if order_y else "0.0f") + "};", "",
        f"float {function_name}({prefix}_state_t *state, float error)", "{",
        "    size_t i;", "    float output = 0.0f;", "",
        f"    for(i = {macro}_NX - 1u; i > 0u; --i) {{",
        "        state->x_hist[i] = state->x_hist[i - 1u];", "    }",
        "    state->x_hist[0] = error;",
        f"    for(i = 0u; i < {macro}_NX; ++i) {{",
        f"        output += {prefix}_b[i] * state->x_hist[i];", "    }",
        f"    for(i = 0u; i < {macro}_NY; ++i) {{",
        f"        output -= {prefix}_a[i] * state->y_hist[i];", "    }",
        f"    if(output > {cf(output_max)}) output = {cf(output_max)};",
        f"    if(output < {cf(output_min)}) output = {cf(output_min)};",
        f"    if({macro}_NY > 0u) {{",
        f"        for(i = {macro}_NY - 1u; i > 0u; --i) {{",
        "            state->y_hist[i] = state->y_hist[i - 1u];", "        }",
        "        state->y_hist[0] = output;", "    }",
        "    return output;", "}", "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def export_pfc_control_lab(
    analysis: PFCControlLabAnalysis,
    line_cycle: PFCLineCycleWaveforms,
    switching: PFCSwitchingWaveforms,
    directory: str | Path,
) -> dict[str, Path]:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    summary = out / "pfc_control_lab_summary.json"
    summary.write_text(json.dumps(analysis.summary_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    paths["summary"] = summary

    current_csv = out / "pfc_current_loop_bode.csv"
    _response_frame(analysis.frequencies_hz, analysis.current_loop.responses).to_csv(current_csv, index=False)
    paths["current_bode"] = current_csv
    voltage_csv = out / "pfc_voltage_loop_bode.csv"
    _response_frame(analysis.frequencies_hz, analysis.voltage_loop.responses).to_csv(voltage_csv, index=False)
    paths["voltage_bode"] = voltage_csv

    sensing_csv = out / "pfc_sensing_summary.csv"
    pd.DataFrame([
        asdict(analysis.current_sense_summary),
        asdict(analysis.vac_sense_summary),
        asdict(analysis.vbus_sense_summary),
    ]).to_csv(sensing_csv, index=False)
    paths["sensing"] = sensing_csv

    waveform_csv = out / "pfc_line_cycle_waveforms.csv"
    wave_data = {"time_s": line_cycle.time_s}
    wave_data.update(line_cycle.signals)
    pd.DataFrame(wave_data).to_csv(waveform_csv, index=False)
    paths["line_cycle"] = waveform_csv
    metrics_path = out / "pfc_line_cycle_metrics.json"
    metrics_path.write_text(json.dumps(asdict(line_cycle.metrics), indent=2), encoding="utf-8")
    paths["metrics"] = metrics_path

    switching_csv = out / "pfc_switching_waveforms.csv"
    switch_data = {"time_s": switching.time_s}
    switch_data.update(switching.signals)
    pd.DataFrame(switch_data).to_csv(switching_csv, index=False)
    paths["switching"] = switching_csv

    current_plot = out / "pfc_current_loop_bode.png"
    _plot_loop(current_plot, analysis.frequencies_hz, [
        ("Li current open loop", analysis.current_loop.responses["open_current"]),
    ], "PFC current-loop stability (open loop)")
    paths["current_plot"] = current_plot

    voltage_plot = out / "pfc_voltage_loop_bode.png"
    _plot_loop(voltage_plot, analysis.frequencies_hz, [
        ("Lv voltage open loop", analysis.voltage_loop.responses["open_voltage"]),
    ], "PFC voltage-loop stability (open loop)")
    paths["voltage_plot"] = voltage_plot

    # Export the final settled AC period rather than all warm-up periods.
    line_hz = analysis.config.power_stage.line_frequency_hz
    dt = float(np.mean(np.diff(line_cycle.time_s)))
    samples_per_line = max(int(round(1.0 / line_hz / dt)), 2)
    selection = slice(max(len(line_cycle.time_s) - samples_per_line, 0), len(line_cycle.time_s))
    t_ms = (line_cycle.time_s[selection] - line_cycle.time_s[selection][0]) * 1e3
    signals = {key: np.asarray(value)[selection] for key, value in line_cycle.signals.items()}

    figure, axes = plt.subplots(6, 1, sharex=True, figsize=(12, 12))
    axes[0].plot(t_ms, signals["vac"], label="Vac actual")
    axes[0].plot(t_ms, signals["vac_measured"], label="Vac measured")
    axes[1].plot(t_ms, signals["i_input_signed"], label="Iin")
    axes[1].plot(t_ms, signals["i_ref"], label="Iref")
    axes[1].plot(t_ms, signals["i_measured_signed"], label="I measured")
    axes[2].plot(t_ms, signals["current_error"], label="Current error")
    axes[3].plot(t_ms, signals["vbus"], label="Vbus")
    axes[3].plot(t_ms, signals["vbus_measured"], label="Vbus measured")
    axes[4].plot(t_ms, signals["input_power"], label="Input power")
    axes[5].plot(t_ms, signals["bus_cap_current"], label="Bus capacitor current")
    axes[5].plot(t_ms, signals["boost_output_current"], label="Boost output current")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    axes[-1].set_xlabel("Time in final AC period (ms)")
    line_plot = out / "pfc_one_ac_period_waveforms.png"
    figure.tight_layout()
    figure.savefig(line_plot, dpi=160)
    plt.close(figure)
    paths["line_plot"] = line_plot

    figure, axes = plt.subplots(6, 1, sharex=True, figsize=(12, 12))
    axes[0].plot(t_ms, signals["vac_rms_estimate"], label="Vac RMS estimate")
    axes[1].plot(t_ms, signals["vloop"], label="Voltage-loop output")
    axes[1].plot(t_ms, signals["voltage_error"], label="Vbus error")
    axes[2].plot(t_ms, signals["gcmd"], label="gcmd")
    axes[3].plot(t_ms, signals["duty_ff"], label="Duty FF")
    axes[3].plot(t_ms, signals["duty_pi"], label="Duty PI")
    axes[3].plot(t_ms, signals["duty_total"], label="Duty total")
    axes[4].plot(t_ms, signals["indu_comp"], label="indu_comp")
    axes[4].plot(t_ms, signals["minimum_pulse_active"], label="Min-pulse active")
    axes[5].plot(t_ms, signals["amc_update_strobe"], label="AMC update")
    axes[5].plot(t_ms, signals["voltage_update_strobe"], label="Voltage-loop update")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    axes[-1].set_xlabel("Time in final AC period (ms)")
    control_plot = out / "pfc_one_ac_period_control_waveforms.png"
    figure.tight_layout()
    figure.savefig(control_plot, dpi=160)
    plt.close(figure)
    paths["control_plot"] = control_plot

    figure, axes = plt.subplots(5, 1, sharex=True, figsize=(12, 10))
    t_us = switching.time_s * 1e6
    ss = switching.signals
    axes[0].plot(t_us, ss["hf_high_gate"], label="HF high gate")
    axes[0].plot(t_us, ss["hf_low_gate"], label="HF low gate")
    axes[0].plot(t_us, ss["lf_polarity_gate"], label="LF polarity")
    axes[1].plot(t_us, ss["switch_node_voltage"], label="Switch node")
    axes[1].plot(t_us, ss["inductor_voltage"], label="Inductor voltage")
    axes[2].plot(t_us, ss["inductor_current"], label="Inductor current")
    axes[2].plot(t_us, ss["boost_output_current"], label="Boost output current")
    axes[3].plot(t_us, ss["high_side_current"], label="HF high current")
    axes[3].plot(t_us, ss["low_side_current"], label="HF low current")
    axes[4].plot(t_us, ss["bus_cap_current"], label="Bus capacitor current")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    axes[-1].set_xlabel("Time (us)")
    switching_plot = out / "pfc_switching_cycle_waveforms.png"
    figure.tight_layout()
    figure.savefig(switching_plot, dpi=160)
    plt.close(figure)
    paths["switching_plot"] = switching_plot

    current_cfg = analysis.config.current_controller
    current_c = export_pfc_controller_c99(
        analysis.current_loop.controller, out / "pfc_current_controller.c",
        prefix="pfc_current_ctrl", function_name="pfc_current_controller_run",
        output_min=current_cfg.output_min, output_max=current_cfg.output_max,
    )
    paths["current_c"] = current_c
    voltage_cfg = analysis.config.voltage_controller
    voltage_c = export_pfc_controller_c99(
        analysis.voltage_loop.controller, out / "pfc_voltage_controller.c",
        prefix="pfc_voltage_ctrl", function_name="pfc_voltage_controller_run",
        output_min=voltage_cfg.output_min, output_max=voltage_cfg.output_max,
    )
    paths["voltage_c"] = voltage_c
    return paths


__all__ = ["export_pfc_control_lab", "export_pfc_controller_c99"]
