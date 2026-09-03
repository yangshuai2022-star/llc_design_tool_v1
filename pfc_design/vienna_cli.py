"""CLI for three-phase Vienna PFC Control Lab."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import click
import numpy as np
import pandas as pd

from .vienna import (
    ViennaControlLabConfig,
    build_vienna_control_lab_analysis,
    build_vienna_switching_waveforms,
    simulate_vienna_line_cycle,
)


@click.command(name="vienna-control-lab")
@click.option("--vll", type=float, default=400.0, show_default=True, help="Line-line RMS voltage")
@click.option("--vdc", type=float, default=700.0, show_default=True, help="Total split-bus voltage")
@click.option("--power", type=float, default=10000.0, show_default=True, help="Output power")
@click.option("--line-hz", type=float, default=50.0, show_default=True)
@click.option("--fsw-khz", type=float, default=65.0, show_default=True)
@click.option("--inductor-uh", type=float, default=600.0, show_default=True)
@click.option("--switch-angle", type=float, default=30.0, show_default=True)
@click.option("--output", type=click.Path(path_type=Path), default=Path("output/vienna_control_lab"))
def cli(vll: float, vdc: float, power: float, line_hz: float, fsw_khz: float,
        inductor_uh: float, switch_angle: float, output: Path) -> None:
    base = ViennaControlLabConfig()
    stage = replace(
        base.power_stage,
        line_line_rms_v=vll,
        bus_voltage_v=vdc,
        output_power_w=power,
        line_frequency_hz=line_hz,
        switching_frequency_hz=fsw_khz*1e3,
        boost_inductance_h=inductor_uh*1e-6,
    )
    cfg = replace(base, power_stage=stage, switching_line_angle_deg=switch_angle)
    analysis = build_vienna_control_lab_analysis(cfg)
    line = simulate_vienna_line_cycle(cfg)
    switching = build_vienna_switching_waveforms(cfg, line, line_angle_deg=switch_angle)

    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "current_loop": asdict(analysis.current_loop.margins),
        "voltage_loop": asdict(analysis.voltage_loop.margins),
        "balance_loop": asdict(analysis.balance_loop.margins),
        "waveform_metrics": asdict(line.metrics),
        "warnings": [*analysis.warnings, *line.warnings],
    }
    (output/"summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame({"time_s": line.time_s, **line.signals}).to_csv(output/"line_cycle.csv", index=False)
    pd.DataFrame({"time_s": switching.time_s, **switching.signals}).to_csv(output/"switching.csv", index=False)
    rows = {"frequency_hz": analysis.frequencies_hz}
    for prefix, loop in (("current",analysis.current_loop),("voltage",analysis.voltage_loop),("balance",analysis.balance_loop)):
        key = {"current":"open_current","voltage":"open_voltage","balance":"open_balance"}[prefix]
        resp = loop.responses[key]
        rows[f"{prefix}_open_gain_db"] = 20*np.log10(np.maximum(np.abs(resp),1e-300))
        rows[f"{prefix}_open_phase_deg"] = np.unwrap(np.angle(resp))*180/np.pi
    pd.DataFrame(rows).to_csv(output/"bode_open_loops.csv", index=False)

    click.echo(f"Vienna current loop PM={analysis.current_loop.margins.phase_margin_deg}")
    click.echo(f"Vienna voltage loop PM={analysis.voltage_loop.margins.phase_margin_deg}")
    click.echo(f"Vienna balance loop PM={analysis.balance_loop.margins.phase_margin_deg}")
    click.echo(f"PF={line.metrics.overall_power_factor:.7f}, THD A/B/C={line.metrics.phase_current_thd_percent}")
    click.echo(f"Output: {output.resolve()}")


if __name__ == "__main__":
    cli()
