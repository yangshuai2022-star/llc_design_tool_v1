"""Command-line entry point for PFC Control Lab."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import click

from .control import (
    LoadModel,
    PFCControlLabConfig,
    build_pfc_control_lab_analysis,
    build_pfc_switching_waveforms,
    export_pfc_control_lab,
    simulate_pfc_line_cycle,
)


@click.command(name="pfc-control-lab")
@click.option("--vin", type=float, default=230.0, show_default=True, help="AC input RMS voltage")
@click.option("--vbus", type=float, default=400.0, show_default=True, help="DC bus command")
@click.option("--power", type=float, default=3300.0, show_default=True, help="Output power")
@click.option("--line-hz", type=float, default=50.0, show_default=True)
@click.option("--fsw-khz", type=float, default=50.0, show_default=True)
@click.option("--inductor-uh", type=float, default=220.0, show_default=True)
@click.option("--line-angle", type=float, default=60.0, show_default=True)
@click.option("--load-model", type=click.Choice(["constant_power", "resistive"]), default="constant_power")
@click.option("--switch-cycles", type=click.IntRange(1, 50), default=2, show_default=True)
@click.option("--switch-samples", type=click.IntRange(100, 10000), default=800, show_default=True)
@click.option("--output", type=click.Path(path_type=Path), default=Path("output/pfc_control_lab"))
def cli(vin: float, vbus: float, power: float, line_hz: float, fsw_khz: float,
        inductor_uh: float, line_angle: float, load_model: str,
        switch_cycles: int, switch_samples: int, output: Path) -> None:
    """Run current-loop, voltage-loop, sensing and waveform analysis."""

    base = PFCControlLabConfig()
    stage = replace(
        base.power_stage,
        vin_rms_v=vin,
        bus_voltage_v=vbus,
        output_power_w=power,
        line_frequency_hz=line_hz,
        switching_frequency_hz=fsw_khz * 1e3,
        boost_inductance_h=inductor_uh * 1e-6,
        line_angle_deg=line_angle,
        load_model=LoadModel(load_model),
    )
    config = replace(
        base, power_stage=stage,
        switching_cycles=switch_cycles,
        switching_samples_per_cycle=switch_samples,
    )
    analysis = build_pfc_control_lab_analysis(config)
    line_cycle = simulate_pfc_line_cycle(config)
    switching = build_pfc_switching_waveforms(config, line_cycle=line_cycle)
    paths = export_pfc_control_lab(analysis, line_cycle, switching, output)
    ci = analysis.current_loop.margins
    cv = analysis.voltage_loop.margins
    click.echo(f"Current loop: fc={ci.critical_gain_crossover_hz}, PM={ci.phase_margin_deg}")
    click.echo(f"Voltage loop: fc={cv.critical_gain_crossover_hz}, PM={cv.phase_margin_deg}")
    click.echo(f"Line cycle: PF={line_cycle.metrics.power_factor:.7f}, THD={line_cycle.metrics.current_thd_percent:.5f}%")
    click.echo(f"Output: {output.resolve()}")
    for name, path in paths.items():
        click.echo(f"  {name}: {path.name}")


if __name__ == "__main__":
    cli()
