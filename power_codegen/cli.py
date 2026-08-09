"""CLI for C99 float32 control-code generation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import click

from llc_design.control.analysis import build_small_signal_analysis
from llc_design.control.digital_loop import PIControllerConfig, build_digital_loop_analysis
from llc_design.core.spec import LLCDesignSpec
from llc_design.models.system import LLCSystemAnalyzer
from pfc_design.control import PFCControlLabConfig, build_pfc_control_lab_analysis, tune_pfc_current_loop
from pfc_design.vienna import ViennaControlLabConfig, build_vienna_control_lab_analysis

from .generator import generate_llc_control_code, generate_ttpl_control_code, generate_vienna_control_code


@click.group()
def cli() -> None:
    """Generate portable C99/float32 real-time control-step code."""


@cli.command("ttpl")
@click.option("--output", type=click.Path(file_okay=False), default="output/generated_ttpl", show_default=True)
@click.option("--autotune/--no-autotune", default=True, help="Apply the conservative one-click current-loop tuner before generation.")
@click.option("--allow-unstable", is_flag=True, default=False)
def ttpl(output: str, autotune: bool, allow_unstable: bool) -> None:
    cfg = PFCControlLabConfig()
    if autotune:
        tune = tune_pfc_current_loop(cfg)
        cfg = replace(cfg, current_controller=tune.controller)
    analysis = build_pfc_control_lab_analysis(cfg)
    result = generate_ttpl_control_code(analysis, output, require_stable=not allow_unstable)
    click.echo(f"TTPL C99 generated: {result.directory}")
    click.echo(f"Validation: {'PASS' if result.validation.passed else 'FAIL'}")


@cli.command("vienna")
@click.option("--output", type=click.Path(file_okay=False), default="output/generated_vienna", show_default=True)
@click.option("--allow-unstable", is_flag=True, default=False)
def vienna(output: str, allow_unstable: bool) -> None:
    analysis = build_vienna_control_lab_analysis(ViennaControlLabConfig())
    result = generate_vienna_control_code(analysis, output, require_stable=not allow_unstable)
    click.echo(f"Vienna C99 generated: {result.directory}")
    click.echo(f"Validation: {'PASS' if result.validation.passed else 'FAIL'}")


@cli.command("llc")
@click.option("--output", type=click.Path(file_okay=False), default="output/generated_llc", show_default=True)
@click.option("--kp", type=float, default=0.002, show_default=True)
@click.option("--ti-ms", type=float, default=3.0, show_default=True)
@click.option("--allow-unstable", is_flag=True, default=False)
def llc(output: str, kp: float, ti_ms: float, allow_unstable: bool) -> None:
    spec = LLCDesignSpec()
    system = LLCSystemAnalyzer().analyze(spec)
    small = build_small_signal_analysis(spec, system_analysis=system, sample_time_s=20e-6)
    loop = build_digital_loop_analysis(
        small,
        controller_config=PIControllerConfig(kp=kp, ti_s=ti_ms*1e-3, sample_time_s=20e-6),
    )
    result = generate_llc_control_code(loop, output, require_stable=not allow_unstable)
    click.echo(f"LLC C99 generated: {result.directory}")
    click.echo(f"Validation: {'PASS' if result.validation.passed else 'FAIL'}")


if __name__ == "__main__":
    cli()
