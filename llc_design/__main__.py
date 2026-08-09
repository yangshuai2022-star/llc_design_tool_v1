"""Command-line interface for the LLC design calculator."""

from __future__ import annotations

from pathlib import Path
from dataclasses import replace

import click

from .core.config import load_spec, save_spec
from .core.spec import LLCDesignSpec, PrimaryTopology
from .core.tank import GainNotReachableError
from .models.devices import DeviceDatabase
from .models.system import LLCSystemAnalyzer
from .magnetics.core import CoreDatabase
from .magnetics.transformer_designer import (TransformerSynthesisSettings,
                                              export_transformer_synthesis,
                                              load_transformer_core_presets,
                                              synthesize_transformer)
from .optimization.sweep import LLCOptimizer, OptimizationConfig
from .report.export import export_calculation_book
from .control.analysis import build_small_signal_analysis, export_small_signal_analysis
from .control.linearize import ControlInputKind
from .control.digital_loop import (
    ADCSamplingConfig,
    AnalogSenseConfig,
    CommandTimingConfig,
    FMLUTMode,
    FrequencyModulatorLUT,
    PIControllerConfig,
    PIFControllerConfig,
    PWMCountMode,
    TwoP2ZControllerConfig,
    build_digital_loop_analysis,
    export_digital_loop_analysis,
)
from .dynamics.plant import DynamicPhasorModel
from .dynamics.waveforms import reconstruct_dynamic_phasor_waveforms
from .dynamics.switched import SwitchedSimulationConfig, simulate_switched_steady_state
from .dynamics.export import export_waveform_bundle


def _base_spec(config: str | None) -> LLCDesignSpec:
    return load_spec(config) if config else LLCDesignSpec()


def _override(spec: LLCDesignSpec, **values) -> LLCDesignSpec:
    changes = {key: value for key, value in values.items() if value is not None}
    return spec.clone(**changes) if changes else spec


@click.group()
def cli():
    """LLC half/full-bridge + full-bridge SR design and optimization tool."""


@cli.command("transformer-design")
@click.option("--config", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--preset", default="TDK_PQ35_35_B65881A_N87", show_default=True,
              help="Ferrite/core preset key; GUI also allows direct datasheet entry.")
@click.option("--bmax", type=float, default=0.18, show_default=True, help="Design Bpk limit (T)")
@click.option("--strand-mm", type=float, default=0.10, show_default=True, help="Litz copper strand diameter (mm)")
@click.option("--strand-step", type=int, default=50, show_default=True, help="Strand-count rounding step")
@click.option("--j", "current_density", type=float, default=6.0, show_default=True, help="Target copper current density (A/mm^2)")
@click.option("--output", type=click.Path(file_okay=False), default="output/transformer_design", show_default=True)
def transformer_design(config, preset, bmax, strand_mm, strand_step, current_density, output):
    """Synthesize LLC transformer turns/Litz and export magnetic/copper losses."""
    spec = _base_spec(config)
    presets = load_transformer_core_presets()
    if preset not in presets:
        raise click.ClickException(f"unknown transformer preset: {preset}; available={', '.join(presets)}")
    settings = TransformerSynthesisSettings(
        max_flux_density_t=bmax,
        strand_copper_diameter_mm=strand_mm,
        strand_outer_diameter_mm=strand_mm * 1.12,
        strand_count_step=strand_step,
        current_density_target_a_per_mm2=current_density,
    )
    result = synthesize_transformer(spec, presets[preset], settings)
    paths = export_transformer_synthesis(result, output)
    click.echo(f"Core: {result.core.shape} {result.core.material_grade} {result.core.part_number}")
    click.echo(f"Turns: {result.primary_turns}:{result.secondary_turns} (ratio={result.actual_turns_ratio:.5f})")
    click.echo(f"Primary Litz: {result.primary_litz.description}; J={result.primary_litz.current_density_a_per_mm2:.2f} A/mm2")
    click.echo(f"Secondary Litz: {result.secondary_litz.description}; J={result.secondary_litz.current_density_a_per_mm2:.2f} A/mm2")
    click.echo(f"Bpk(max): {result.worst_b_peak_t*1e3:.2f} mT; fill={result.fill_factor*100:.1f}%")
    click.echo(f"Loss nominal: core={result.nominal_loss.core_w:.3f} W, CuP={result.nominal_loss.primary_copper_w:.3f} W, CuS={result.nominal_loss.secondary_copper_w:.3f} W, total={result.nominal_loss.total_w:.3f} W")
    click.echo(f"Feasible: {result.feasible}")
    for reason in result.reasons:
        click.echo(f"  - {reason}")
    click.echo(f"Export: {paths['json']}")


@cli.command()
@click.option("--config", type=click.Path(exists=True, dir_okay=False), default=None,
              help="JSON input file; command options override it.")
@click.option("--vbus", type=float, default=None, help="Nominal DC bus voltage (V)")
@click.option("--vout", type=float, default=None, help="Output voltage (V)")
@click.option("--pout", type=float, default=None, help="Output power (W)")
@click.option("--fr-khz", type=float, default=None, help="Resonant frequency (kHz)")
@click.option("--fmin-khz", type=float, default=None, help="Minimum frequency (kHz)")
@click.option("--fmax-khz", type=float, default=None, help="Maximum frequency (kHz)")
@click.option("--ln", type=float, default=None, help="Lm/Lr ratio")
@click.option("--q", "q_full", type=float, default=None, help="Full-load FHA Q")
@click.option("--np", "primary_turns", type=int, default=None, help="Primary turns")
@click.option("--ns", "secondary_turns", type=int, default=None, help="Secondary turns")
@click.option("--bus-cap-uf", type=float, default=None, help="Installed bus capacitance (uF)")
@click.option("--hold-ms", type=float, default=None, help="Requested hold-up time (ms)")
@click.option("--device", default=None, help="Primary MOSFET reference part (see: python -m llc_design devices)")
@click.option("--sr-device", default=None, help="SR MOSFET reference part (see: python -m llc_design devices)")
@click.option("--primary-parallel", type=int, default=None, help="MOSFETs in parallel per primary switch")
@click.option("--sr-parallel", type=int, default=None, help="MOSFETs per SR bridge position")
@click.option("--half-bridge", is_flag=True, default=False, help="Use primary half bridge")
@click.option("--transformer-core", default=None, help="Force transformer reference core")
@click.option("--inductor-core", default=None, help="Force resonant-inductor reference core")
@click.option("--transformer-family", multiple=True, type=click.Choice(["PQ", "EE", "EC", "EER", "ETD"]), help="Restrict automatic transformer core search; repeat for multiple families")
@click.option("--inductor-family", multiple=True, type=click.Choice(["PQ", "EE", "EC", "EER", "ETD"]), help="Restrict automatic resonant-inductor core search")
@click.option("--output", type=click.Path(file_okay=False), default="output/llc_baseline",
              show_default=True, help="Calculation-book output directory")
@click.option("--plots/--no-plots", default=True, help="Generate PNG plots")
def run(config, vbus, vout, pout, fr_khz, fmin_khz, fmax_khz, ln, q_full,
        primary_turns, secondary_turns, bus_cap_uf, hold_ms, device, sr_device,
        primary_parallel, sr_parallel,
        half_bridge, transformer_core, inductor_core, transformer_family, inductor_family, output, plots):
    """Run one complete LLC calculation and export the calculation book."""
    spec = _base_spec(config)
    if half_bridge and primary_turns is None:
        primary_turns = max(1, round(spec.primary_turns / 2))
    spec = _override(
        spec,
        vbus_nom_v=vbus,
        vout_v=vout,
        pout_w=pout,
        resonant_frequency_hz=None if fr_khz is None else fr_khz * 1e3,
        minimum_frequency_hz=None if fmin_khz is None else fmin_khz * 1e3,
        maximum_frequency_hz=None if fmax_khz is None else fmax_khz * 1e3,
        ln_ratio=ln,
        q_full_load=q_full,
        primary_turns=primary_turns,
        secondary_turns=secondary_turns,
        bus_capacitance_f=None if bus_cap_uf is None else bus_cap_uf * 1e-6,
        requested_hold_time_s=None if hold_ms is None else hold_ms * 1e-3,
        primary_device=device,
        sr_device=sr_device,
        primary_parallel_devices=primary_parallel,
        sr_parallel_devices_per_position=sr_parallel,
        primary_topology=PrimaryTopology.HALF_BRIDGE if half_bridge else None,
        transformer_core_families=tuple(transformer_family) if transformer_family else None,
        resonant_inductor_core_families=tuple(inductor_family) if inductor_family else None,
    )
    try:
        analysis = LLCSystemAnalyzer().analyze(
            spec, preferred_transformer_core=transformer_core,
            preferred_inductor_core=inductor_core)
    except GainNotReachableError as exc:
        click.echo(f"错误: 谐振增益不可达 - {exc}", err=True)
        click.echo("请调整匝比 (--np/--ns)、满载 Q (--q) 或谐振频率 (--fr-khz) 后重试。",
                   err=True)
        raise SystemExit(1)
    paths = export_calculation_book(analysis, output)
    if not plots:
        for key in ("plot_gain_curves", "plot_loss_breakdown", "plot_magnetics_loss_breakdown", "plot_efficiency_workpoints"):
            path = paths.get(key)
            if path and path.exists():
                path.unlink()

    nominal = analysis.nominal
    click.echo("\nLLC V2 calculation")
    click.echo(f"  Topology: {spec.primary_topology.value} + {spec.secondary_topology.value}")
    click.echo(f"  Input/output: {spec.vbus_nom_v:.0f} Vdc -> {spec.vout_v:.1f} V / {spec.pout_w/1000:.2f} kW")
    click.echo(f"  Tank: Lr={analysis.tank.lr_h*1e6:.3f} uH, Cr={analysis.tank.cr_f*1e9:.3f} nF, Lm={analysis.tank.lm_h*1e6:.3f} uH")
    click.echo(f"  Transformer: {analysis.transformer.core.part_number}, {spec.primary_turns}:{spec.secondary_turns}, fill={analysis.transformer.fill_factor*100:.1f}%")
    click.echo(f"  Resonant inductor: {analysis.resonant_inductor.core.part_number}, {analysis.resonant_inductor.turns} T, {analysis.resonant_inductor.layers} layers")
    click.echo(f"  Nominal: fs={nominal.operating_point.switching_frequency_hz/1e3:.2f} kHz, loss={nominal.total_loss_w:.2f} W, eta={nominal.efficiency*100:.3f}%")
    click.echo(f"  Feasibility: {'PASS' if analysis.feasible else 'FAIL'}")
    for reason in analysis.feasibility_reasons:
        click.echo(f"    - {reason}")
    click.echo(f"  Report: {paths['report']}")


@cli.command("waveforms")
@click.option("--config", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--vbus", type=float, default=None, help="DC bus voltage for the waveform work point")
@click.option("--load", "load_fraction", type=float, default=1.0, show_default=True,
              help="Load fraction, 1.0 = rated power")
@click.option("--mode", type=click.Choice(["fast", "detailed"]), default="fast", show_default=True)
@click.option("--cycles", type=int, default=2, show_default=True)
@click.option("--samples", type=int, default=512, show_default=True,
              help="Samples per switching cycle")
@click.option("--output", type=click.Path(file_okay=False), default="output/llc_waveforms",
              show_default=True)
def waveforms(config, vbus, load_fraction, mode, cycles, samples, output):
    """Generate synchronized LLC key-node voltage/current waveforms."""
    spec = _base_spec(config)
    system = LLCSystemAnalyzer().analyze(spec)
    small = build_small_signal_analysis(
        spec,
        vbus_v=spec.vbus_nom_v if vbus is None else vbus,
        load_fraction=load_fraction,
        system_analysis=system,
    )
    model = DynamicPhasorModel(small.parameters)
    if mode == "fast":
        bundle = reconstruct_dynamic_phasor_waveforms(
            model, small.steady_state,
            cycles=cycles, samples_per_cycle=samples)
    else:
        bundle = simulate_switched_steady_state(
            model, small.steady_state,
            SwitchedSimulationConfig(
                samples_per_cycle=samples,
                output_cycles=cycles,
            ),
        )
    paths = export_waveform_bundle(bundle, output)
    click.echo(f"Waveform model: {bundle.model_name}")
    click.echo(f"Work point: {small.operating_point.vbus_v:.1f} V / {load_fraction*100:.1f}% / "
               f"{small.operating_point.switching_frequency_hz/1e3:.3f} kHz")
    for key in ("i_resonant", "v_resonant_cap", "i_transformer_secondary", "i_output_cap", "v_output"):
        signal = bundle.signals[key]
        stats = signal.statistics
        click.echo(f"  {signal.label}: RMS={stats.rms:.5g} {signal.unit}, "
                   f"min/max={stats.minimum:.5g}/{stats.maximum:.5g} {signal.unit}")
    click.echo(f"Report: {paths['report']}")


@cli.command("small-signal")
@click.option("--config", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--vbus", type=float, default=None, help="DC bus voltage for linearization")
@click.option("--load", "load_fraction", type=float, default=1.0, show_default=True)
@click.option("--sample-us", type=float, default=20.0, show_default=True,
              help="Fixed digital-control sample period in microseconds")
@click.option("--input-kind", type=click.Choice([item.value for item in ControlInputKind]),
              default=ControlInputKind.FREQUENCY_HZ.value, show_default=True)
@click.option("--timer-clock-mhz", type=float, default=None,
              help="Timer clock, required for timer_counts input")
@click.option("--delay-samples", type=int, default=0, show_default=True,
              help="Integer computation/PWM update delay")
@click.option("--series-resistance", type=float, default=None,
              help="Override primary-side equivalent tank damping resistance (ohm)")
@click.option("--output", type=click.Path(file_okay=False), default="output/llc_small_signal",
              show_default=True)
def small_signal(config, vbus, load_fraction, sample_us, input_kind,
                 timer_clock_mhz, delay_samples, series_resistance, output):
    """Build Gvf(s), Bode, exact ZOH G(z), difference equation and C99 code."""
    spec = _base_spec(config)
    system = LLCSystemAnalyzer().analyze(spec)
    kind = ControlInputKind(input_kind)
    timer_clock_hz = None if timer_clock_mhz is None else timer_clock_mhz * 1e6
    result = build_small_signal_analysis(
        spec,
        vbus_v=spec.vbus_nom_v if vbus is None else vbus,
        load_fraction=load_fraction,
        sample_time_s=sample_us * 1e-6,
        control_input_kind=kind,
        timer_clock_hz=timer_clock_hz,
        input_delay_samples=delay_samples,
        system_analysis=system,
        series_resistance_ohm=series_resistance,
    )
    paths = export_small_signal_analysis(result, output)
    tf = result.continuous_transfer
    click.echo("LLC seven-state dynamic-phasor/EDF small-signal model")
    click.echo(f"  Work point: {result.operating_point.vbus_v:.1f} V / "
               f"{load_fraction*100:.1f}% / {result.operating_point.switching_frequency_hz/1e3:.3f} kHz")
    click.echo(f"  EDF steady output: {result.steady_state.output_voltage_v:.6f} V")
    click.echo(f"  G(0): {tf.dc_gain:.9g} {tf.output_unit}/{tf.input_unit}")
    click.echo(f"  Continuous/discrete stable: {result.continuous_plant.stable}/{result.discrete_plant.stable}")
    click.echo(f"  Difference equation: {result.discrete_plant.difference_equation.text(precision=9)}")
    click.echo(f"  Report: {paths['report']}")


@cli.command("digital-loop")
@click.option("--config", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--vbus", type=float, default=None, help="DC bus voltage for the loop work point")
@click.option("--load", "load_fraction", type=float, default=1.0, show_default=True)
@click.option("--sample-us", type=float, default=20.0, show_default=True)
@click.option("--controller", type=click.Choice(["pi", "pif", "2p2z"]), default="pif", show_default=True)
@click.option("--kp", type=float, default=0.01, show_default=True)
@click.option("--ti-ms", type=float, default=1.0, show_default=True)
@click.option("--pif-fc", type=float, default=3500.0, show_default=True, help="PIF output LPF cutoff (Hz)")
@click.option("--b0", type=float, default=0.0)
@click.option("--b1", type=float, default=0.0)
@click.option("--b2", type=float, default=0.0)
@click.option("--a1", type=float, default=0.0, help="2P2Z denominator a1")
@click.option("--a2", type=float, default=0.0, help="2P2Z denominator a2")
@click.option("--pcmd", type=float, default=None, help="Manual PCMD; omitted = infer from work-point frequency")
@click.option("--fm-lut", type=click.Path(exists=True, dir_okay=False), default=None,
              help="CSV/text PCMD LUT. Default is the 20-point firmware TBPRD table.")
@click.option("--fm-mode", type=click.Choice(["tbprd", "frequency"]), default="tbprd", show_default=True)
@click.option("--timer-mhz", type=float, default=120.0, show_default=True)
@click.option("--count-mode", type=click.Choice(["up-down", "up"]), default="up-down", show_default=True)
@click.option("--cla-delay-us", type=float, default=1.0, show_default=True)
@click.option("--adc-conversion-cycles", type=float, default=13.0, show_default=True)
@click.option("--output", type=click.Path(file_okay=False), default="output/llc_digital_loop", show_default=True)
def digital_loop(config, vbus, load_fraction, sample_us, controller, kp, ti_ms,
                 pif_fc, b0, b1, b2, a1, a2, pcmd, fm_lut, fm_mode,
                 timer_mhz, count_mode, cla_delay_us, adc_conversion_cycles,
                 output):
    """Build the complete PI/PIF/2P2Z + FM + ADC + LLC voltage loop."""
    spec = _base_spec(config)
    system = LLCSystemAnalyzer().analyze(spec)
    sample_time_s = sample_us * 1e-6
    small = build_small_signal_analysis(
        spec,
        vbus_v=spec.vbus_nom_v if vbus is None else vbus,
        load_fraction=load_fraction,
        sample_time_s=sample_time_s,
        control_input_kind=ControlInputKind.FREQUENCY_HZ,
        timer_clock_hz=timer_mhz * 1e6,
        system_analysis=system,
    )
    if controller == "pi":
        controller_cfg = PIControllerConfig(
            kp=kp, ti_s=ti_ms * 1e-3, sample_time_s=sample_time_s)
    elif controller == "pif":
        controller_cfg = PIFControllerConfig(
            kp=kp, ti_s=ti_ms * 1e-3, lpf_cutoff_hz=pif_fc,
            sample_time_s=sample_time_s)
    else:
        controller_cfg = TwoP2ZControllerConfig(
            b0=b0, b1=b1, b2=b2, a1=a1, a2=a2,
            sample_time_s=sample_time_s)

    mode = FMLUTMode.PCMD_TO_TBPRD if fm_mode == "tbprd" else FMLUTMode.PCMD_TO_FREQUENCY
    pwm_mode = PWMCountMode.UP_DOWN if count_mode == "up-down" else PWMCountMode.UP
    if fm_lut:
        lut = FrequencyModulatorLUT.from_text(
            Path(fm_lut).read_text(encoding="utf-8"),
            mode=mode,
            timer_clock_hz=timer_mhz * 1e6,
            count_mode=pwm_mode,
        )
    else:
        lut = FrequencyModulatorLUT.firmware_default()
        if timer_mhz != 120.0 or pwm_mode != PWMCountMode.UP_DOWN:
            lut = FrequencyModulatorLUT(
                lut.pcmd, lut.values, lut.mode,
                timer_mhz * 1e6, pwm_mode, lut.name)

    result = build_digital_loop_analysis(
        small,
        controller_config=controller_cfg,
        fm_lut=lut,
        command_pu=pcmd,
        analog_sense=AnalogSenseConfig(
            rup_ohm=117e3,
            rlow_ohm=1.6e3,
            divider_capacitance_f=1e-9,
            opamp_gain=1.0,
            adc_series_resistance_ohm=220.0,
            adc_shunt_capacitance_f=2e-9,
        ),
        adc_sampling=ADCSamplingConfig(
            control_sample_time_s=sample_time_s,
            adc_clock_hz=60e6,
            acquisition_time_s=300e-9,
            conversion_cycles=adc_conversion_cycles,
            soc_count=3,
            recursive_previous_weight=0.25,
        ),
        command_timing=CommandTimingConfig(
            computation_delay_s=cla_delay_us * 1e-6),
    )
    paths = export_digital_loop_analysis(result, output)
    margin = result.margins_nominal_delay
    click.echo("LLC complete digital voltage loop")
    click.echo(f"  Controller: {result.controller.name}")
    click.echo(f"  PCMD: {result.fm_operating_point.command_pu:.8g}")
    click.echo(f"  Kfm: {result.fm_operating_point.gain_hz_per_pu:.9g} Hz/pu")
    click.echo(f"  Gain crossover: {margin.critical_gain_crossover_hz}")
    click.echo(f"  Phase margin: {margin.phase_margin_deg}")
    click.echo(f"  Gain margin: {margin.gain_margin_db}")
    click.echo(f"  Discrete closed-loop stable: {result.discrete_approximation.stable}")
    click.echo(f"  Output: {paths['settings_json']}")


@cli.command("gui")
@click.option("--config", type=click.Path(exists=True, dir_okay=False), default=None)
def gui(config):
    """Launch the optional PySide6 engineering GUI."""
    try:
        from .gui.app import run_gui
    except ImportError as exc:
        raise click.ClickException(
            "GUI dependencies are not installed. Run: pip install -e '.[gui]'"
        ) from exc
    raise SystemExit(run_gui(config))


@cli.command()
@click.option("--config", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--profile", type=click.Choice(["quick", "full"]), default="quick",
              show_default=True)
@click.option("--half-bridge", is_flag=True, default=False,
              help="Optimize a half-bridge LLC and scale the turns search accordingly")
@click.option("--max-candidates", type=int, default=None,
              help="Optional early stop for development runs")
@click.option("--top", type=int, default=10, show_default=True)
@click.option("--output", type=click.Path(file_okay=False), default="output/llc_optimization",
              show_default=True)
def optimize(config, profile, half_bridge, max_candidates, top, output):
    """Run multi-variable tank/turns/device/SR optimization."""
    spec = _base_spec(config)
    if half_bridge:
        spec = spec.clone(primary_topology=PrimaryTopology.HALF_BRIDGE,
                          primary_turns=max(1, round(spec.primary_turns / 2)))
    cfg = OptimizationConfig.quick() if profile == "quick" else OptimizationConfig.full()
    if spec.primary_topology == PrimaryTopology.HALF_BRIDGE:
        cfg = replace(cfg, primary_turn_values=tuple(max(1, round(n / 2))
                                                     for n in cfg.primary_turn_values))
    click.echo(f"Running {profile} sweep: up to {cfg.count} candidates")
    result = LLCOptimizer().run(spec, cfg, max_candidates)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    result.table.to_csv(out / "optimization_all.csv", index=False)
    result.pareto.to_csv(out / "optimization_pareto.csv", index=False)
    if result.best_analysis is not None:
        export_calculation_book(result.best_analysis, out / "best_design")
    columns = [c for c in ["feasible", "weighted_loss_w", "nominal_efficiency_pct",
                           "ln", "q_full", "fr_khz", "primary_turns",
                           "secondary_turns", "lr_uh", "cr_nf", "lm_uh",
                           "transformer_core", "inductor_core", "minimum_zvs_margin"]
               if c in result.table.columns]
    click.echo(result.table.loc[:, columns].head(top).to_string(index=False))
    click.echo(f"Results: {out / 'optimization_all.csv'}")


@cli.command("write-template")
@click.option("--path", type=click.Path(dir_okay=False), default="llc_baseline.json",
              show_default=True)
def write_template(path):
    """Write the agreed 400 V -> 53 V / 3 kW baseline JSON input."""
    output = save_spec(LLCDesignSpec(), path)
    click.echo(str(output))


@cli.command()
def devices():
    """List bundled reference MOSFET records."""
    db = DeviceDatabase()
    click.echo("Primary devices")
    for d in db.primary:
        click.echo(f"  {d.part_number:<22} {d.vds_max_v:>4.0f} V  Rds25={d.rds_on_25_ohm*1e3:>5.1f} mOhm  {d.technology}")
    click.echo("SR devices")
    for d in db.sr:
        click.echo(f"  {d.part_number:<22} {d.vds_max_v:>4.0f} V  Rds25={d.rds_on_25_ohm*1e3:>5.2f} mOhm  {d.technology}")


@cli.command()
@click.option("--family", type=click.Choice(["PQ", "EE", "EC", "EER", "ETD"]), default=None)
@click.option("--purpose", type=click.Choice(["transformer", "inductor"]), default=None)
def cores(family, purpose):
    """List standard PQ/EE/EC/EER/ETD magnetic-core records."""
    db = CoreDatabase()
    rows = db.cores
    if family:
        rows = [core for core in rows if core.family == family]
    if purpose:
        rows = [core for core in rows if core.supports(purpose)]
    for core in rows:
        click.echo(
            f"{core.part_number:<24} {core.family:<4} {core.shape:<12} "
            f"{core.material:<5} Ae/Amin={core.ae_mm2:>5.0f}/{core.amin_mm2:<5.0f} mm2  "
            f"Aw={core.aw_mm2:>5.0f} mm2  Ve={core.ve_mm3/1000:>6.1f} cm3  "
            f"{','.join(core.purposes)}")


@cli.command()
def materials():
    """List ferrite material models and fitted operating ranges."""
    db = CoreDatabase().material_db
    for material in db.materials:
        f0, f1 = material.frequency_range_hz
        b0, b1 = material.flux_range_t
        click.echo(
            f"{material.key:<22} {material.manufacturer:<16} {material.grade:<10} "
            f"f={f0/1e3:.0f}..{f1/1e3:.0f} kHz  B={b0:.2f}..{b1:.2f} T  "
            f"Bsat100={material.bsat_at(100.0):.3f} T")


if __name__ == "__main__":
    cli()
