"""CLI entry point for PFC Design Calculator."""

import os
import subprocess
import click
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .core.spec import DesignSpec, MosfetDatabase, DiodeDatabase
from .core.operating_point import compute_mathcad_operating_point, OperatingPoint
from .magnetics.core_database import CoreDatabase
from .models.system import SystemAnalyzer
from .optimization.sweep import ParamSweep
from .optimization.profile import (
    classify_results, get_recommended_designs,
    infeasible_reason_summary, per_core_best, per_mosfet_best,
)


@click.group()
def cli():
    """PFC Design Calculator - Two-phase interleaved PFC design tool."""
    pass


@cli.command()
@click.option("--vin", default=176.0, help="Minimum input voltage (Vrms)")
@click.option("--vout", default=410.0, help="Output voltage (Vdc)")
@click.option("--pout", default=7100.0, help="Total output power (W)")
@click.option("--fsw", default=65000.0, help="Switching frequency (Hz)")
@click.option("--ripple", default=0.3, help="Current ripple ratio")
@click.option("--core", default=None, help="Preferred core part number")
@click.option("--mosfet", default=None, help="Preferred MOSFET part number")
@click.option("--n-cores", default=2, help="Number of stacked cores")
@click.option("--plot/--no-plot", default=True, help="Generate plots")
@click.option("--save-plot", default=None, help="Save plots to directory")
@click.option("--trace/--no-trace", default=False, help="Export line-cycle loss trace CSV")
@click.option("--diode", default=None, help="Boost diode part number (e.g. C4D10120H)")
@click.option("--thermal", is_flag=True, default=False,
              help="Self-consistent loss<->temperature loop (Tj/T_winding)")
def run(vin, vout, pout, fsw, ripple, core, mosfet, n_cores, plot, save_plot,
        trace, diode, thermal):
    """Single-point PFC design analysis with plots."""
    spec = DesignSpec(vin_min=vin, vout=vout, pout_total=pout, fsw=fsw,
                      ripple_ratio=ripple, thermal_model=thermal)
    if diode:
        spec.diode_part = diode   # None -> keep the spec default (SiC part)
    db = CoreDatabase()
    mdb = MosfetDatabase()

    preferred_core = db.get_by_part_number(core) if core else None
    preferred_mosfet = mdb.get(mosfet) if mosfet else None

    if core and not preferred_core:
        click.echo(f"Core '{core}' not found, auto-selecting.")
    if mosfet and not preferred_mosfet:
        click.echo(f"MOSFET '{mosfet}' not found, auto-selecting.")

    if diode and not DiodeDatabase().get(diode):
        click.echo(f"Diode '{diode}' not found, falling back to spec Vf/Rd.")

    analyzer = SystemAnalyzer(db)
    result = analyzer.analyze(spec, preferred_core=preferred_core, mosfet=preferred_mosfet, n_cores=n_cores)

    _print_summary(result)
    _print_loss_table(result)

    if trace:
        out_dir = save_plot or os.path.join(os.path.expanduser("~"), "pfc_design", "output")
        os.makedirs(out_dir, exist_ok=True)
        _export_loss_trace_csv(result, os.path.join(out_dir, "loss_trace.csv"))

    if plot:
        if save_plot is None:
            save_plot = os.path.join(os.path.expanduser("~"), "pfc_design", "output")
        _generate_plots(result, save_plot)


@cli.command()
@click.option("--vin", default=176.0, help="Min input voltage (Vrms)")
@click.option("--vout", default=410.0, help="Output voltage (Vdc)")
@click.option("--pout", default=7100.0, help="Total output power (W)")
@click.option("--fsw", default=0.0, help="Fixed switching freq (kHz). If set, fsw is not swept.")
@click.option("--fsw-start", default=45.0, help="fsw sweep start (kHz)")
@click.option("--fsw-stop", default=120.0, help="fsw sweep stop (kHz)")
@click.option("--fsw-step", default=10.0, help="fsw sweep step (kHz)")
@click.option("--ripple-start", default=0.15, help="Ripple ratio start")
@click.option("--ripple-stop", default=0.50, help="Ripple ratio stop")
@click.option("--ripple-n", default=6, help="Ripple ratio points")
@click.option("--tech", default="Si", help="MOSFET technology filter (Si/SiC/Si CoolMOS/Si SuperJunction/GaN/all)")
@click.option("--core-material", default="Sendust", help="Core material class filter")
@click.option("--core-limit", default=10, help="Maximum number of core candidates")
@click.option("--n-cores", default=2, help="Stacked cores")
@click.option("--allow-aggressive-ripple", is_flag=True, default=False,
              help="Allow designs with actual ripple exceeding limits (warning only)")
@click.option("--diverse", is_flag=True, default=False,
              help="Show per-core and per-MOSFET best candidates")
@click.option("--refine", is_flag=True, default=False,
              help="Local-refine top grid designs (scipy Nelder-Mead on fsw x ripple)")
@click.option("--heuristic", is_flag=True, default=False,
              help="Run differential-evolution global search after the grid sweep")
@click.option("--diode", default=None, help="Boost diode part number (e.g. C4D10120H)")
@click.option("--thermal", is_flag=True, default=False,
              help="Self-consistent loss<->temperature loop in the sweep")
@click.option("--top", default=10, help="Show top N results")
@click.option("--csv", default=None, help="Save full sweep results to CSV")
def optimize(vin, vout, pout, fsw, fsw_start, fsw_stop, fsw_step,
             ripple_start, ripple_stop, ripple_n, tech, core_material,
             core_limit, n_cores, top, csv, allow_aggressive_ripple,
             diverse, refine, heuristic, diode, thermal):
    """Sweep optimization: ripple x core x MOSFET (fsw optional sweep)."""
    spec = DesignSpec(vin_min=vin, vout=vout, pout_total=pout,
                      thermal_model=thermal)
    if diode:
        spec.diode_part = diode   # None -> keep the spec default (SiC part)

    db = CoreDatabase()
    mdb = MosfetDatabase()
    cores = db.top_for_pfc(ae_min_cm2=0.5, material_class=core_material, n=core_limit)
    mosfets = mdb.query(vds_min=spec.vout * 1.25, technology=tech)

    # Fixed or swept fsw
    if fsw > 0:
        fsw_arr = np.array([fsw * 1000])
        fsw_label = f"fsw: {fsw}kHz (fixed)"
    else:
        fsw_arr = np.arange(fsw_start * 1000, (fsw_stop + fsw_step/2) * 1000, fsw_step * 1000)
        fsw_label = f"fsw: {fsw_start}-{fsw_stop}kHz"

    if len(mosfets) == 0:
        click.echo(f"No MOSFETs found for '{tech}' with Vds > {spec.vout*1.25:.0f}V. Try --tech all")
        return
    if len(cores) == 0:
        click.echo(f"No cores found for material '{core_material}'. Try --core-material Sendust or --core-material all is not supported yet.")
        return

    click.echo(f"\n{'='*75}")
    click.echo(f"  PFC Optimization Sweep")
    click.echo(f"  {fsw_label} / ripple: {ripple_start}-{ripple_stop}")
    click.echo(f"  Cores: {len(cores)} ({core_material})  |  MOSFETs: {len(mosfets)} ({tech})")
    click.echo(f"{'='*75}")

    sweep_vars = {
        "fsw": fsw_arr,
        "ripple_ratio": np.linspace(ripple_start, ripple_stop, int(ripple_n)),
        "core_idx": list(range(len(cores))),
        "mosfet_idx": list(range(len(mosfets))),
    }

    click.echo(f"Total points: {np.prod([len(v) for v in sweep_vars.values()]):.0f}")
    click.echo("Running sweep...")

    sweeper = ParamSweep(db)
    df = sweeper.sweep(spec, sweep_vars, n_cores=n_cores, cores=cores, mosfets=mosfets,
                       allow_aggressive_ripple=allow_aggressive_ripple)

    # --- Classify results ---
    df = classify_results(df)
    recommendations = get_recommended_designs(df)

    if csv:
        csv_dir = os.path.dirname(os.path.abspath(csv))
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        df.to_csv(csv, index=False)
        click.echo(f"Full sweep CSV saved: {csv}")

    feasible = df[df["feasible"]]
    n_prod = df["design_profile_feasible_production"].sum()
    n_bal = df["design_profile_feasible_balanced"].sum()
    n_agg = df["design_profile_feasible_aggressive"].sum()
    click.echo(f"\nResults: {len(df)} total, "
               f"{len(feasible)} feasible  "
               f"[production:{n_prod}  balanced:{n_bal}  aggressive:{n_agg}]\n")

    # --- Recommendations ---
    _print_recommendations(recommendations)

    if len(feasible) > 0:
        top_n = feasible.nsmallest(top, "P_total_W")

        # --- Comparison Table ---
        click.echo(f"{'Rank':<5} {'fsw':>5} {'r_trg':>5} {'Core':<22} {'MOSFET':<22} "
                   f"{'Lpk':>6} {'Leff/L':>6} {'Bmax':>6} {'kw':>5} "
                   f"{'Δr_m':>5} {'risk':>6} {'Rec':<12} "
                   f"{'P_tot':>7} {'eta':>6}")
        click.echo("-" * 145)
        for i, (_, row) in enumerate(top_n.iterrows()):
            L_eff_ratio = row.get("L_eff_ratio", 0)
            margin = row.get("actual_ripple_margin", 0)
            risk = row.get("actual_ripple_risk_level", "?")
            risk_short = {"ok": "ok", "warning": "warn",
                          "high_warning": "HIwrn", "high_risk": "HIrisk"}.get(risk, risk[:6])
            rec = row.get("recommendation_class", "")
            rec_short = (rec[:11] + "…") if len(rec) > 12 else rec
            click.echo(
                f"#{i+1:<4} {row['fsw_kHz']:>4.0f}k {row['ripple_ratio']:>.2f}  "
                f"{row['core']:<22} {row['mosfet']:<22} "
                f"{row['L_eff_uH']:>5.0f}uH {L_eff_ratio:>4.2f}  "
                f"{row['B_max_T']:>5.3f}T {row['kw']*100:>4.0f}% "
                f"{margin:>5.2f} {risk_short:>6} {rec_short:<12} "
                f"{row['P_total_W']:>6.1f}W "
                f"{row['efficiency_pct']:>5.2f}%"
            )

        # Best design detail
        best = top_n.iloc[0]
        click.echo(f"\n+-- Best Design ------------------------------------+")
        click.echo(f"| fsw={best['fsw_kHz']:.0f}kHz  ripple={best['ripple_ratio']:.2f}  "
                   f"eta={best['efficiency_pct']:.2f}%")
        click.echo(f"| Core:  {best['core']} ({best['core_material']})")
        click.echo(f"| MOSFET: {best['mosfet']} ({best['mosfet_tech']}, "
                   f"Rds={best['mosfet_Rds25']:.0f}mOhm)")
        click.echo(f"| N={best['turns']:.0f}  L_eff={best['L_eff_uH']:.0f}uH  "
                   f"Bmax={best['B_max_T']:.3f}T  sat={best['sat_pct']:.0f}%")
        click.echo(f"| Wire: {best['wire_d_mm']:.1f}mm x{best['n_parallel']:.0f}  "
                   f"kw={best['kw']*100:.1f}%")
        click.echo(f"| P_ind={best['P_ind_W']:.1f}W  P_mos={best['P_mosfet_W']:.1f}W  "
                   f"P_dio={best['P_diode_W']:.1f}W  P_br={best['P_bridge_W']:.1f}W")
        click.echo(f"| Inductor: core={best['P_ind_core_W']:.1f}W  copper={best['P_ind_copper_W']:.1f}W")
        click.echo(f"| MOSFET: cond={best['P_mosfet_cond_W']:.1f}W  sw={best['P_mosfet_sw_W']:.1f}W  "
                   f"Coss={best['P_mosfet_coss_W']:.1f}W  gate={best['P_mosfet_gate_W']:.1f}W")
        click.echo(f"| Diode: fwd={best['P_diode_fwd_W']:.1f}W  rr={best['P_diode_rr_W']:.1f}W  "
                   f"Cap={best['P_cap_W']:.1f}W")
        click.echo(f"+---------------------------------------------------+")

        # --- Diversity ---
        if diverse:
            _print_diversity(df)

        # --- Efficiency Plateau Plot ---
        _plot_efficiency_plateau(feasible)

    if refine or heuristic:
        _run_heuristic_stage(df, spec, cores, mosfets, db, n_cores,
                             refine, heuristic)

    _print_infeasible_summary(df)


@cli.command()
@click.option("--vin", default=176.0, help="Design-point line voltage (Vrms, worst case)")
@click.option("--vin-list", default="176,220,264", help="Line voltages for the map (Vrms, comma-separated)")
@click.option("--load-list", default="10,25,50,75,100", help="Load points, % of Pout (comma-separated)")
@click.option("--vout", default=410.0, help="Output voltage (Vdc)")
@click.option("--pout", default=7100.0, help="Rated output power (W)")
@click.option("--fsw", default=65000.0, help="Switching frequency (Hz)")
@click.option("--ripple", default=0.3, help="Current ripple ratio")
@click.option("--core", default=None, help="Core part number")
@click.option("--mosfet", default=None, help="MOSFET part number")
@click.option("--diode", default=None, help="Boost diode part number")
@click.option("--n-cores", default=2, help="Number of stacked cores")
@click.option("--thermal", is_flag=True, default=False,
              help="Self-consistent loss<->temperature loop")
@click.option("--csv", default=None, help="Save map to CSV")
@click.option("--plot/--no-plot", default=True, help="Save efficiency curves PNG")
def map(vin, vin_list, load_list, vout, pout, fsw, ripple, core, mosfet,
        diode, n_cores, thermal, csv, plot):
    """Efficiency map: line voltage x load, fixed hardware (rated-point design)."""
    from .models.system import SystemAnalyzer
    from .core.operating_point import OperatingPoint

    vins = [float(x) for x in vin_list.split(",")]
    loads = [float(x) for x in load_list.split(",")]

    spec = DesignSpec(vin_min=vin, vout=vout, pout_total=pout, fsw=fsw,
                      ripple_ratio=ripple, thermal_model=thermal)
    if diode:
        spec.diode_part = diode   # None -> keep the spec default (SiC part)
    db = CoreDatabase()
    mdb = MosfetDatabase()
    analyzer = SystemAnalyzer(db)
    preferred_core = db.get_by_part_number(core) if core else None
    preferred_mosfet = mdb.get(mosfet) if mosfet else None

    # Fixed hardware: design once at the rated (worst-case) point
    base_op = compute_mathcad_operating_point(spec)
    base = analyzer.analyze(spec, base_op, preferred_core=preferred_core,
                            mosfet=preferred_mosfet, n_cores=n_cores)
    design = base["inductor_design"]
    click.echo(f"\nHardware fixed at design point: {spec.vin_min:.0f}V / "
               f"{spec.pout_total/1000:.1f}kW / fsw={spec.fsw/1000:.0f}kHz / "
               f"ripple={spec.ripple_ratio:.2f}")
    click.echo(f"  Core {design.core.part_number} x{design.n_cores}  N={design.n_turns}  "
               f"L_eff={design.L_eff_at_ipeak_uh:.0f}uH  |  "
               f"{base['mosfet'].part_number}  |  "
               f"{base['diode'].part_number if base['diode'] else spec.diode_type}")
    click.echo("  (CCM model — below ~20% load the real controller goes "
               "burst/DCM, numbers optimistic)\n")

    rows = []
    for vin_map in vins:
        for load_pct in loads:
            s = spec.clone()
            s.pout_total = pout * load_pct / 100.0
            op = OperatingPoint(s, vin_rms=vin_map)
            r = analyzer.analyze(s, op, preferred_core=preferred_core,
                                 mosfet=preferred_mosfet, n_cores=n_cores,
                                 design=design)
            th = r["thermal"]
            rows.append({
                "vin_V": vin_map, "load_pct": load_pct,
                "pout_W": s.pout_total,
                "efficiency_pct": r["efficiency"] * 100,
                "total_loss_W": r["total_loss"],
                "P_inductor_W": r["breakdown"].get("Inductor_core_total", 0)
                                + r["breakdown"].get("Inductor_copper_lf_total", 0)
                                + r["breakdown"].get("Inductor_copper_hf_total", 0)
                                + r["breakdown"].get("Inductor_copper_total", 0),
                "P_mosfet_W": r["breakdown"].get("MOSFET_conduction_total", 0)
                              + r["breakdown"].get("MOSFET_switching_total", 0)
                              + r["breakdown"].get("MOSFET_Coss_total", 0)
                              + r["breakdown"].get("MOSFET_gate_drive_total", 0),
                "P_diode_W": r["breakdown"].get("Diode_forward_total", 0)
                             + r["breakdown"].get("Diode_reverse_recovery_total", 0),
                "P_bridge_W": r["breakdown"].get("Bridge_forward_Vf_total", 0),
                "P_cap_W": r["breakdown"].get("Capacitor_ESR_total", 0),
                "Tj_mosfet_C": th["t_j_mosfet"] if th["enabled"] else 80.0,
            })

    click.echo(f"{'Vin':>5} {'Load':>5} {'Pout':>6} {'eta':>6} "
               f"{'Loss':>6} {'P_ind':>5} {'P_mos':>5} {'P_dio':>5} "
               f"{'P_br':>5} {'P_cap':>5} {'Tj_mos':>6}")
    click.echo("-" * 76)
    for row in rows:
        click.echo(f"{row['vin_V']:>4.0f}V {row['load_pct']:>4.0f}% "
                   f"{row['pout_W']/1000:>5.1f}k {row['efficiency_pct']:>5.2f}% "
                   f"{row['total_loss_W']:>5.1f}W {row['P_inductor_W']:>4.1f}W "
                   f"{row['P_mosfet_W']:>4.1f}W {row['P_diode_W']:>4.1f}W "
                   f"{row['P_bridge_W']:>4.1f}W {row['P_cap_W']:>4.1f}W "
                   f"{row['Tj_mosfet_C']:>5.0f}C")

    if csv:
        import pandas as pd
        csv_dir = os.path.dirname(os.path.abspath(csv))
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        pd.DataFrame(rows).to_csv(csv, index=False)
        click.echo(f"\nMap CSV saved: {csv}")

    if plot:
        _plot_eff_map(rows, vins)


def _plot_eff_map(rows, vins):
    """Efficiency vs load curves, one line per input voltage."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for vin in vins:
        sub = [r for r in rows if r["vin_V"] == vin]
        x = [r["load_pct"] for r in sub]
        y = [r["efficiency_pct"] for r in sub]
        ax.plot(x, y, "o-", label=f"{vin:.0f}V")
    ax.set_xlabel("Load (% of rated)")
    ax.set_ylabel("Efficiency (%)")
    ax.set_title("PFC Efficiency Map (fixed hardware)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_dir = os.path.join(os.path.expanduser("~"), "pfc_design", "output")
    os.makedirs(save_dir, exist_ok=True)
    f = os.path.join(save_dir, "efficiency_map.png")
    plt.tight_layout()
    plt.savefig(f, dpi=150, bbox_inches="tight")
    plt.close(fig)
    click.echo(f"Efficiency map plot saved: {f}")


@cli.command()
@click.option("--tech", default="", help="Technology filter (SiC Schottky/Si Fast Recovery)")
def diodes(tech):
    """List available boost diodes."""
    from .core.spec import DiodeDatabase
    results = DiodeDatabase().query(technology=tech)
    click.echo(f"\n{'Part Number':<22} {'Tech':<16} {'Vrr':>5} {'If25':>5} "
               f"{'Vf25':>5} {'Vf150':>5} {'Qrr':>6} {'Price':>6}")
    click.echo("-" * 76)
    for d in results:
        click.echo(f"{d.part_number:<22} {d.technology:<16} {d.vrr_max:>4.0f}V "
                   f"{d.if_25c:>4.0f}A {d.vf_25c:>5.2f}V {d.vf_150c:>5.2f}V "
                   f"{d.qrr_nc:>5.0f}nC ${d.price_usd:>5.1f}")


@cli.command()
@click.option("--vin", default=176.0, help="Input voltage (Vrms)")
@click.option("--vout", default=410.0, help="Output voltage (Vdc)")
@click.option("--pout", default=7100.0, help="Total output power (W)")
@click.option("--fsw", default=65000.0, help="Switching frequency (Hz)")
@click.option("--ripple", default=0.3, help="Current ripple ratio")
@click.option("--core", default=None, help="Preferred core part number")
@click.option("--mosfet", default=None, help="Boost MOSFET (bridge PFC baseline)")
@click.option("--diode", default="C4D10120H", help="Boost diode (bridge PFC baseline)")
@click.option("--hf-mosfet", default=None, help="Totem HF leg device (default GS66516T GaN)")
@click.option("--slow-mosfet", default="IPW65R032M8", help="Totem line-frequency leg device")
@click.option("--deadtime", default=100.0, help="SR deadtime (ns)")
@click.option("--vfb", default=2.0, help="Body/reverse conduction drop (V)")
@click.option("--n-cores", default=2, help="Number of stacked cores")
@click.option("--thermal", is_flag=True, default=False,
              help="Self-consistent loss<->temperature loop")
@click.option("--plot/--no-plot", default=False, help="Save loss comparison bar chart")
def totem(vin, vout, pout, fsw, ripple, core, mosfet, diode, hf_mosfet,
          slow_mosfet, deadtime, vfb, n_cores, thermal, plot):
    """Compare bridge boost vs totem-pole bridgeless PFC (same magnetics)."""
    from .models.system import SystemAnalyzer
    from .models.totem_pole import TotemPoleAnalyzer

    spec = DesignSpec(vin_min=vin, vout=vout, pout_total=pout, fsw=fsw,
                      ripple_ratio=ripple, diode_part=diode,
                      thermal_model=thermal)
    db = CoreDatabase()
    mdb = MosfetDatabase()
    preferred_core = db.get_by_part_number(core) if core else None
    preferred_mosfet = mdb.get(mosfet) if mosfet else None

    click.echo(f"\n{'='*78}")
    click.echo(f"  Topology comparison — bridge boost vs totem-pole bridgeless")
    click.echo(f"  {vin:.0f}V -> {vout:.0f}V | {pout/1000:.1f}kW | fsw={fsw/1000:.0f}kHz "
               f"| ripple={ripple:.2f} | n_phases={spec.n_phases}")
    click.echo(f"{'='*78}")

    boost = SystemAnalyzer(db).analyze(spec, preferred_core=preferred_core,
                                       mosfet=preferred_mosfet, n_cores=n_cores)
    totem_r = TotemPoleAnalyzer(db).analyze(
        spec, preferred_core=preferred_core, mosfet=mdb.get(hf_mosfet)
        if hf_mosfet else None, slow_mosfet=mdb.get(slow_mosfet),
        n_cores=n_cores, deadtime_s=deadtime * 1e-9, vf_body=vfb)

    design = boost["inductor_design"]
    click.echo(f"  Inductor (identical in both): {design.core.part_number} "
               f"x{design.n_cores}  N={design.n_turns}  "
               f"L_eff={design.L_eff_at_ipeak_uh:.0f}uH")
    click.echo(f"  Boost: {boost['mosfet'].part_number} + {boost['diode'].part_number if boost['diode'] else spec.diode_type}")
    click.echo(f"  Totem: HF={totem_r['mosfet'].part_number} "
               f"(GaN), slow={totem_r['slow_mosfet'].part_number} "
               f"| deadtime={deadtime:.0f}ns, Vfb={vfb:.1f}V\n")

    # Build per-component comparison (per-phase entries already scaled by n)
    def _total(result, prefixes):
        return sum(v for k, v in result["breakdown"].items()
                   if any(k.startswith(p) for p in prefixes))

    rows = [
        ("Inductor core", "Inductor_core_",
         _total(boost, ["Inductor_core_"]),
         _total(totem_r, ["Inductor_core_"])),
        ("Inductor copper", "Inductor_copper_",
         _total(boost, ["Inductor_copper_"]),
         _total(totem_r, ["Inductor_copper_"])),
        ("MOSFET conduction+switching",
         "MOSFET_conduction_|MOSFET_switching_|HF_active_conduction_|HF_active_switching_|HF_SR_conduction_|HF_SR_switch_off_|HF_SR_deadtime_|HF_SR_Coss_|Slow_leg_conduction_",
         _total(boost, ["MOSFET_conduction_", "MOSFET_switching_"]),
         _total(totem_r, ["HF_active_conduction_", "HF_active_switching_",
                          "HF_SR_conduction_", "HF_SR_switch_off_",
                          "HF_SR_deadtime_", "HF_SR_Coss_",
                          "Slow_leg_conduction_"])),
        ("Coss + gate drive",
         "MOSFET_Coss_|MOSFET_gate_|HF_active_Coss_|HF_active_gate_|HF_SR_gate_|Slow_leg_gate_",
         _total(boost, ["MOSFET_Coss_", "MOSFET_gate_"]),
         _total(totem_r, ["HF_active_Coss_", "HF_active_gate_",
                          "HF_SR_gate_", "Slow_leg_gate_"])),
        ("Diode (fwd + RR)", "Diode_",
         _total(boost, ["Diode_"]),
         0.0),
        ("Bridge rectifier", "Bridge_",
         _total(boost, ["Bridge_"]),
         0.0),
        ("Capacitor ESR", "Capacitor_",
         _total(boost, ["Capacitor_"]),
         _total(totem_r, ["Capacitor_"])),
    ]

    click.echo(f"{'Component':<34} {'Boost(W)':>9} {'Totem(W)':>9} {'Delta(W)':>9}")
    click.echo("-" * 62)
    for label, _, b, t in rows:
        click.echo(f"{label:<34} {b:>8.1f}W {t:>8.1f}W {t-b:>8.1f}")
    click.echo("-" * 62)
    click.echo(f"{'TOTAL':<34} {boost['total_loss']:>8.1f}W "
               f"{totem_r['total_loss']:>8.1f}W "
               f"{totem_r['total_loss']-boost['total_loss']:>8.1f}")
    saved = boost["total_loss"] - totem_r["total_loss"]
    click.echo(f"\nEfficiency:  boost={boost['efficiency']*100:.2f}%  "
               f"totem={totem_r['efficiency']*100:.2f}%  "
               f"(+{(totem_r['efficiency']-boost['efficiency'])*100:.2f} pt, "
               f"{saved:.1f} W saved)")

    th = totem_r["thermal"]
    if th["enabled"]:
        click.echo(f"Totem thermal: Tj_hf={th['t_j_hf']:.0f}C  "
                   f"Tj_slow={th['t_j_slow']:.0f}C  "
                   f"T_wind={th['t_winding']:.0f}C")

    if plot:
        _plot_topology_compare(rows, boost, totem_r)


def _plot_topology_compare(rows, boost, totem_r):
    """Grouped bar chart: boost vs totem per-component losses."""
    labels = [r[0] for r in rows]
    b_vals = [r[2] for r in rows]
    t_vals = [r[3] for r in rows]
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - w / 2, b_vals, w, label=f"Bridge boost ({boost['efficiency']*100:.2f}%)",
           color="#d1495b")
    ax.bar(x + w / 2, t_vals, w, label=f"Totem-pole ({totem_r['efficiency']*100:.2f}%)",
           color="#2e86ab")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Loss (W)")
    ax.set_title(f"Topology comparison — "
                 f"{boost['spec'].pout_total/1000:.1f}kW, "
                 f"fsw={boost['spec'].fsw/1000:.0f}kHz")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    save_dir = os.path.join(os.path.expanduser("~"), "pfc_design", "output")
    os.makedirs(save_dir, exist_ok=True)
    f = os.path.join(save_dir, "topology_compare.png")
    plt.tight_layout()
    plt.savefig(f, dpi=150, bbox_inches="tight")
    plt.close(fig)
    click.echo(f"Comparison chart saved: {f}")


@cli.command()
@click.option("--material", default="Sendust", help="Material class")
def cores(material):
    """List available magnetic cores."""
    db = CoreDatabase()
    results = db.query(material_class=material, max_results=50)
    click.echo(f"\n{'Part Number':<25} {'Material':<20} {'OD':>5} {'Ae':>5} {'AL':>6} {'Aw':>5}")
    click.echo("-" * 72)
    for c in results:
        click.echo(f"{c.part_number:<25} {c.material:<20} {c.od_mm:>4.1f}mm "
                   f"{c.ae_cm2:>4.2f}cm {c.al_nH_per_t2:>5.0f}  {c.aw_cm2:>4.2f}cm")


@cli.command()
@click.option("--tech", default="Si", help="Technology filter (Si/SiC/Si CoolMOS/Si SuperJunction/GaN/all)")
def mosfets(tech):
    """List available MOSFETs."""
    mdb = MosfetDatabase()
    results = mdb.query(technology=tech)
    click.echo(f"\n{'Part Number':<22} {'Tech':<16} {'Vds':>5} {'Id25':>5} {'Rds':>6} {'Qg':>5} {'Price':>6}")
    click.echo("-" * 72)
    for m in results:
        click.echo(f"{m.part_number:<22} {m.technology:<16} {m.vds_max:>4.0f}V {m.id_25c:>4.0f}A "
                   f"{m.rds_on_25c*1000:>4.0f}mΩ {m.qg_nc:>4.0f}nC ${m.price_usd:>5.1f}")


@cli.command()
def version():
    """Show version."""
    from . import __version__
    click.echo(f"pfc_design v{__version__}")


# ─── Output helpers ────────────────────────────────────────────

def _print_summary(result):
    spec = result["spec"]
    design = result["inductor_design"]
    mos = result["mosfet"]
    dm = design.design_metadata
    click.echo(f"\n=== PFC Design Summary ===")
    click.echo(f"{spec.vin_min}-{spec.vin_max}V → {spec.vout}V | "
               f"{spec.pout_total/1000:.1f}kW | fsw={spec.fsw/1000:.0f}kHz | ripple r={spec.ripple_ratio:.2f}")
    click.echo(f"Core: {design.core.part_number} x{design.n_cores} | "
               f"N={design.n_turns} | L_eff={design.L_eff_at_ipeak_uh:.0f}uH")
    im = result.get("inductor_metrics", {})
    if im:
        ratio = im.get("L_eff_ratio", 0)
        margin = im.get("actual_ripple_margin", 0)
        risk = im.get("actual_ripple_risk_level", "?")
        click.echo(f"L design: L_target={im.get('L_target_uH', 0):.0f}uH  "
                   f"L_eff={im.get('L_eff_at_ipeak_uH', 0):.0f}uH  "
                   f"L_eff/L={ratio:.2f}  (trace用L_eff)")
        click.echo(f"Ripple:   target r={im.get('target_ripple_ratio_peak_basis', 0):.3f}  "
                   f"actual r={im.get('actual_ripple_ratio_peak_basis', 0):.3f}  "
                   f"margin={margin:.2f}  risk={risk}")
        click.echo(f"          (ΔIpp_ref={im.get('delta_i_pp_ref_A', 0):.2f}A  "
                   f"ΔIpp_actual={im.get('delta_i_pp_actual_A', 0):.2f}A)")
    click.echo(f"MOSFET: {mos.part_number} ({mos.technology}) | "
               f"Rds25={mos.rds_on_25c*1000:.0f}mΩ | Vds={mos.vds_max:.0f}V")
    dio = result.get("diode")
    if dio is not None:
        th = result.get("thermal", {})
        t_j_dio = th.get("t_j_diode", 80.0)
        click.echo(f"Diode:  {dio.part_number} ({dio.technology}) | "
                   f"Vf({t_j_dio:.0f}C)={dio.vf_at(t_j_dio):.2f}V | "
                   f"Vrr={dio.vrr_max:.0f}V")
    wire_d = design.wire_d_mm
    wire_n = design.n_parallel
    click.echo(f"Wire: {wire_d}mm x{wire_n} | Window fill: {design.kw*100:.1f}%")

    th = result.get("thermal", {})
    if th.get("enabled"):
        click.echo(f"Thermal: Tj_mos={th['t_j_mosfet']:.0f}C  "
                   f"Tj_dio={th['t_j_diode']:.0f}C  "
                   f"T_wind={th['t_winding']:.0f}C  "
                   f"(Rth_mos={th['rth_mosfet_ja']:.2f} K/W)")


def _print_loss_table(result):
    click.echo(f"\n{'Component':<35} {'Loss (W)':>8}  {'%':>6}")
    click.echo("-" * 52)
    total = result["total_loss"]
    for name, val in result["breakdown"].items():
        if val > 0.3:
            # Clean up the key for display
            label = (name.replace("_total", " (total)")
                     .replace("_", " "))
            click.echo(f"{label:<35} {val:>7.1f}W {val/total*100:>5.1f}%")
    click.echo("-" * 52)
    click.echo(f"{'TOTAL':<35} {total:>7.1f}W {'100.0%':>6}")
    click.echo(f"{'Efficiency':<35} {result['efficiency']*100:>7.2f}%")


def _generate_plots(result, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    spec = result["spec"]
    op = result["op"]
    design = result["inductor_design"]
    core = design.core

    # --- Loss Pie Chart ---
    fig, ax = plt.subplots(figsize=(10, 7))
    items = {k: v for k, v in result["breakdown"].items() if v > result["total_loss"] * 0.005}
    colors = plt.cm.Set3(np.linspace(0, 1, len(items)))
    wedges, _, autotexts = ax.pie(
        list(items.values()), labels=None, autopct='%1.1f%%',
        startangle=90, pctdistance=0.82, colors=colors,
        wedgeprops=dict(width=0.35, edgecolor='w')
    )
    for t in autotexts:
        t.set_fontsize(9)
        t.set_fontweight('bold')
    legend_labels = [f"{l}: {s:.1f}W" for l, s in items.items()]
    ax.legend(wedges, legend_labels, title="Components",
              loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=8)
    ax.text(0, 0, f"η={result['efficiency']*100:.1f}%\n{result['total_loss']:.1f}W",
            ha='center', va='center', fontsize=13, fontweight='bold')
    ax.set_title("PFC Loss Breakdown", fontsize=14, fontweight='bold')
    plt.tight_layout()
    f1 = os.path.join(save_dir, "loss_pie.png")
    plt.savefig(f1, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- L vs I Saturation ---
    from .magnetics.saturation import generate_L_vs_I
    i_arr, L_arr = generate_L_vs_I(
        design.L_noload_uh, design.n_turns, core.le_cm,
        core.dc_bias_coeffs, max(50, op.iin_peak * 2)
    )
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(i_arr, L_arr, 'b-', linewidth=2)
    ax2.axvline(x=op.iin_rms, color='g', linestyle='--', label=f'Irms={op.iin_rms:.1f}A')
    ax2.axvline(x=op.iin_peak, color='r', linestyle='--', label=f'Ipeak={op.iin_peak:.1f}A')
    ax2.axhline(y=design.L_target_uh, color='orange', linestyle=':', label=f'L_tgt={design.L_target_uh:.0f}uH')
    ax2.set_xlabel("DC Current (A)")
    ax2.set_ylabel("Inductance (μH)")
    ax2.set_title(f"L vs DC Bias (L₀={design.L_noload_uh:.0f}μH, N={design.n_turns})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    f2 = os.path.join(save_dir, "L_vs_I.png")
    plt.savefig(f2, dpi=150, bbox_inches='tight')
    plt.close(fig2)

    click.echo(f"\nPlots saved: {f1}, {f2}")

    # Auto-open in Windows only (explorer.exe does not exist elsewhere)
    if os.name == 'nt':
        subprocess.run(["explorer.exe", save_dir], capture_output=True)


def _export_loss_trace_csv(result, path):
    """Write loss_trace.csv with per-angle loss densities.

    All field names use _phase suffix for per-phase current/voltage,
    and density fields are per-radian at that theta point.
    """
    trace = result.get("trace")
    if trace is None:
        click.echo("No line-cycle trace available.")
        return

    spec = result["spec"]
    mosfet = result["mosfet"]
    design = result["inductor_design"]

    # Per-angle constants
    rds_tj = mosfet.rds_on_25c * (1 + mosfet.rds_alpha * (80 - 25))
    coss_er = mosfet.coss_er
    tr = mosfet.tr
    tf = mosfet.tf
    vout = spec.vout
    fsw = spec.fsw
    vf = spec.diode_vf
    rd = spec.diode_rd
    qrr = 2e-6 if spec.diode_type.upper() != "SIC" else 0.0
    # Switching energy: datasheet Eon/Eoff curves if available, else tr/tf ramp
    if mosfet.eon_ref_uj > 0 and mosfet.eoff_ref_uj > 0:
        k_on = mosfet.eon_ref_uj * 1e-6 / (mosfet.e_ref_v * mosfet.e_ref_i)
        k_off = mosfet.eoff_ref_uj * 1e-6 / (mosfet.e_ref_v * mosfet.e_ref_i)
        e_on_fn = lambda i: k_on * vout * i
        e_off_fn = lambda i: k_off * vout * i
    else:
        e_on_fn = lambda i: 0.5 * vout * i * tr
        e_off_fn = lambda i: 0.5 * vout * i * tf

    # Winding Rdc / Rac (simplified — approximate, same as loss model)
    from .magnetics.winding import skin_effect_factor, proximity_factor, dc_resistance
    from .core.constants import rho_cu, skin_depth
    core_obj = design.core
    a_wire_mm2 = np.pi * (design.wire_d_mm / 2) ** 2 * design.n_parallel
    a_wire_m2 = a_wire_mm2 * 1e-6
    mlt_m = core_obj.mlt_cm / 100
    total_length_m = design.n_turns * mlt_m
    t_winding = spec.t_ambient + 40
    rho = rho_cu(t_winding)
    rdc_wire = dc_resistance(rho, total_length_m, a_wire_m2)
    d_wire_m = design.wire_d_mm / 1000
    delta_skin = skin_depth(fsw, t_winding)
    f_skin = skin_effect_factor(d_wire_m, delta_skin)
    rac_wire = rdc_wire * f_skin

    # Core loss Steinmetz coefficients
    from .magnetics.core_database import CoreDatabase
    db = CoreDatabase()
    st = db.get_steinmetz(core_obj.material)
    if st is None:
        st = db.get_steinmetz(core_obj.material_class)
    k, alpha, beta = (st.k, st.alpha, st.beta) if st else (0, 0, 0)
    L_eff_H = design.L_eff_at_ipeak_uh * 1e-6
    N_turns = design.n_turns
    Ae_m2 = design.ae_total_cm2 * 1e-4

    rows = []
    for i in range(len(trace.theta)):
        duty_i = trace.duty[i]
        off_duty = 1.0 - duty_i
        delta_i_i = trace.delta_i_pp[i]
        i_rms_sq_i = trace.i_rms_local_sq[i]

        # MOSFET conduction density
        mosfet_cond = rds_tj * duty_i * i_rms_sq_i

        # MOSFET switching energy density
        mosfet_eon = e_on_fn(trace.i_valley[i])
        mosfet_eoff = e_off_fn(trace.i_peak[i])
        # Per-switching-cycle energy → per-radian average over half cycle
        # E per radian = fsw * E_per_switching * (T_half / pi) ≈ E_per_sw / dtheta
        # Actually simpler: the density for this theta bin is fsw * E(θ) / pi
        mosfet_sw_eon_density = fsw * mosfet_eon
        mosfet_sw_eoff_density = fsw * mosfet_eoff

        # Coss density (constant per switching cycle)
        mosfet_coss_density = 0.5 * coss_er * vout ** 2 * fsw

        # Diode forward
        diode_fwd = vf * off_duty * trace.i_avg_phase[i] + rd * off_duty * i_rms_sq_i
        # Diode RR (constant per switching cycle)
        diode_rr = fsw * qrr * vout

        # Core loss density
        if st and N_turns > 0:
            delta_B_pp = L_eff_H * delta_i_i / (N_turns * Ae_m2)
            B_ac_peak = delta_B_pp / 2.0
            core_loss = k * fsw ** alpha * B_ac_peak ** beta if B_ac_peak > 0 else 0.0
        else:
            core_loss = 0.0

        # Winding densities
        winding_lf = trace.i_avg_phase[i] ** 2 * rdc_wire
        winding_hf = (delta_i_i ** 2 / 12.0) * rac_wire

        rows.append({
            "theta_deg": round(trace.theta_deg[i], 4),
            "sin_theta": round(trace.sin_theta[i], 6),
            "vin_abs": round(trace.vin_abs[i], 4),
            "i_avg_phase": round(trace.i_avg_phase[i], 4),
            "duty": round(duty_i, 6),
            "delta_i_pp": round(delta_i_i, 4),
            "i_peak": round(trace.i_peak[i], 4),
            "i_valley": round(trace.i_valley[i], 4),
            "i_rms_local_sq": round(i_rms_sq_i, 4),
            "mosfet_cond_density": round(mosfet_cond, 6),
            "mosfet_eon_density": round(mosfet_sw_eon_density, 6),
            "mosfet_eoff_density": round(mosfet_sw_eoff_density, 6),
            "mosfet_coss_density": round(mosfet_coss_density, 6),
            "diode_forward_density": round(diode_fwd, 6),
            "diode_rr_density": round(diode_rr, 6),
            "core_loss_density": round(core_loss, 6),
            "winding_lf_density": round(winding_lf, 6),
            "winding_hf_density": round(winding_hf, 6),
        })

    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    click.echo(f"Loss trace CSV saved: {path}  ({len(rows)} rows)")


def _plot_efficiency_plateau(df_feasible):
    """Efficiency vs switching frequency scatter, highlighting top designs."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter all feasible points
    sc = ax.scatter(
        df_feasible["fsw_kHz"], df_feasible["efficiency_pct"],
        c=df_feasible["P_total_W"], cmap="RdYlGn_r",
        s=60, alpha=0.6, edgecolors='grey', linewidth=0.3
    )
    cbar = plt.colorbar(sc)
    cbar.set_label("Total Loss (W)")

    # Highlight top 5
    top5 = df_feasible.nsmallest(5, "P_total_W")
    ax.scatter(top5["fsw_kHz"], top5["efficiency_pct"],
               c='red', s=120, marker='*', edgecolors='darkred',
               zorder=10, label=f'Top 5')

    # Labels for top 3
    for _, row in top5.head(3).iterrows():
        ax.annotate(
            f"{row['mosfet'][:12]}\n{row['core'][:12]}",
            (row["fsw_kHz"], row["efficiency_pct"]),
            fontsize=7, ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7)
        )

    ax.set_xlabel("Switching Frequency (kHz)")
    ax.set_ylabel("Efficiency (%)")
    ax.set_title("PFC Efficiency Plateau — fsw × ripple × core × MOSFET sweep")
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    save_dir = os.path.join(os.path.expanduser("~"), "pfc_design", "output")
    os.makedirs(save_dir, exist_ok=True)
    f = os.path.join(save_dir, "efficiency_plateau.png")
    plt.tight_layout()
    plt.savefig(f, dpi=150, bbox_inches='tight')
    plt.close(fig)
    click.echo(f"Efficiency plateau plot saved: {f}")


def _print_recommendations(recommendations):
    """Print the four recommendation classes."""
    rec_specs = [
        ("lowest_loss_feasible", "Lowest Loss Feasible",
         "None — no design passes base constraints"),
        ("recommended_robust", "Recommended (Robust / Production)",
         "None — no design meets production constraints"),
        ("recommended_balanced", "Recommended (Balanced)",
         "None — no design meets balanced constraints"),
        ("aggressive_low_loss", "Aggressive Low-Loss",
         "None — no design meets aggressive constraints"),
    ]
    click.echo("--- Recommendations ---")
    for key, label, none_msg in rec_specs:
        row = recommendations.get(key)
        if row is not None:
            risk = row.get("actual_ripple_risk_level", "?")
            risk_short = {"ok": "ok", "warning": "warn",
                          "high_warning": "HIwrn", "high_risk": "HIrisk"}.get(risk, risk[:6])
            rec = row.get("recommendation_class", "")
            click.echo(
                f"  [{label}]  "
                f"core={row['core']}  mosfet={row['mosfet']}  "
                f"fsw={row['fsw_kHz']:.0f}kHz  "
                f"eta={row['efficiency_pct']:.2f}%  "
                f"P_total={row['P_total_W']:.1f}W  "
                f"ripple_margin={row['actual_ripple_margin']:.2f} "
                f"({risk_short})  "
                f"class={rec}"
            )
        else:
            click.echo(f"  [{label}]  {none_msg}")
    click.echo()


def _print_diversity(df):
    """Print per-core and per-MOSFET best feasible candidates."""
    core_best = per_core_best(df)
    mos_best = per_mosfet_best(df)

    if len(core_best) > 0:
        click.echo("--- Per-Core Best ---")
        for _, row in core_best.iterrows():
            risk = row.get("actual_ripple_risk_level", "?")
            risk_short = {"ok": "ok", "warning": "warn",
                          "high_warning": "HIwrn", "high_risk": "HIrisk"}.get(risk, risk[:6])
            click.echo(
                f"  core={row['core']:<22}  "
                f"mosfet={row['mosfet']:<22}  "
                f"fsw={row['fsw_kHz']:.0f}kHz  "
                f"P_total={row['P_total_W']:.1f}W  "
                f"eta={row['efficiency_pct']:.2f}%  "
                f"risk={risk_short}"
            )
        click.echo()

    if len(mos_best) > 0:
        click.echo("--- Per-MOSFET Best ---")
        for _, row in mos_best.iterrows():
            risk = row.get("actual_ripple_risk_level", "?")
            risk_short = {"ok": "ok", "warning": "warn",
                          "high_warning": "HIwrn", "high_risk": "HIrisk"}.get(risk, risk[:6])
            click.echo(
                f"  mosfet={row['mosfet']:<22}  "
                f"core={row['core']:<22}  "
                f"fsw={row['fsw_kHz']:.0f}kHz  "
                f"P_total={row['P_total_W']:.1f}W  "
                f"eta={row['efficiency_pct']:.2f}%  "
                f"risk={risk_short}"
            )
        click.echo()


def _print_infeasible_summary(df):
    """Print infeasible-reason breakdown."""
    summary = infeasible_reason_summary(df)
    failed = df["constraints"].value_counts().head(5)
    feasible_count = df["feasible"].sum()
    if feasible_count == 0:
        click.echo("No feasible design found.")
    click.echo("\n--- Infeasible Reason Summary ---")
    click.echo(f"  failed_bmax:            {summary['failed_bmax_count']}")
    click.echo(f"  failed_window:          {summary['failed_window_count']}")
    click.echo(f"  failed_saturation:      {summary['failed_saturation_count']}")
    click.echo(f"  failed_actual_ripple:   {summary['failed_actual_ripple_count']}")
    click.echo(f"  failed_l_eff_ratio:     {summary['failed_l_eff_ratio_count']}")
    click.echo("\nTop constraint strings:")
    for reason, count in failed.items():
        click.echo(f"  {count:>4}  {reason}")


def _run_heuristic_stage(df, spec, cores, mosfets, db, n_cores,
                         do_refine, do_heuristic):
    """Post-grid refinement: local polish of top grid designs + DE global search."""
    from .optimization.scipy_opt import refine_design
    from .optimization.heuristic import global_search
    from .models.system import SystemAnalyzer

    analyzer = SystemAnalyzer(db)
    mdb = MosfetDatabase()
    rows = []

    feasible = df[df["feasible"]]
    grid_best_loss = (float(feasible["P_total_W"].min())
                      if len(feasible) > 0 else None)

    if do_refine and len(feasible) > 0:
        click.echo("\n--- Local Refine (Nelder-Mead on fsw x ripple) ---")
        top_grid = feasible.nsmallest(min(5, len(feasible)), "P_total_W")
        for _, row in top_grid.iterrows():
            core = db.get_by_part_number(row["core"])
            mosfet = mdb.get(row["mosfet"])
            if core is None or mosfet is None:
                continue
            r = refine_design(analyzer, spec, core, mosfet,
                              x0=(row["fsw_kHz"], row["ripple_ratio"]),
                              n_cores=n_cores)
            r["origin"] = f"refine (grid seed fsw={row['fsw_kHz']:.0f}k)"
            rows.append(r)

    if do_heuristic:
        click.echo("\n--- Global Search (differential evolution) ---")
        click.echo(f"  searching fsw x ripple x {len(cores)} cores x {len(mosfets)} MOSFETs...")
        r = global_search(analyzer, spec, cores, mosfets, n_cores=n_cores)
        r["origin"] = "global search (DE)"
        rows.append(r)

    if not rows:
        return

    click.echo(f"\n{'Stage':<34} {'fsw':>5} {'r':>4} {'Core':<18} {'MOSFET':<18} "
               f"{'P_tot':>7} {'eta':>6} {'ok?':>4} {'Δr_m':>5} {'evals':>6}")
    click.echo("-" * 112)
    for r in rows:
        risk = "yes" if r["feasible"] else "no"
        click.echo(
            f"{r['origin']:<34} {r['fsw_kHz']:>4.1f}k {r['ripple_ratio']:>4.2f} "
            f"{r['core']:<18} {r['mosfet']:<18} "
            f"{r['loss_W']:>6.1f}W {r['efficiency_pct']:>5.2f}% "
            f"{risk:>4} {r['actual_ripple_margin']:>5.2f} {r['evaluations']:>6}"
        )

    feasible_heu = [r for r in rows if r["feasible"]]
    if feasible_heu:
        best = min(feasible_heu, key=lambda r: r["loss_W"])
        click.echo("\n--- Heuristic improvement vs grid ---")
        click.echo(f"  best heuristic: {best['loss_W']:.1f}W @ "
                   f"{best['fsw_kHz']:.1f}kHz  ({best['core']} + {best['mosfet']})")
        if grid_best_loss is not None:
            delta = grid_best_loss - best["loss_W"]
            click.echo(f"  best grid:     {grid_best_loss:.1f}W")
            if delta > 1e-9:
                click.echo(f"  improvement:   {delta:.1f}W saved "
                           f"({delta/grid_best_loss*100:.1f}%)")
            else:
                click.echo("  improvement:   none (grid best already optimal in region)")
    else:
        click.echo("\nNo feasible design found by heuristic stage.")




if __name__ == "__main__":
    cli()
