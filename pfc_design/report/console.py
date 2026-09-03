"""Rich console table output for PFC design results."""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel


def design_summary(result: dict) -> Table:
    """Generate a Rich table with design summary."""
    spec = result["spec"]
    design = result["inductor_design"]

    table = Table(title="PFC Design Summary", title_style="bold cyan")
    table.add_column("Parameter", style="dim")
    table.add_column("Value", justify="right")

    table.add_row("Input Voltage", f"{spec.vin_min}-{spec.vin_max} Vrms")
    table.add_row("Output Voltage", f"{spec.vout} Vdc")
    table.add_row("Output Power", f"{spec.pout_total/1000:.1f} kW")
    table.add_row("Phases", str(spec.n_phases))
    table.add_row("Switching Freq", f"{spec.fsw/1000:.0f} kHz")
    table.add_row("Ripple Ratio", f"{spec.ripple_ratio:.2f}")

    table.add_section()
    table.add_row("Core", design.core.part_number)
    table.add_row("Core Material", design.core.material)
    table.add_row("Stacked Cores", str(design.n_cores))
    table.add_row("Turns", str(design.n_turns))
    table.add_row("L_target", f"{design.L_target_uh:.1f} μH")
    table.add_row("L_noload", f"{design.L_noload_uh:.1f} μH")
    table.add_row("L_eff", f"{design.L_eff_at_ipeak_uh:.1f} μH")
    table.add_row("Wire", f"{design.wire_d_mm}mm × {design.n_parallel}")
    table.add_row("Window Fill", f"{design.kw*100:.1f}%")

    return table


def loss_table(result: dict) -> Table:
    """Generate a Rich table with detailed loss breakdown."""
    table = Table(title="Loss Breakdown", title_style="bold red")
    table.add_column("Component")
    table.add_column("Sub-type")
    table.add_column("Loss (W)", justify="right")
    table.add_column("%", justify="right")

    total_loss = result["total_loss"]
    breakdown = result["breakdown"]

    for name, val in breakdown.items():
        if val > 0.01:
            if " " in name:
                comp, sub = name.split(" ", 1)
            else:
                comp, sub = name, "-"
            pct = val / total_loss * 100 if total_loss > 0 else 0
            table.add_row(comp, sub, f"{val:.2f}", f"{pct:.1f}%")

    table.add_section()
    table.add_row("TOTAL", "", f"{total_loss:.2f}", "100%")
    table.add_row("Efficiency", "", f"{result['efficiency']*100:.2f}%", "")

    return table


def print_full_report(result: dict):
    """Print a complete design report to console."""
    console = Console()
    console.print(design_summary(result))
    console.print()
    console.print(loss_table(result))

    # Capacitor info
    cap_meta = result["losses"]["capacitor"].metadata
    console.print()
    cap_panel = Panel(
        f"ΔV_lf: {cap_meta['delta_V_LF_Vpp']:.1f}Vpp\n"
        f"ΔV_sw: {cap_meta['delta_V_SW_Vpp']*1000:.1f}mVpp\n"
        f"Ic_rms: {cap_meta['Ic_rms_A']:.1f}A\n"
        f"Lifetime: {cap_meta['lifetime_hours']:.0f}h ({cap_meta['lifetime_years']:.1f} years)",
        title="Output Capacitor",
        border_style="green"
    )
    console.print(cap_panel)
