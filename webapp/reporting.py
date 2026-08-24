"""Report generation for the LLC web tool.

PDF (fpdf2 + matplotlib Agg) and Excel (pandas + openpyxl) exports are built
exclusively from the analyze_llc() result so the report always mirrors what the
browser shows.  The PDF is English-only on purpose: the bundled fpdf2 core
fonts are Latin-1, and engineering section names follow the product spec.
"""
from __future__ import annotations

import io
import os
from datetime import date
from pathlib import Path
from typing import Any

from fpdf import FPDF  # noqa: E402

from .service import analyze_llc

# Matplotlib is imported lazily (only when a chart is actually rendered) so the
# API server boots in ~1 s instead of paying the font-cache build on startup.
# The cache lives in a project-local writable directory so it is built once and
# reused on later starts.
_MPL_CACHE_DIR = Path(__file__).resolve().parent / ".mplcache"
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt

_LOSS_LABELS: dict[str, str] = {
    "primary_conduction_w": "Primary conduction",
    "primary_turnoff_w": "Primary turn-off",
    "primary_gate_w": "Primary gate",
    "primary_coss_w": "Primary Coss",
    "primary_deadtime_w": "Primary deadtime",
    "sr_conduction_w": "SR conduction",
    "sr_deadtime_w": "SR deadtime",
    "sr_turnoff_w": "SR turn-off",
    "sr_coss_w": "SR Coss",
    "sr_gate_w": "SR gate",
    "transformer_core_w": "Transformer core",
    "transformer_primary_copper_w": "Transformer pri Cu",
    "transformer_secondary_copper_w": "Transformer sec Cu",
    "resonant_inductor_core_w": "Lr core",
    "resonant_inductor_copper_w": "Lr Cu",
    "resonant_capacitor_w": "Resonant cap",
    "output_capacitor_w": "Output cap",
    "auxiliary_w": "Auxiliary",
}


# ---------------------------------------------------------------------------
# Chart helpers (light, print-friendly)
# ---------------------------------------------------------------------------

def _gain_map_png(result: dict[str, Any]) -> bytes:
    plt = _matplotlib()
    gain_map = result["gain_map"]
    fig, ax = plt.subplots(figsize=(7.4, 3.4), dpi=110)
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756"]
    for index, curve in enumerate(gain_map["curves"]):
        color = colors[index % len(colors)]
        ax.plot(
            [f / 1000 for f in curve["frequency_hz"]],
            curve["gain"],
            color=color,
            linewidth=1.4,
            label=f"{int(round(curve['load_fraction'] * 100))}% load",
        )
    for target in gain_map["targets"]:
        ax.axhline(target["gain"], color="#8a8f98", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.axvline(gain_map["fr_hz"] / 1000, color="#d62728", linewidth=0.8, linestyle=":", alpha=0.8)
    ax.text(gain_map["fr_hz"] / 1000, ax.get_ylim()[1] * 0.98, " fr", color="#d62728", fontsize=8)
    ax.set_xlabel("Switching frequency / kHz")
    ax.set_ylabel("Normalized gain")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=3, frameon=False, loc="upper right")
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    return buffer.getvalue()


def _loss_pie_png(result: dict[str, Any]) -> bytes:
    plt = _matplotlib()
    breakdown = {key: value for key, value in result["nominal_loss_breakdown"].items() if value > 0.01}
    items = sorted(breakdown.items(), key=lambda kv: kv[1], reverse=True)
    labels = [_LOSS_LABELS.get(key, key) for key, _ in items]
    values = [value for _, value in items]
    if len(items) > 6:
        head, tail = items[:5], items[5:]
        labels = [_LOSS_LABELS.get(key, key) for key, _ in head] + ["Other"]
        values = [value for _, value in head] + [sum(v for _, v in tail)]
    fig, ax = plt.subplots(figsize=(4.6, 3.0), dpi=110)
    ax.pie(values, labels=labels, autopct="%1.0f%%", startangle=90, counterclock=False,
           textprops={"fontsize": 8}, pctdistance=0.72)
    ax.axis("equal")
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

class _ReportPdf(FPDF):
    _ACCENT = (46, 92, 160)
    _RULE = (210, 216, 226)
    _MUTED = (110, 118, 132)

    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*self._MUTED)
        self.cell(0, 6, "Power Design Toolkit - LLC Design Report", align="L")
        self.cell(0, 6, f"Page {self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*self._RULE)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(2)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*self._MUTED)
        self.cell(0, 6, "Generated by Power Design Toolkit - engineering reference tool; verify before release.", align="C")

    def section(self, title: str) -> None:
        self.ln(3)
        self.set_fill_color(*self._ACCENT)
        self.rect(self.l_margin, self.get_y(), 2.2, 7.2, "F")
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(20, 26, 38)
        self.set_x(self.l_margin + 4)
        self.cell(0, 7.2, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1.5)

    def kv_table(self, rows: list[tuple[str, str]], cols: int = 2) -> None:
        self.set_font("Helvetica", "", 9)
        fill = False
        for index, (key, value) in enumerate(rows):
            if index % cols == 0:
                self._table_row_head()
            self.set_fill_color(244, 247, 252) if fill else self.set_fill_color(255, 255, 255)
            self.set_text_color(60, 70, 86)
            self.set_font("Helvetica", "", 9)
            self.cell(46, 6.4, key, border=1, fill=True)
            self.set_font("Helvetica", "B", 9)
            self.set_text_color(20, 26, 38)
            self.cell(0, 6.4, value, border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
            fill = not fill

    def _table_row_head(self) -> None:
        pass

    def data_table(self, headers: list[str], rows: list[list[str]]) -> None:
        widths = [self.w - self.l_margin - self.r_margin] + [0] * (len(headers) - 1)
        col_width = (self.w - self.l_margin - self.r_margin) / len(headers)
        widths = [col_width] * len(headers)
        self.set_font("Helvetica", "B", 8)
        self.set_fill_color(46, 92, 160)
        self.set_text_color(255, 255, 255)
        for index, header in enumerate(headers):
            self.cell(widths[index], 6.4, header, border=1, fill=True, align="C")
        self.ln()
        self.set_font("Helvetica", "", 8)
        self.set_text_color(20, 26, 38)
        for row_index, row in enumerate(rows):
            self.set_fill_color(244, 247, 252) if row_index % 2 else self.set_fill_color(255, 255, 255)
            for index, cell in enumerate(row):
                self.cell(widths[index], 5.8, cell, border=1, fill=True, align="C")
            self.ln()
        self.ln(2)

    def paragraph(self, text: str) -> None:
        self.set_font("Helvetica", "", 9)
        self.set_text_color(40, 48, 62)
        self.multi_cell(0, 4.6, text)
        self.ln(1.5)


def _fmt(v: Any, digits: int = 3) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}"


def build_pdf_report(spec_payload: dict[str, Any], project: str = "", engineer: str = "") -> bytes:
    result = analyze_llc(spec_payload)
    summary = result["summary"]

    pdf = _ReportPdf(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ---- Title block ----
    pdf.set_fill_color(15, 28, 52)
    pdf.rect(0, 0, pdf.w, 34, "F")
    pdf.set_y(11)
    pdf.set_font("Helvetica", "B", 17)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 9, "LLC Resonant Converter Design Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(180, 200, 226)
    pdf.cell(0, 6, "Power Design Toolkit  ·  V7.5 Web", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_y(40)
    pdf.set_text_color(20, 26, 38)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(60, 6, f"Project: {project or '-'}")
    pdf.cell(60, 6, f"Engineer: {engineer or '-'}")
    pdf.cell(0, 6, f"Date: {date.today().isoformat()}", new_x="LMARGIN", new_y="NEXT")

    # ---- 1. Specification ----
    pdf.section("1. Specification")
    spec = result["spec"]
    pdf.kv_table([
        ("Input voltage min", f"{_fmt(spec['vbus_min_normal_v'], 0)} V"),
        ("Input voltage nom", f"{_fmt(spec['vbus_nom_v'], 0)} V"),
        ("Input voltage max", f"{_fmt(spec['vbus_max_v'], 0)} V"),
        ("Hold-up end voltage", f"{_fmt(spec['vbus_hold_end_v'], 0)} V"),
        ("Output voltage", f"{_fmt(spec['vout_v'], 1)} V"),
        ("Output power", f"{_fmt(spec['pout_w'], 0)} W"),
        ("Resonant frequency", f"{_fmt(spec['resonant_frequency_hz'] / 1000, 0)} kHz"),
        ("Primary topology", spec["primary_topology"]),
        ("Secondary topology", spec["secondary_topology"]),
        ("Primary turns", f"{spec['primary_turns']:g} T"),
        ("Secondary turns", f"{spec['secondary_turns']:g} T"),
        ("Turns ratio", _fmt(summary["turns_ratio"], 3)),
    ])

    # ---- 2. Circuit ----
    pdf.section("2. Circuit")
    pdf.paragraph(
        f"{spec['primary_topology'].replace('_', ' ')} with {spec['secondary_topology'].replace('_', ' ')} "
        f"rectifier, transformer turns ratio {summary['turns_ratio']:.3f}:1, "
        f"series resonant tank (Lr-Cr) with magnetizing inductance Lm = Ln x Lr. "
        f"Primary switching devices: {spec['primary_device']} (x{spec['primary_parallel_devices']:g}); "
        f"SR devices: {spec['sr_device']} (x{spec['sr_parallel_devices_per_position']:g} per position)."
    )

    # ---- 3. Tank ----
    pdf.section("3. Resonant Tank")
    pdf.kv_table([
        ("Lr", f"{_fmt(summary['lr_h'] * 1e6, 2)} uH"),
        ("Cr", f"{_fmt(summary['cr_f'] * 1e9, 2)} nF"),
        ("Lm", f"{_fmt(summary['lm_h'] * 1e6, 2)} uH"),
        ("Characteristic impedance Zr", f"{_fmt(summary['zr_ohm'], 2)} ohm"),
        ("Equivalent AC load (nominal)", f"{_fmt(summary['rac_nom_ohm'], 3)} ohm"),
        ("Ln = Lm / Lr", f"{_fmt(spec['ln_ratio'], 2)}"),
        ("Q @ full load", f"{_fmt(spec['q_full_load'], 2)}"),
        ("Nominal switching frequency", f"{_fmt(summary['nominal_switching_frequency_hz'] / 1000, 2)} kHz"),
    ])

    # ---- 4. Transformer ----
    pdf.section("4. Transformer")
    pdf.kv_table([
        ("Core", summary["transformer_core"]),
        ("Primary turns", f"{summary['transformer_primary_turns']:g} T"),
        ("Secondary turns", f"{summary['transformer_secondary_turns']:g} T"),
        ("Fill factor", f"{_fmt(summary['transformer_fill_factor'] * 100, 1)} %"),
        ("Peak flux density (worst point)", f"{_fmt(summary['transformer_worst_b_peak_t'], 3)} T"),
        ("Total gap", f"{_fmt(summary['transformer_gap_total_mm'], 2)} mm"),
        ("Estimated hotspot", f"{_fmt(summary['transformer_hotspot_c'], 1)} C"),
        ("Feasible", "YES" if summary["transformer_feasible"] else "NO"),
    ])
    pdf.paragraph(
        f"Resonant inductor: core {summary['resonant_inductor_core']}, "
        f"{summary['resonant_inductor_turns']:g} turns, {summary['resonant_inductor_layers']:g} layers, "
        f"estimated hotspot {_fmt(summary['resonant_inductor_hotspot_c'], 1)} C."
    )

    # ---- 5. Gain ----
    pdf.section("5. Gain Map")
    pdf.image(_gain_map_png(result), w=pdf.w - pdf.l_margin - pdf.r_margin)

    # ---- 6. ZVS ----
    pdf.section("6. ZVS Analysis")
    zvs_rows: list[list[str]] = []
    for point in result["operating_points"]:
        passed = point["zvs_margin"] >= summary["zvs_required_margin"]
        zvs_rows.append([
            point["label"],
            _fmt(point["switching_frequency_hz"] / 1000, 1),
            _fmt(point["input_phase_deg"], 1),
            _fmt(point["commutation_current_a"], 2),
            _fmt(point["zvs_margin"], 2),
            "PASS" if passed else "FAIL",
        ])
    pdf.data_table(
        ["Operating point", "fs / kHz", "Phase / deg", "Commutation I / A", "ZVS margin", "Status"],
        zvs_rows,
    )
    pdf.kv_table([
        ("Required ZVS margin", f"{_fmt(summary['zvs_required_margin'], 2)}x"),
        ("Operating frequency range", f"{_fmt(summary['frequency_range_hz'][0] / 1000, 0)} - {_fmt(summary['frequency_range_hz'][1] / 1000, 0)} kHz"),
    ])

    # ---- 7. Loss ----
    pdf.section("7. Loss Breakdown")
    pdf.image(_loss_pie_png(result), w=76)
    pdf.ln(2)
    loss_rows = [
        [_LOSS_LABELS.get(key, key), _fmt(value, 2), f"{value / max(summary['nominal_total_loss_w'], 1e-9) * 100:.1f} %"]
        for key, value in sorted(
            ((k, v) for k, v in result["nominal_loss_breakdown"].items() if v > 0.01),
            key=lambda kv: kv[1], reverse=True,
        )
    ]
    pdf.data_table(["Loss item", "Loss / W", "Share"], loss_rows)
    pdf.kv_table([
        ("Total loss (nominal)", f"{_fmt(summary['nominal_total_loss_w'], 1)} W"),
        ("Nominal efficiency", f"{_fmt(summary['nominal_efficiency'] * 100, 3)} %"),
        ("Minimum efficiency", f"{_fmt(summary['minimum_efficiency'] * 100, 3)} % ({summary['minimum_efficiency_label']})"),
        ("Worst-case loss", f"{_fmt(summary['worst_loss_w'], 1)} W ({summary['worst_loss_label']})"),
    ])

    # ---- 8. BOM ----
    pdf.section("8. BOM (reference)")
    pdf.data_table(
        ["Item", "Part / value", "Qty", "Note"],
        [
            ["Primary MOSFET", spec["primary_device"], f"{spec['primary_parallel_devices']:g}", "Power stage"],
            ["SR MOSFET", spec["sr_device"], f"{spec['sr_parallel_devices_per_position']:g}", "Per SR position"],
            ["Transformer core", summary["transformer_core"], "1", f"${_fmt(summary['transformer_cost_usd'], 2)} ref"],
            ["Resonant inductor core", summary["resonant_inductor_core"], "1", f"${_fmt(summary['resonant_inductor_cost_usd'], 2)} ref"],
            ["Resonant capacitor Cr", f"{_fmt(summary['cr_f'] * 1e9, 2)} nF", "1", "Rating per spec"],
        ],
    )
    pdf.kv_table([
        ("Power-stage + magnetics cost (ref)", f"${_fmt(summary['bom_cost_usd'], 2)}"),
    ])

    # ---- 9. Conclusion ----
    pdf.section("9. Conclusion")
    if result["feasible"]:
        pdf.paragraph(
            f"The design is FEASIBLE under the current engineering model: nominal efficiency "
            f"{_fmt(summary['nominal_efficiency'] * 100, 3)} % at {_fmt(summary['nominal_switching_frequency_hz'] / 1000, 2)} kHz, "
            f"ZVS margin {_fmt(summary['nominal_zvs_margin'], 2)}x (required {_fmt(summary['zvs_required_margin'], 2)}x), "
            f"estimated maximum hotspot {_fmt(summary['max_hotspot_c'], 1)} C."
        )
    else:
        pdf.paragraph("The design is NOT feasible under the current engineering model. Reasons:")
        for reason in result["feasibility_reasons"]:
            pdf.paragraph(f"- {reason}")
    if result["warnings"]:
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 6, "Engineering warnings", new_x="LMARGIN", new_y="NEXT")
        for warning in result["warnings"]:
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(90, 96, 110)
            pdf.multi_cell(0, 4.4, f"- {warning}")
            pdf.ln(0.8)

    return bytes(pdf.output())


# ---------------------------------------------------------------------------
# Excel
# ---------------------------------------------------------------------------

def build_excel_report(spec_payload: dict[str, Any]) -> bytes:
    import pandas as pd

    result = analyze_llc(spec_payload)
    summary = result["summary"]
    spec = result["spec"]

    spec_frame = pd.DataFrame(
        [{"parameter": key, "value": value} for key, value in spec.items()]
    )
    op_frame = pd.DataFrame(result["operating_points"])
    loss_frame = pd.DataFrame(
        [{"loss_item": _LOSS_LABELS.get(key, key), "loss_w": value}
         for key, value in result["nominal_loss_breakdown"].items() if value > 0.01]
    )
    bom_rows = [
        {"item": "Primary MOSFET", "part": spec["primary_device"], "qty": spec["primary_parallel_devices"], "note": "Power stage"},
        {"item": "SR MOSFET", "part": spec["sr_device"], "qty": spec["sr_parallel_devices_per_position"], "note": "Per SR position"},
        {"item": "Transformer core", "part": summary["transformer_core"], "qty": 1, "note": f"${summary['transformer_cost_usd']} ref"},
        {"item": "Resonant inductor core", "part": summary["resonant_inductor_core"], "qty": 1, "note": f"${summary['resonant_inductor_cost_usd']} ref"},
        {"item": "Resonant capacitor", "part": f"{summary['cr_f'] * 1e9:.2f} nF", "qty": 1, "note": "Rating per spec"},
    ]
    bom_frame = pd.DataFrame(bom_rows)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        spec_frame.to_excel(writer, sheet_name="Specification", index=False)
        op_frame.to_excel(writer, sheet_name="Operating Points", index=False)
        loss_frame.to_excel(writer, sheet_name="Loss Breakdown", index=False)
        bom_frame.to_excel(writer, sheet_name="BOM", index=False)
        pd.DataFrame([{"summary_item": key, "value": value} for key, value in summary.items()]).to_excel(
            writer, sheet_name="Summary", index=False
        )
    return buffer.getvalue()
