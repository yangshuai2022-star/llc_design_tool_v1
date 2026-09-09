from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from fpdf import FPDF

from .. import __version__
from ..core.spec import TankParameterMode
from ..models.devices import DeviceDatabase
from ..models.system import SystemAnalysis
from ..validation import validate_bundled_data


@dataclass(frozen=True)
class FormulaStep:
    name: str
    formula: str
    substitution: str
    result: str
    method: str = ""


def _value(value: float, unit: str = "", digits: int = 6) -> str:
    suffix = f" {unit}" if unit else ""
    return f"{float(value):.{digits}g}{suffix}"


def build_formula_steps(analysis: SystemAnalysis) -> dict[str, list[FormulaStep]]:
    spec = analysis.spec
    tank = analysis.tank
    point = analysis.nominal
    op = point.operating_point
    primary = point.primary
    sr = point.synchronous_rectifier
    db = DeviceDatabase()
    primary_device = db.get_primary(spec.primary_device)
    sr_device = db.get_sr(spec.sr_device)

    turns_ratio = spec.primary_turns / spec.secondary_turns
    rload = spec.vout_v**2 / spec.pout_w
    omega_r = 2.0 * math.pi * tank.fr_hz
    omega_s = 2.0 * math.pi * op.switching_frequency_hz
    z_series = 1j * omega_s * tank.lr_h + 1.0 / (1j * omega_s * tank.cr_f)
    z_lm = 1j * omega_s * tank.lm_h
    z_parallel = 1.0 / (1.0 / op.rac_ohm + 1.0 / z_lm)
    transfer = z_parallel / (z_series + z_parallel)
    allowed_primary_v = primary_device.vds_max_v * spec.primary_voltage_derating
    allowed_sr_v = sr_device.vds_max_v * spec.sr_voltage_derating

    tank_steps = [
        FormulaStep(
            "Transformer turns ratio",
            "n := Np / Ns",
            f"n := {spec.primary_turns} / {spec.secondary_turns}",
            f"n = {_value(turns_ratio)}",
        ),
        FormulaStep(
            "DC load resistance",
            "Rload := Vo^2 / Po",
            f"Rload := {spec.vout_v:g}^2 / {spec.pout_w:g}",
            f"Rload = {_value(rload, 'ohm')}",
        ),
        FormulaStep(
            "FHA load reflected to primary",
            "Rac := (8/pi^2) * n^2 * Rload",
            f"Rac := (8/pi^2) * {turns_ratio:g}^2 * {rload:.6g}",
            f"Rac = {_value(tank.rac_nom_ohm, 'ohm')}",
        ),
    ]
    if TankParameterMode(spec.parameter_mode) == TankParameterMode.USER_DEFINED:
        tank_steps.extend([
            FormulaStep(
                "Characteristic impedance from entered tank",
                "Zr := sqrt(Lr / Cr)",
                f"Zr := sqrt({tank.lr_h:.6g} / {tank.cr_f:.6g})",
                f"Zr = {_value(tank.zr_ohm, 'ohm')}",
                "USER_DEFINED verification path; entered Lr, Cr and Lm are not optimized.",
            ),
            FormulaStep(
                "Resonant frequency from entered tank",
                "fr := 1 / (2*pi*sqrt(Lr*Cr))",
                f"fr := 1 / (2*pi*sqrt({tank.lr_h:.6g}*{tank.cr_f:.6g}))",
                f"fr = {_value(tank.fr_hz / 1e3, 'kHz')}",
            ),
            FormulaStep(
                "Inductance ratio from entered tank",
                "Ln := Lm / Lr",
                f"Ln := {tank.lm_h:.6g} / {tank.lr_h:.6g}",
                f"Ln = {_value(tank.ln_ratio)}",
            ),
            FormulaStep(
                "Loaded quality factor from entered tank",
                "Qe := Zr / Rac",
                f"Qe := {tank.zr_ohm:.6g} / {tank.rac_nom_ohm:.6g}",
                f"Qe = {_value(tank.q_full_load)}",
            ),
        ])
    else:
        tank_steps.extend([
            FormulaStep(
                "Characteristic impedance",
                "Zr := Qe * Rac",
                f"Zr := {tank.q_full_load:g} * {tank.rac_nom_ohm:.6g}",
                f"Zr = {_value(tank.zr_ohm, 'ohm')}",
            ),
            FormulaStep(
                "Resonant angular frequency",
                "wr := 2 * pi * fr",
                f"wr := 2 * pi * {tank.fr_hz:.6g}",
                f"wr = {_value(omega_r, 'rad/s')}",
            ),
            FormulaStep(
                "Resonant inductance",
                "Lr := Zr / wr",
                f"Lr := {tank.zr_ohm:.6g} / {omega_r:.6g}",
                f"Lr = {_value(tank.lr_h * 1e6, 'uH')}",
            ),
            FormulaStep(
                "Resonant capacitance",
                "Cr := 1 / (wr * Zr)",
                f"Cr := 1 / ({omega_r:.6g} * {tank.zr_ohm:.6g})",
                f"Cr = {_value(tank.cr_f * 1e9, 'nF')}",
            ),
            FormulaStep(
                "Magnetizing inductance",
                "Lm := Ln * Lr",
                f"Lm := {tank.ln_ratio:g} * {tank.lr_h * 1e6:.6g} uH",
                f"Lm = {_value(tank.lm_h * 1e6, 'uH')}",
            ),
        ])

    return {
        "Resonant tank synthesis": tank_steps,
        "Nominal operating point": [
            FormulaStep(
                "Required normalized gain",
                "Mreq := n * (Vo + Vrect) / (Kbridge * Vbus)",
                f"Mreq := {turns_ratio:g} * ({spec.vout_v:g} + {spec.rectifier_equivalent_drop_v:g}) / ({spec.bridge_gain:g} * {op.vbus_v:g})",
                f"Mreq = {_value(op.required_gain)}",
            ),
            FormulaStep(
                "Series tank impedance",
                "Zs := j*ws*Lr + 1/(j*ws*Cr)",
                f"Zs := j*{omega_s:.6g}*{tank.lr_h:.6g} + 1/(j*{omega_s:.6g}*{tank.cr_f:.6g})",
                f"Zs = {z_series.real:.6g} + j*{z_series.imag:.6g} ohm",
            ),
            FormulaStep(
                "Parallel branch impedance",
                "Zp := 1 / (1/Rac + 1/(j*ws*Lm))",
                f"Zp := 1 / (1/{op.rac_ohm:.6g} + 1/(j*{omega_s:.6g}*{tank.lm_h:.6g}))",
                f"Zp = {z_parallel.real:.6g} + j*{z_parallel.imag:.6g} ohm",
            ),
            FormulaStep(
                "Tank gain and frequency root",
                "M(fs) := abs(Zp / (Zs + Zp)); solve M(fs) = Mreq",
                f"abs(({z_parallel.real:.5g}+j*{z_parallel.imag:.5g}) / ({(z_series+z_parallel).real:.5g}+j*{(z_series+z_parallel).imag:.5g}))",
                f"fs = {_value(op.switching_frequency_hz / 1e3, 'kHz')}; M = {_value(abs(transfer))}; branch = {op.branch}",
                "Logarithmic scan (1600 samples) brackets every root, then Brent's method refines it.",
            ),
            FormulaStep(
                "Resonant current",
                "Ir,rms := Vbridge,1,rms / abs(Zin); Ir,pk := sqrt(2)*Ir,rms",
                f"Ir,rms := {op.bridge_fundamental_rms_v:.6g} / {abs(op.input_impedance_ohm):.6g}",
                f"Ir,rms = {_value(op.resonant_current_rms_a, 'A')}; Ir,pk = {_value(op.resonant_current_peak_a, 'A')}",
            ),
            FormulaStep(
                "Commutation current",
                "Icomm := max(0.75*Imag,pk, abs(Ir,pk*sin(phi)))",
                f"Icomm := max(0.75*{op.magnetizing_current_peak_a:.6g}, abs({op.resonant_current_peak_a:.6g}*sin({op.input_phase_deg:.6g} deg)))",
                f"Icomm = {_value(op.commutation_current_a, 'A')}",
            ),
        ],
        "Primary bridge and ZVS": [
            FormulaStep(
                "Primary conduction loss",
                "Pcond := Ir,rms^2 * Nseries * Rds,on(Tj) / Nparallel",
                f"Pcond := {op.resonant_current_rms_a:.6g}^2 * {spec.bridge_series_devices} * ({primary_device.rds_at(spec.primary_junction_temperature_c):.6g}/{spec.primary_parallel_devices})",
                f"Pcond = {_value(primary.conduction_w, 'W')}",
            ),
            FormulaStep(
                "Primary gate-drive loss",
                "Pgate := Ndevice * Nparallel * Qg * Vg * fs",
                f"Pgate := {spec.bridge_device_count} * {spec.primary_parallel_devices} * {primary_device.qg_c:.6g} * {primary_device.gate_voltage_v:.6g} * {op.switching_frequency_hz:.6g}",
                f"Pgate = {_value(primary.gate_drive_w, 'W')}",
            ),
            FormulaStep(
                "Primary turn-off loss",
                "Eoff := Eref*(Vbus/Vref)*(Ioff/Iref); Poff := Ndevice*Nparallel*Eoff*fs",
                f"Ioff := {op.commutation_current_a:.6g}*{spec.primary_turnoff_current_factor:.6g}/{spec.primary_parallel_devices}; fs := {op.switching_frequency_hz:.6g}",
                f"Poff = {_value(primary.turnoff_w, 'W')}",
            ),
            FormulaStep(
                "ZVS charge margin",
                "MQ := Icomm * tdead / (2*Nparallel*max(Qoss, Coss*Vbus))",
                f"MQ := {op.commutation_current_a:.6g} * {spec.primary_deadtime_s:.6g} / (2*{spec.primary_parallel_devices}*max({primary_device.qoss_c:.6g}, {primary_device.coss_er_f:.6g}*{op.vbus_v:.6g}))",
                f"MQ = {_value(primary.zvs_charge_margin)}",
            ),
            FormulaStep(
                "ZVS energy margin",
                "ME := 0.5*(Lr+Lm)*Icomm^2 / (2*Nparallel*0.5*Coss*Vbus^2)",
                f"ME := 0.5*({tank.lr_h:.6g}+{tank.lm_h:.6g})*{op.commutation_current_a:.6g}^2 / (2*{spec.primary_parallel_devices}*0.5*{primary_device.coss_er_f:.6g}*{op.vbus_v:.6g}^2)",
                f"ME = {_value(primary.zvs_energy_margin)}; effective margin = {_value(min(primary.zvs_charge_margin, primary.zvs_energy_margin))}",
            ),
            FormulaStep(
                "Residual Coss loss after incomplete ZVS",
                "r := clamp(1-min(MQ,ME),0,1)^2; Pcoss := Ndevice*Nparallel*Eoss*fs*r",
                f"r := {primary.zvs_residual_fraction:.6g}; Eoss := 0.5*{primary_device.coss_er_f:.6g}*{op.vbus_v:.6g}^2",
                f"Pcoss = {_value(primary.residual_coss_w, 'W')}",
            ),
            FormulaStep(
                "Primary dead-time diode loss",
                "tcharge := Qreq/Icomm; tdiode := max(tdead-tcharge,0); Pdiode := Ncomm*Vf*Icomm*tdiode*fs",
                f"tdead := {spec.primary_deadtime_s:.6g}; Icomm := {op.commutation_current_a:.6g}; Ncomm := {2*spec.bridge_series_devices}",
                f"Pdiode = {_value(primary.deadtime_diode_w, 'W')}",
            ),
            FormulaStep(
                "Primary voltage derating",
                "Vstress <= Vds,max * derating",
                f"{primary.voltage_stress_v:.6g} <= {primary_device.vds_max_v:.6g} * {spec.primary_voltage_derating:.6g}",
                f"{_value(primary.voltage_stress_v, 'V')} <= {_value(allowed_primary_v, 'V')} : {'PASS' if primary.voltage_stress_v <= allowed_primary_v else 'FAIL'}",
            ),
        ],
        "Synchronous rectifier and capacitors": [
            FormulaStep(
                "SR conduction loss",
                "Psr,cond := Is,rms^2 * 2*Rds,on(Tj) / Nparallel",
                f"Psr,cond := {op.secondary_current_rms_a:.6g}^2 * 2*{sr_device.rds_at(spec.sr_junction_temperature_c):.6g}/{spec.sr_parallel_devices_per_position}",
                f"Psr,cond = {_value(sr.conduction_w, 'W')}",
            ),
            FormulaStep(
                "SR dead-time diode loss",
                "Pdiode := 4*Vf*(1/2)*Is,pk*ws*tw^2*fs; tw := max(tdead+tadvance,0)",
                f"tw := max({spec.sr_deadtime_s:.6g}+{spec.sr_turnoff_advance_s:.6g},0); Is,pk := {op.secondary_current_peak_a:.6g}",
                f"Pdiode = {_value(sr.deadtime_diode_w, 'W')}",
            ),
            FormulaStep(
                "SR turn-off loss",
                "Ioff := Is,pk*sin(min(ws*tadvance,pi/2))/Nparallel; Poff := 4*Nparallel*Eoff*fs",
                f"Is,pk := {op.secondary_current_peak_a:.6g}; tadvance := {spec.sr_turnoff_advance_s:.6g}",
                f"Poff = {_value(sr.turnoff_w, 'W')}",
            ),
            FormulaStep(
                "SR output-capacitance loss",
                "Eoss := 0.5*Coss*Vstress^2; Pcoss := 4*Nparallel*Eoss*fs*Kloss",
                f"Vstress := {sr.voltage_stress_v:.6g}; Kloss := {spec.sr_coss_dissipation_factor:.6g}",
                f"Pcoss = {_value(sr.coss_w, 'W')}",
            ),
            FormulaStep(
                "SR gate-drive loss",
                "Pgate := 4*Nparallel*Qg*Vg*fs",
                f"Pgate := 4*{spec.sr_parallel_devices_per_position}*{sr_device.qg_c:.6g}*{sr_device.gate_voltage_v:.6g}*{op.switching_frequency_hz:.6g}",
                f"Pgate = {_value(sr.gate_drive_w, 'W')}",
            ),
            FormulaStep(
                "SR voltage derating",
                "Vsr := Kov * Vo; Vsr <= Vds,max * derating",
                f"Vsr := {spec.sr_voltage_overshoot_factor:.6g} * {spec.vout_v:.6g}; limit := {sr_device.vds_max_v:.6g} * {spec.sr_voltage_derating:.6g}",
                f"{_value(sr.voltage_stress_v, 'V')} <= {_value(allowed_sr_v, 'V')} : {'PASS' if sr.voltage_stress_v <= allowed_sr_v else 'FAIL'}",
            ),
            FormulaStep(
                "Resonant-capacitor voltage",
                "Xc := 1/(2*pi*fs*Cr); Vcr,rms := Ir,rms*Xc; Vcr,pk := Vbias + sqrt(2)*Vcr,rms",
                f"Xc := 1/(2*pi*{op.switching_frequency_hz:.6g}*{tank.cr_f:.6g})",
                f"Vcr,rms = {_value(point.resonant_capacitor.voltage_rms_v, 'V')}; Vcr,pk = {_value(point.resonant_capacitor.voltage_peak_v, 'V')}",
            ),
            FormulaStep(
                "Resonant-capacitor ESR loss",
                "Pcr := Ir,rms^2 * ESRcr",
                f"Pcr := {op.resonant_current_rms_a:.6g}^2 * {spec.resonant_cap_esr_ohm:.6g}",
                f"Pcr = {_value(point.resonant_capacitor.loss_w, 'W')}",
            ),
            FormulaStep(
                "Output-capacitor ripple",
                "icap(t) := (pi*Io/2)*sin(pi*t/Tripple) - Io; q(t) := integral(icap dt); v(t) := q/C + ESR*icap",
                f"Io={op.output_current_a:.6g} A; Tripple=1/(2*{op.switching_frequency_hz:.6g}); C={spec.output_capacitance_f:.6g} F; ESR={spec.output_cap_esr_ohm:.6g} ohm",
                f"DeltaVpp = {_value(point.output_capacitor.total_ripple_vpp, 'V')}; Pcap = {_value(point.output_capacitor.capacitor_loss_w, 'W')}",
                "The waveform is numerically integrated with 4096 samples; it is not replaced by a ripple shortcut.",
            ),
        ],
        "Magnetics, hold-up and totals": [
            FormulaStep(
                "Transformer peak flux density",
                "Bpk := Vpri,square / (4*Np*Amin*fs)",
                f"Bpk := {op.transformer_square_equivalent_v:.6g} / (4*{spec.primary_turns}*{analysis.transformer.core.amin_m2:.6g}*{op.switching_frequency_hz:.6g})",
                f"Bpk = {_value(point.transformer.b_peak_min_area_t, 'T')}; limit = {_value(spec.transformer_max_b_t, 'T')}",
            ),
            FormulaStep(
                "Transformer equivalent air gap",
                "g := mu0*Np^2*Ae/Lm - le/mur",
                f"g := 4*pi*1e-7*{spec.primary_turns}^2*{analysis.transformer.core.ae_m2:.6g}/{tank.lm_h:.6g} - {analysis.transformer.core.le_m:.6g}/{analysis.transformer.core.mu_r:.6g}",
                f"g = {_value(analysis.transformer.gap_total_mm, 'mm')}; fill = {_value(analysis.transformer.fill_factor*100, '%')}",
            ),
            FormulaStep(
                "Resonant-inductor peak flux density",
                "Bpk := Lr * Ir,pk / (N * Ae)",
                f"Bpk := {tank.lr_h:.6g} * {op.resonant_current_peak_a:.6g} / ({analysis.resonant_inductor.turns} * {analysis.resonant_inductor.core.ae_m2:.6g})",
                f"Bpk = {_value(point.resonant_inductor.b_peak_t, 'T')}; limit = {_value(spec.resonant_inductor_max_b_t, 'T')}",
            ),
            FormulaStep(
                "Resonant-inductor equivalent air gap",
                "g := mu0*N^2*Ae/Lr - le/mur",
                f"N := {analysis.resonant_inductor.turns}; Ae := {analysis.resonant_inductor.core.ae_m2:.6g}; Lr := {tank.lr_h:.6g}",
                f"g = {_value(analysis.resonant_inductor.gap_total_mm, 'mm')}; fill = {_value(analysis.resonant_inductor.fill_factor*100, '%')}",
            ),
            FormulaStep(
                "Magnetic core loss by iGSE",
                "Pv := (1/T)*integral(ki*abs(dB/dt)^alpha*(DeltaB)^(beta-alpha) dt); Pcore := Pv*Ve",
                f"waveform samples := {spec.magnetic_waveform_samples}; transformer material := {analysis.transformer.core.material}; inductor material := {analysis.resonant_inductor.core.material}",
                f"Pcore,xfmr = {_value(point.transformer.core_w, 'W')}; Pcore,Lr = {_value(point.resonant_inductor.core_w, 'W')}",
                "Temperature-interpolated material coefficients are applied to reconstructed B(t).",
            ),
            FormulaStep(
                "Litz winding loss by harmonic layer MMF",
                "i(t) -> FFT(Ih); Pcu := Pdc + Pskin + Pprox + Pbundle + Pterm (+ Pgap for Lr)",
                f"harmonics := 1..{spec.litz_max_harmonic}; transformer stack := P/2-S-P/2",
                f"Pcu,xfmr = {_value(point.transformer.primary_copper_w + point.transformer.secondary_copper_w, 'W')}; Pcu,Lr = {_value(point.resonant_inductor.copper_w, 'W')}",
                "Round-strand skin effect, layer MMF, transposition and termination terms remain separate in the ledger.",
            ),
            FormulaStep(
                "Magnetic thermal fixed-point iteration",
                "Tpred := Tamb + Ploss(T)*Rth; Tnext := 0.55*T + 0.45*Tpred",
                f"Tamb := {spec.ambient_temperature_c:.6g}; tolerance := {spec.magnetic_thermal_tolerance_c:.6g}; max iterations := {spec.magnetic_thermal_max_iterations}",
                f"Thot,xfmr = {_value(point.transformer.estimated_hotspot_c, 'degC')}; Thot,Lr = {_value(point.resonant_inductor.estimated_hotspot_c, 'degC')}",
            ),
            FormulaStep(
                "DC-link required capacitance",
                "Creq := 2*Pin*thold / (Vstart^2 - Vend^2)",
                f"Creq := 2*{spec.pout_w/spec.efficiency_assumption:.6g}*{spec.requested_hold_time_s:.6g} / ({spec.vbus_nom_v:.6g}^2-{spec.vbus_hold_end_v:.6g}^2)",
                f"Creq = {_value(analysis.bus_capacitor.required_capacitance_f * 1e6, 'uF')}; installed = {_value(spec.bus_capacitance_f * 1e6, 'uF')}",
            ),
            FormulaStep(
                "Total modeled loss",
                "Ploss := sum(non-overlapping loss buckets)",
                " + ".join(f"{float(value):.4g}" for value in point.breakdown().values()),
                f"Ploss = {_value(point.total_loss_w, 'W')}",
            ),
            FormulaStep(
                "Auxiliary loss model",
                "Paux := 5 W + 0.004*Po",
                f"Paux := 5 + 0.004*{op.pout_w:.6g}",
                f"Paux = {_value(point.auxiliary_w, 'W')}",
            ),
            FormulaStep(
                "Efficiency",
                "eta := Po / (Po + Ploss)",
                f"eta := {op.pout_w:.6g} / ({op.pout_w:.6g} + {point.total_loss_w:.6g})",
                f"eta = {_value(point.efficiency * 100, '%')}",
            ),
        ],
    }


class _FormulaPdf(FPDF):
    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "", 8)
        self.set_text_color(95, 105, 120)
        self.cell(0, 5, "Power Design Toolkit - auditable LLC worksheet", align="R")
        self.ln(7)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(110, 120, 135)
        self.cell(0, 5, f"V{__version__}  |  Page {self.page_no()}/{{nb}}", align="C")

    def section(self, title: str) -> None:
        self.set_fill_color(25, 62, 105)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 8, title, fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def formula_step(self, index: int, step: FormulaStep) -> None:
        self.set_fill_color(242, 246, 251)
        self.set_draw_color(184, 199, 217)
        self.set_text_color(30, 43, 60)
        self.set_font("Helvetica", "B", 9)
        self.cell(0, 6, f"{index}. {step.name}", border="LTR", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_font("Courier", "", 8)
        self.multi_cell(
            0, 4.5, step.formula, border="LR", fill=True,
            new_x="LMARGIN", new_y="NEXT",
        )
        self.set_text_color(71, 84, 101)
        self.multi_cell(
            0, 4.5, step.substitution, border="LR", fill=True,
            new_x="LMARGIN", new_y="NEXT",
        )
        self.set_text_color(9, 102, 69)
        self.set_font("Courier", "B", 8)
        self.multi_cell(
            0, 5, step.result, border="LBR", fill=True,
            new_x="LMARGIN", new_y="NEXT",
        )
        if step.method:
            self.set_text_color(93, 104, 120)
            self.set_font("Helvetica", "I", 7)
            self.multi_cell(
                0, 4, f"Method: {step.method}",
                new_x="LMARGIN", new_y="NEXT",
            )
        self.ln(2)


def build_formula_pdf(analysis: SystemAnalysis, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    steps = build_formula_steps(analysis)
    provenance = validate_bundled_data()

    pdf = _FormulaPdf(format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()
    pdf.set_fill_color(14, 39, 70)
    pdf.rect(0, 0, pdf.w, 62, "F")
    pdf.set_y(16)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 10, "LLC Design Calculation Worksheet", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(190, 213, 238)
    pdf.cell(0, 7, "Mathcad-style formula, substitution and result trace", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_y(70)
    pdf.set_text_color(30, 40, 54)
    pdf.set_font("Helvetica", "", 10)
    spec = analysis.spec
    summary = [
        ("Toolkit", f"Power Design Toolkit V{__version__}"),
        ("Generated", datetime.now(timezone.utc).date().isoformat()),
        ("Topology", f"{spec.primary_topology.value} + {spec.secondary_topology.value}"),
        ("Power stage", f"{spec.vbus_nom_v:g} Vdc -> {spec.vout_v:g} V / {spec.pout_w/1000:g} kW"),
        ("Result", "PASS" if analysis.feasible else "FAIL"),
        ("Evidence", "Software regression only; hardware verification is not claimed"),
    ]
    for label, value in summary:
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(38, 7, label)
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(0, 7, value, new_x="LMARGIN", new_y="NEXT")

    pdf.ln(3)
    pdf.set_fill_color(255, 246, 220)
    pdf.set_draw_color(225, 177, 76)
    pdf.set_font("Helvetica", "B", 9)
    pdf.multi_cell(
        0,
        5,
        "Engineering boundary: bundled devices, cores and material curves are reference data. "
        "Replace them with exact datasheet and measured models before hardware release.",
        border=1,
        fill=True,
        new_x="LMARGIN",
        new_y="NEXT",
    )

    step_number = 1
    for section, section_steps in steps.items():
        pdf.add_page()
        pdf.section(section)
        for step in section_steps:
            pdf.formula_step(step_number, step)
            step_number += 1

    pdf.add_page()
    pdf.section("Operating-point results")
    pdf.set_text_color(30, 43, 60)
    pdf.set_font("Helvetica", "B", 7)
    widths = (27, 20, 24, 22, 23, 23, 25)
    headers = ("Point", "Load", "fs/kHz", "Ir,rms/A", "Phase/deg", "Loss/W", "Efficiency")
    for width, header in zip(widths, headers):
        pdf.cell(width, 6, header, border=1, align="C")
    pdf.ln()
    pdf.set_font("Helvetica", "", 7)
    for item in analysis.operating_points:
        item_op = item.operating_point
        values = (
            item.label,
            f"{item_op.load_fraction*100:.0f}%",
            f"{item_op.switching_frequency_hz/1e3:.3f}",
            f"{item_op.resonant_current_rms_a:.3f}",
            f"{item_op.input_phase_deg:.2f}",
            f"{item.total_loss_w:.3f}",
            f"{item.efficiency*100:.3f}%",
        )
        for width, value in zip(widths, values):
            pdf.cell(width, 6, value, border=1, align="C")
        pdf.ln()

    pdf.ln(5)
    pdf.section("Loss ledger at nominal point")
    pdf.set_text_color(30, 43, 60)
    pdf.set_font("Helvetica", "", 8)
    for name, value in analysis.nominal.breakdown().items():
        pdf.cell(76, 5, name.replace("_", " "))
        pdf.cell(28, 5, f"{float(value):.6g} W", align="R", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(5)
    pdf.section("Feasibility decision trace")
    pdf.set_text_color(30, 43, 60)
    pdf.set_font("Helvetica", "", 8)
    pdf.multi_cell(
        0,
        5,
        "PASS requires every requested operating point to solve and all hard screens to pass: "
        "magnetic geometry/flux/hotspot, minimum inductive angle, primary ZVS margin >= 1, "
        "MOSFET voltage derating, resonant-capacitor voltage/current margins, output ripple and "
        "DC-link hold-up. Preferred-margin and data-range findings remain warnings.",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.set_font("Helvetica", "B", 8)
    pdf.cell(0, 5, f"Decision: {'PASS' if analysis.feasible else 'FAIL'}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 7)
    if analysis.feasibility_reasons:
        for reason in analysis.feasibility_reasons:
            pdf.multi_cell(0, 4, f"FAIL - {reason}", new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.multi_cell(0, 4, "No hard-screen failures were recorded.", new_x="LMARGIN", new_y="NEXT")
    for warning in analysis.warnings:
        pdf.multi_cell(0, 4, f"WARN - {warning}", new_x="LMARGIN", new_y="NEXT")

    pdf.add_page()
    pdf.section("Evidence and model limits")
    pdf.set_text_color(30, 43, 60)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(
        0,
        5,
        "The formula chain above is recomputed from the same SystemAnalysis object used by the GUI. "
        "Frequency is a numerical root of the FHA complex-impedance gain equation. Output-capacitor "
        "ripple is a sampled integration of one rectified-current lobe. Magnetic loss uses reconstructed "
        "waveforms, temperature iteration, iGSE and detailed winding-loss models; their reported loss "
        "components are included in the loss ledger rather than replaced by a closed-form shortcut.",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 6, "Data provenance", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    for dataset in provenance.datasets:
        pdf.multi_cell(
            0,
            5,
            f"{dataset.dataset}: status={dataset.status.value}; "
            f"hardware_release={dataset.verified_for_hardware}; source={dataset.source}",
            new_x="LMARGIN",
            new_y="NEXT",
        )
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 6, "Falsification conditions", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    pdf.multi_cell(
        0,
        5,
        "Downgrade the result if the software baseline changes outside its recorded tolerances, "
        "if exact component data cannot reproduce the input records, or if external simulation or "
        "bench measurements differ beyond an approved engineering tolerance.",
        new_x="LMARGIN",
        new_y="NEXT",
    )

    pdf.output(output)
    return output


__all__ = ["FormulaStep", "build_formula_pdf", "build_formula_steps"]
