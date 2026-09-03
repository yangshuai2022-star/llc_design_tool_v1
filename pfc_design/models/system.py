"""System-level loss aggregation and efficiency analysis."""

import numpy as np

from ..core.spec import (DesignSpec, MosfetSpec, MosfetDatabase,
                         DiodeDatabase)
from ..core.operating_point import OperatingPoint, compute_mathcad_operating_point
from ..core.line_cycle import build_line_cycle_trace
from ..magnetics.core_database import CoreDatabase
from ..magnetics.core_entry import CoreSpec
from .inductor import InductorDesigner, InductorLoss, InductorDesign
from .mosfet import MosfetLoss
from .diode import DiodeLoss
from .bridge_rectifier import BridgeRectifierLoss
from .capacitor import CapacitorBank
from .base import LossResult

# Thermal fixed-point iteration
THERMAL_MAX_ITER = 10
THERMAL_TOL_C = 0.1


class SystemAnalyzer:
    """Top-level system loss model with explicit per-phase / shared split."""

    def __init__(self, db: CoreDatabase | None = None):
        self.db = db or CoreDatabase()
        self.mosfet_db = MosfetDatabase()
        self.diode_db = DiodeDatabase()
        self.inductor_designer = InductorDesigner(self.db)
        self.inductor_loss = InductorLoss(self.db)
        self.mosfet_loss = MosfetLoss()
        self.diode_loss = DiodeLoss()
        self.bridge_loss = BridgeRectifierLoss()
        self.capacitor = CapacitorBank()

    def analyze(self, spec: DesignSpec,
                op: OperatingPoint | None = None,
                preferred_core: CoreSpec | None = None,
                mosfet: MosfetSpec | None = None,
                n_cores: int = 2,
                design: InductorDesign | None = None) -> dict:
        """Run full system analysis.

        Args:
            spec: design specification
            op: operating point (default: low-line worst case)
            preferred_core: specific core to use
            mosfet: MosfetSpec from database (auto-selects if None)
            n_cores: number of stacked cores
            design: pre-designed inductor to reuse (efficiency map use case);
                    re-designs from spec if None

        Returns:
            dict with all loss results, design data, breakdown
        """
        if op is None:
            op = compute_mathcad_operating_point(spec)

        # Auto-select MOSFET if not provided
        if mosfet is None:
            candidates = self.mosfet_db.query(
                vds_min=spec.vout * 1.3,
                technology="Si"
            )
            mosfet = candidates[0] if candidates else self.mosfet_db.get("C3M0065100K")

        # Resolve boost diode from database (legacy spec fields fallback)
        diode = None
        if spec.diode_part:
            diode = self.diode_db.get(spec.diode_part)

        # 1. Inductor design (or reuse a pre-designed one)
        if design is None:
            design = self.inductor_designer.design(spec, op, preferred_core, n_cores)

        # 2. Build line-cycle trace using effective L at peak current
        #    (accounts for DC bias saturation — the inductor's real operating point)
        L_eff = design.L_eff_at_ipeak_uh
        trace = build_line_cycle_trace(op, spec, L_eff)

        # 2a. Report actual ripple vs target ripple at line peak
        inductor_metrics = compute_inductor_metrics(design, trace, spec)

        # 3. Per-phase losses (line-cycle integrated), with optional
        #    self-consistent loss↔temperature fixed-point iteration.
        t_mos, t_dio = 80.0, 80.0
        t_winding = spec.t_ambient + 40.0
        rth_mos = (spec.rth_mosfet_jc + spec.rth_mosfet_cs
                   + spec.rth_mosfet_sa)

        def _phase_losses():
            ind = self.inductor_loss.compute(design, spec, op, trace=trace,
                                             t_winding=t_winding)
            mos = self.mosfet_loss.compute(spec, op, mosfet=mosfet,
                                           trace=trace, t_j=t_mos)
            dio = self.diode_loss.compute(spec, op, trace=trace, diode=diode,
                                          t_j=t_dio)
            return ind, mos, dio

        if spec.thermal_model:
            for _ in range(THERMAL_MAX_ITER):
                ind_loss, mos_loss, dio_loss = _phase_losses()
                t_mos_new = (spec.t_ambient
                             + rth_mos * mos_loss.power_loss_W)
                t_dio_new = (spec.t_ambient
                             + spec.rth_diode_ja * dio_loss.power_loss_W)
                t_wind_new = (spec.t_ambient
                              + spec.rth_inductor * ind_loss.power_loss_W)
                if (abs(t_mos_new - t_mos) < THERMAL_TOL_C
                        and abs(t_dio_new - t_dio) < THERMAL_TOL_C
                        and abs(t_wind_new - t_winding) < THERMAL_TOL_C):
                    break
                t_mos, t_dio, t_winding = t_mos_new, t_dio_new, t_wind_new
            ind_loss, mos_loss, dio_loss = _phase_losses()
        else:
            ind_loss, mos_loss, dio_loss = _phase_losses()

        thermal = {
            "enabled": spec.thermal_model,
            "t_ambient": spec.t_ambient,
            "t_j_mosfet": t_mos,
            "t_j_diode": t_dio,
            "t_winding": t_winding,
            "rth_mosfet_ja": rth_mos,
            "rth_diode_ja": spec.rth_diode_ja,
            "rth_inductor": spec.rth_inductor,
            "converged": spec.thermal_model,
        }

        # 4. Shared component losses
        br_loss = self.bridge_loss.compute(spec, op)
        cap_loss = self.capacitor.compute(spec, op)

        # 5. Explicit aggregation
        #    P_total = n_phases * (P_ind_phase + P_mos_phase + P_dio_phase)
        #            + P_bridge_total + P_capacitor_total
        n = spec.n_phases
        p_per_phase = ind_loss.power_loss_W + mos_loss.power_loss_W + dio_loss.power_loss_W
        p_shared = br_loss.power_loss_W + cap_loss.power_loss_W
        total_loss = n * p_per_phase + p_shared
        efficiency = spec.pout_total / (spec.pout_total + total_loss)

        # Breakdown with explicit _total tags: the values below are already
        # the n-phase totals (sub_losses are per phase), so the keys must not
        # say "per_phase" — that misleads readers into multiplying by n again.
        breakdown = {}
        for label, val in ind_loss.sub_losses.items():
            breakdown[f"Inductor_{label}_total"] = n * val
        for label, val in mos_loss.sub_losses.items():
            breakdown[f"MOSFET_{label}_total"] = n * val
        for label, val in dio_loss.sub_losses.items():
            breakdown[f"Diode_{label}_total"] = n * val
        for label, val in br_loss.sub_losses.items():
            breakdown[f"Bridge_{label}_total"] = val
        for label, val in cap_loss.sub_losses.items():
            breakdown[f"Capacitor_{label}_total"] = val

        return {
            "spec": spec,
            "op": op,
            "mosfet": mosfet,
            "diode": diode,
            "inductor_design": design,
            "inductor_metrics": inductor_metrics,
            "losses": {
                "inductor": ind_loss,
                "mosfet": mos_loss,
                "diode": dio_loss,
                "bridge": br_loss,
                "capacitor": cap_loss,
            },
            "trace": trace,
            "per_phase_loss": p_per_phase,
            "shared_loss": p_shared,
            "total_loss": total_loss,
            "efficiency": efficiency,
            "thermal": thermal,
            "breakdown": breakdown,
        }


def compute_inductor_metrics(design: InductorDesign, trace, spec: DesignSpec) -> dict:
    """Actual-vs-target ripple metrics for an inductor design and its trace.

    Shared by the bridge-boost analyzer and the totem-pole analyzer so
    both topologies report the inductor health identically.
    """
    idx_peak = np.argmin(np.abs(trace.theta - np.pi / 2))
    delta_i_pp_actual = float(trace.delta_i_pp[idx_peak])
    dm = design.design_metadata
    target_ripple_peak_basis = spec.ripple_ratio
    actual_ripple_peak_basis = delta_i_pp_actual / trace.iin_pk_phase
    delta_i_pp_ref = dm.get("DeltaI_pp_ref", 0.0)

    L_eff = design.L_eff_at_ipeak_uh
    L_eff_ratio = L_eff / design.L_target_uh if design.L_target_uh > 0 else 0.0
    actual_ripple_margin = (
        actual_ripple_peak_basis / target_ripple_peak_basis
        if target_ripple_peak_basis > 0 else float('inf')
    )
    actual_ripple_exceeds_target = actual_ripple_margin > 1.0

    if actual_ripple_margin <= 1.10:
        ripple_risk_level = "ok"
    elif actual_ripple_margin <= 1.25:
        ripple_risk_level = "warning"
    elif actual_ripple_margin <= 1.50:
        ripple_risk_level = "high_warning"
    else:
        ripple_risk_level = "high_risk"

    return {
        "L_target_uH": design.L_target_uh,
        "L_eff_at_ipeak_uH": L_eff,
        "L_eff_ratio": L_eff_ratio,
        "target_ripple_ratio_peak_basis": target_ripple_peak_basis,
        "actual_ripple_ratio_peak_basis": actual_ripple_peak_basis,
        "delta_i_pp_ref_A": delta_i_pp_ref,
        "delta_i_pp_actual_A": delta_i_pp_actual,
        "actual_ripple_margin": actual_ripple_margin,
        "actual_ripple_exceeds_target": actual_ripple_exceeds_target,
        "actual_ripple_risk_level": ripple_risk_level,
        "Iin_pk_phase_A": trace.iin_pk_phase,
        "trace_uses_L": "L_eff_at_ipeak_uh",
    }
