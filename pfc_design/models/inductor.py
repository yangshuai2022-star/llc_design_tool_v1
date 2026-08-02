"""Inductor design and loss model for two-phase interleaved PFC."""

import numpy as np

from ..core.spec import DesignSpec
from ..core.operating_point import OperatingPoint
from ..core.constants import rho_cu, MU0, UH_TO_H, CM2_TO_M2
from ..magnetics.core_entry import CoreSpec
from ..magnetics.core_database import CoreDatabase
from ..magnetics.winding import (
    skin_effect_factor, proximity_factor, dc_resistance, skin_depth
)
from ..magnetics.saturation import (
    effective_inductance, calculate_b_max, calculate_b_ac, h_dc_oersted,
    percent_permeability
)
from .base import LossResult

# Ripple-spec match: the design counts as meeting the target when
# L_eff(I_peak) reaches 98% of the inductance implied by the spec.
L_TARGET_MATCH_TOL = 0.02


def _ripple_peak_current(iin_pk_phase: float, vin_pk_min: float, d_pk: float,
                         fsw: float, leff_uh: float) -> float:
    """Peak current = line peak + half the ripple driven by the drooped L_eff.

    il_peak = Iin_pk + Vin_pk*D_pk / (2 * L_eff * fsw)

    Using the realized (DC-bias-drooped) inductance instead of the target
    keeps the saturation guard honest: the old loop judged B against the
    target ripple even when L_eff collapsed to ~60% of target.
    """
    return iin_pk_phase + vin_pk_min * d_pk / (2.0 * leff_uh * UH_TO_H * fsw)


class InductorDesign:
    """Result of inductor design calculations."""

    def __init__(self, core: CoreSpec, n_turns: float, L_target_uh: float,
                 L_noload_uh: float, L_eff_at_ipeak_uh: float,
                 wire_d_mm: float, n_parallel: int, n_cores: int = 1,
                 design_metadata: dict | None = None):
        self.core = core
        self.n_turns = n_turns
        self.L_target_uh = L_target_uh
        self.L_noload_uh = L_noload_uh
        self.L_eff_at_ipeak_uh = L_eff_at_ipeak_uh
        self.wire_d_mm = wire_d_mm
        self.n_parallel = n_parallel
        self.n_cores = n_cores
        self.design_metadata = design_metadata or {}

    @property
    def kw(self) -> float:
        a_wire_mm2 = np.pi * (self.wire_d_mm / 2) ** 2 * self.n_parallel
        a_cu_total_mm2 = a_wire_mm2 * self.n_turns
        aw_mm2 = self.core.aw_cm2 * 100
        return a_cu_total_mm2 / aw_mm2 if aw_mm2 > 0 else 1.0

    @property
    def ae_total_cm2(self) -> float:
        return self.core.ae_cm2 * self.n_cores

    @property
    def ve_total_cm3(self) -> float:
        return self.core.ve_cm3 * self.n_cores


class InductorDesigner:
    """Designs a PFC boost inductor."""

    def __init__(self, db: CoreDatabase | None = None):
        self.db = db or CoreDatabase()

    def design(self, spec: DesignSpec, op: OperatingPoint,
               preferred_core: CoreSpec | None = None,
               n_cores: int = 1) -> InductorDesign:
        """Design the inductor."""

        # Step 1: Calculate required inductance
        if spec.L_target is not None:
            L_target_uh = spec.L_target
        else:
            L_target_uh = self._calculate_L(spec, op)

        # Intermediate values for metadata
        vin_pk_min = np.sqrt(2) * op.vin_rms
        d_pk = np.clip(1.0 - vin_pk_min / spec.vout, 0.05, 0.95)
        iin_rms_phase = spec.pout_total / spec.n_phases / spec.eta_target / op.vin_rms
        iin_pk_phase = np.sqrt(2) * iin_rms_phase
        delta_i_ref = spec.ripple_ratio * iin_pk_phase

        # IL_peak = line-frequency peak + half the switching ripple.
        # Starts from the target ripple; the design loop below re-derives
        # il_peak from the realized L_eff each iteration (fixed point) so
        # the saturation guard always sees the true peak current.
        il_peak = iin_pk_phase + delta_i_ref / 2.0

        # Step 2: Select core
        if preferred_core is not None:
            core = preferred_core
        else:
            core = self._select_core(spec, op, L_target_uh)

        # Step 3: Calculate turns from L_target and AL
        al_total = core.al_nH_per_t2 * n_cores
        n_al = np.sqrt(L_target_uh * 1000 / al_total) if al_total > 0 else 50
        n_turns = round(n_al)

        # Step 4: Select wire (independent of turns)
        wire_d_mm, n_parallel = self._select_wire(spec, op, n_turns)

        # Step 5-6: Iterate turns so the *effective* inductance at the
        # peak current honors the ripple spec, subject to B_max < 0.7*Bsat
        # and window fill < 60%. L_eff(I_peak) is unimodal in the droop
        # region (rises then falls as turns push deeper into saturation),
        # so we scan upward and keep the best feasible candidate.
        # The DC-bias droop (L_eff < L_noload) is exactly what the old
        # loop ignored — it designed to the no-load inductance and the
        # actual ripple came out above spec (the high_risk finding).
        safe_b = core.bs_T * 0.7
        a_wire_mm2 = np.pi * (wire_d_mm / 2) ** 2 * n_parallel
        aw_mm2 = core.aw_cm2 * 100
        l_eff_target = L_target_uh * (1.0 - L_TARGET_MATCH_TOL)
        best_n, best_leff = n_turns, -1.0
        fallback_n, fallback_b = n_turns, float('inf')
        prev_leff = -1.0
        limited_by = "iteration_cap"
        for _ in range(200):
            l0_uh = core.al_nH_per_t2 * n_cores * n_turns ** 2 / 1000
            # Honest ripple: recompute il_peak from the drooped L_eff so the
            # B-guard sees the actual peak current, not the target ripple.
            leff = effective_inductance(l0_uh, n_turns, il_peak,
                                        core.le_cm, core.dc_bias_coeffs)
            il_peak = _ripple_peak_current(
                iin_pk_phase, vin_pk_min, d_pk, spec.fsw, leff)
            leff = effective_inductance(l0_uh, n_turns, il_peak,
                                        core.le_cm, core.dc_bias_coeffs)
            b_peak = calculate_b_max(leff, il_peak, n_turns,
                                     core.ae_cm2 * n_cores)
            kw = a_wire_mm2 * n_turns / aw_mm2 if aw_mm2 > 0 else 1.0

            if b_peak < fallback_b:
                fallback_n, fallback_b = n_turns, b_peak
            feasible_turns = b_peak <= safe_b and kw <= 0.60
            if feasible_turns and leff > best_leff:
                best_n, best_leff = n_turns, leff
            if feasible_turns and leff >= l_eff_target:
                limited_by = "none"
                break
            if b_peak <= safe_b and leff < prev_leff * 1.002:
                # B-safe region, L_eff past its droop peak: no better N exists
                limited_by = "droop_peak"
                break
            prev_leff = leff
            n_turns += 1

        if best_leff < 0:
            # No B-safe N exists in the scan: return the B-max-minimizing N
            # (deepest droop), which the sweep will flag as infeasible.
            n_turns = int(fallback_n)
        else:
            n_turns = int(best_n)
        al_total = core.al_nH_per_t2 * n_cores
        L_noload_uh = al_total * n_turns ** 2 / 1000

        # Re-resolve the self-consistent operating point for the chosen
        # design: il_peak and L_eff(I_peak) must agree with each other.
        for _ in range(8):
            L_eff_peak = effective_inductance(L_noload_uh, n_turns, il_peak,
                                              core.le_cm, core.dc_bias_coeffs)
            il_peak = _ripple_peak_current(
                iin_pk_phase, vin_pk_min, d_pk, spec.fsw, L_eff_peak)
        L_eff_peak = effective_inductance(L_noload_uh, n_turns, il_peak,
                                          core.le_cm, core.dc_bias_coeffs)
        # Report values in the definition-consistent form: the actual ripple
        # implied by the final drooped L_eff, and the peak current it gives.
        delta_i_pp_actual = (vin_pk_min * d_pk
                             / (L_eff_peak * UH_TO_H * spec.fsw))
        il_peak = iin_pk_phase + delta_i_pp_actual / 2.0
        L_eff = effective_inductance(L_noload_uh, n_turns, op.iin_rms,
                                     core.le_cm, core.dc_bias_coeffs)
        b_peak = calculate_b_max(L_eff_peak, il_peak, n_turns,
                                 core.ae_cm2 * n_cores)
        l_eff_target_met = L_eff_peak >= l_eff_target

        # Step 7: (wire already selected)

        return InductorDesign(
            core=core, n_turns=n_turns,
            L_target_uh=L_target_uh,
            L_noload_uh=L_noload_uh,
            L_eff_at_ipeak_uh=L_eff_peak,
            wire_d_mm=wire_d_mm,
            n_parallel=n_parallel,
            n_cores=n_cores,
            design_metadata={
                "Vin_pk_min": vin_pk_min,
                "D_at_line_peak": d_pk,
                "Iin_rms_phase": iin_rms_phase,
                "Iin_pk_phase": iin_pk_phase,
                "IL_peak_with_ripple": il_peak,
                "DeltaI_pp_ref": delta_i_ref,
                "DeltaI_pp_actual": delta_i_pp_actual,
                "ripple_definition": "ripple_ratio = DeltaI_pp_at_line_peak / Iin_pk_phase",
                "L_eff_target_uh": l_eff_target,
                "L_eff_ratio_to_target": (L_eff_peak / L_target_uh
                                          if L_target_uh > 0 else 0.0),
                "L_eff_target_met": l_eff_target_met,
                "l_eff_limited_by": limited_by,
                "design_loop": "L_eff(I_peak) targeted, Bmax/window guarded",
            },
        )

    def _calculate_L(self, spec: DesignSpec, op: OperatingPoint) -> float:
        """Boost PFC inductor sizing formula.

        L = Vin_pk_min * D_pk / (fsw * DeltaI_pp_ref)

        where ripple_ratio = DeltaI_pp_at_line_peak / Iin_pk_phase.
        """
        vin_pk_min = np.sqrt(2) * op.vin_rms
        d_pk = np.clip(1.0 - vin_pk_min / spec.vout, 0.05, 0.95)
        iin_rms_phase = spec.pout_total / spec.n_phases / spec.eta_target / op.vin_rms
        iin_pk_phase = np.sqrt(2) * iin_rms_phase
        delta_i_ref = spec.ripple_ratio * iin_pk_phase
        L_target_H = vin_pk_min * d_pk / (spec.fsw * delta_i_ref)
        return L_target_H * 1e6  # H → μH

    def _required_ae_n(self, spec: DesignSpec, op: OperatingPoint) -> float:
        """Compute required Ae*N product (cm²) to keep B_avg below target.

        From Mathcad: integrate(Vac(t)*D(t)/fsw, t=0..0.01) / (0.01 * Bcore_avg)
        """
        b_target = spec.B_max_target
        # Average voltage-time product over half line cycle
        v_d_product = op.vac_t * op.duty_t
        v_s_per_cycle = np.trapezoid(v_d_product, op.t) / spec.fsw
        half_period = 0.5 / spec.f_line
        return v_s_per_cycle / (half_period * b_target) * 1e4  # m²→cm²

    def _select_core(self, spec: DesignSpec, op: OperatingPoint,
                     L_uh: float) -> CoreSpec:
        ae_n = self._required_ae_n(spec, op)
        ae_min = ae_n / 60  # assume ~60 turns

        candidates = self.db.query(
            material_class=spec.core_material_pref,
            ae_min_cm2=ae_min * 0.5,
            max_results=20
        )

        if not candidates:
            candidates = self.db.query(max_results=10)

        if not candidates:
            raise ValueError("No suitable core found in database")

        # Prefer ~35.8mm OD Sendust (Mathcad reference)
        for c in candidates:
            if 33 <= c.od_mm <= 37 and c.material_class == spec.core_material_pref:
                return c
        return candidates[0]

    def _select_wire(self, spec: DesignSpec, op: OperatingPoint,
                     n_turns: float) -> tuple[float, int]:
        a_cu_mm2 = op.iin_rms / spec.J_max
        std_diameters = [0.8, 1.0, 1.2, 1.5, 1.8, 2.0]
        for d in std_diameters:
            a_wire = np.pi * (d / 2) ** 2
            n_par = max(1, int(np.ceil(a_cu_mm2 / a_wire)))
            if n_par <= 4:
                return d, n_par
        return 2.0, max(1, int(np.ceil(a_cu_mm2 / (np.pi * 1.0 ** 2))))


class InductorLoss:
    """Computes inductor losses: core + copper."""

    def __init__(self, db: CoreDatabase | None = None):
        self.db = db or CoreDatabase()

    def compute(self, design: InductorDesign, spec: DesignSpec,
                op: OperatingPoint, trace=None,
                t_winding: float | None = None) -> LossResult:
        """Compute inductor losses: core + copper.

        Args:
            design: inductor design result
            spec: design specification
            op: operating point
            trace: optional LineCycleTrace for line-cycle integration
            t_winding: winding temperature in °C (defaults to
                       spec.t_ambient + 40, the legacy assumption)
        """
        core = design.core
        n_cores = design.n_cores
        ae_total = core.ae_cm2 * n_cores
        ve_total = core.ve_cm3 * n_cores

        if trace is not None:
            p_fe = self._core_loss_line_cycle(design, spec, trace)
            p_cu_lf, p_cu_hf = self._copper_loss_split(design, spec, trace,
                                                       t_winding=t_winding)
            p_cu = p_cu_lf + p_cu_hf
            b_max = self._bmax_line_cycle(design, trace)
            loss_model = "line_cycle"
            model_confidence = "medium"
            sub_losses = {"core": p_fe, "copper_lf": p_cu_lf, "copper_hf": p_cu_hf}
            extra_meta = {
                "B_max_line_cycle_T": b_max,
                "copper_LF_W": p_cu_lf,
                "copper_HF_W": p_cu_hf,
                "winding_ac_model": "simplified_skin_proximity",
                "winding_confidence": "medium",
                "saturation_model": "simplified",
            }
        else:
            p_fe = self._core_loss_legacy(design, spec, op)
            p_cu = self._copper_loss_legacy(design, spec, op,
                                            t_winding=t_winding)
            loss_model = "legacy_single_point"
            model_confidence = "low"
            b_max = calculate_b_max(design.L_eff_at_ipeak_uh, op.iin_peak,
                                    design.n_turns, ae_total)
            sub_losses = {"core": p_fe, "copper": p_cu}
            extra_meta = {}

        metadata = {
            "core_part": core.part_number,
            "core_material": core.material,
            "n_cores": n_cores,
            "turns": design.n_turns,
            "L_target_uH": design.L_target_uh,
            "L_noload_uH": design.L_noload_uh,
            "L_eff_uH": design.L_eff_at_ipeak_uh,
            "B_max_T": b_max,
            "saturation_pct": (design.L_eff_at_ipeak_uh / design.L_noload_uh * 100
                               if design.L_noload_uh > 0 else 0),
            "window_fill": design.kw,
            "loss_model": loss_model,
            "model_confidence": model_confidence,
            **extra_meta,
        }

        return LossResult(
            component="Inductor (per phase)",
            power_loss_W=p_fe + p_cu,
            sub_losses=sub_losses,
            metadata=metadata,
        )

    # ── Legacy methods (kept for comparison) ────────────────────────

    def _core_loss_legacy(self, design: InductorDesign, spec: DesignSpec,
                          op: OperatingPoint) -> float:
        core = design.core
        ae_total = core.ae_cm2 * design.n_cores
        ve_total = core.ve_cm3 * design.n_cores

        vac_peak = op.vm
        d_peak = 1.0 - vac_peak / spec.vout
        delta_i = vac_peak * d_peak / (design.L_eff_at_ipeak_uh * UH_TO_H * spec.fsw)
        b_ac = calculate_b_ac(design.L_eff_at_ipeak_uh, delta_i,
                              design.n_turns, ae_total)

        st = self.db.get_steinmetz(core.material)
        if st is None:
            st = self.db.get_steinmetz(core.material_class)

        if st is not None and b_ac > 0:
            ve_m3 = ve_total * 1e-6
            return st.core_loss(spec.fsw, b_ac, ve_m3, method='ose')

        b_avg_gauss = calculate_b_max(design.L_eff_at_ipeak_uh,
                                      op.iin_peak * 2 / np.pi,
                                      design.n_turns, ae_total) * 1e4
        pv_W_cm3 = 3.89e-3 * (b_avg_gauss / 1000) ** 2.57 * (spec.fsw / 1000) ** 1.11
        return float(pv_W_cm3 * ve_total)

    def _copper_loss_legacy(self, design: InductorDesign, spec: DesignSpec,
                            op: OperatingPoint,
                            t_winding: float | None = None) -> float:
        core = design.core
        a_wire_mm2 = np.pi * (design.wire_d_mm / 2) ** 2 * design.n_parallel
        a_wire_m2 = a_wire_mm2 * 1e-6
        mlt_m = core.mlt_cm / 100
        total_length_m = design.n_turns * mlt_m
        if t_winding is None:
            t_winding = spec.t_ambient + 40
        rho = rho_cu(t_winding)
        rdc = dc_resistance(rho, total_length_m, a_wire_m2)
        d_wire_m = design.wire_d_mm / 1000
        delta = skin_depth(spec.fsw, t_winding)
        f_skin = skin_effect_factor(d_wire_m, delta)
        n_layers_est = 1
        f_prox = proximity_factor(n_layers_est, d_wire_m, delta)
        # Dowell F_R (f_prox) includes skin for multi-layer windings; the
        # exact round-wire skin factor applies to the single-layer case.
        f_ac = f_prox if n_layers_est > 1 else f_skin
        r_eff = spec.ripple_ratio
        i_rms_total = op.iin_rms * np.sqrt(1 + (r_eff / 3) ** 2)
        return float(i_rms_total ** 2 * rdc * f_ac)

    # ── Line-cycle core loss (OSE per angle, half-cycle averaged) ──

    def _core_loss_line_cycle(self, design: InductorDesign, spec: DesignSpec,
                              trace) -> float:
        """Line-cycle-integrated OSE core loss.

        For each theta:  delta_B_pp = L_eff * delta_i_pp / (N * Ae)
                         B_ac_peak = delta_B_pp / 2   ← documented definition
                         pv(theta) = k * fsw^alpha * B_ac_peak^beta
        Then: Pcore = Ve * average_over_half_cycle(pv, theta)
        """
        from ..core.line_cycle import average_over_half_cycle

        st = self.db.get_steinmetz(design.core.material)
        if st is None:
            st = self.db.get_steinmetz(design.core.material_class)
        if st is None:
            return self._core_loss_legacy(design, spec, trace.op)

        ae_m2 = design.ae_total_cm2 * 1e-4
        ve_m3 = design.ve_total_cm3 * 1e-6
        L_eff_H = design.L_eff_at_ipeak_uh * 1e-6

        delta_B_pp = L_eff_H * trace.delta_i_pp / (design.n_turns * ae_m2)
        B_ac_peak = delta_B_pp / 2.0  # B_ac_peak = DeltaB_pp / 2

        pv_density = st.k * spec.fsw ** st.alpha * B_ac_peak ** st.beta
        pcore_mean = average_over_half_cycle(pv_density, trace.theta)
        return float(pcore_mean * ve_m3)

    # ── Line-cycle Bmax (max of instantaneous B over half cycle) ──

    def _bmax_line_cycle(self, design: InductorDesign, trace) -> float:
        """Bmax = max( L_eff * I_peak(theta) / (N * Ae) ) over half cycle."""
        L_eff_H = design.L_eff_at_ipeak_uh * 1e-6
        ae_m2 = design.ae_total_cm2 * 1e-4
        B_inst = L_eff_H * trace.i_peak / (design.n_turns * ae_m2)
        return float(np.max(B_inst))

    # ── Split copper loss: LF (Rdc) + HF (Rac) ─────────────────────

    def _copper_loss_split(self, design: InductorDesign, spec: DesignSpec,
                           trace, t_winding: float | None = None
                           ) -> tuple[float, float]:
        """Return (Pcu_lf, Pcu_hf).

        Pcu_lf = I_line_rms_phase^2 * Rdc(T)
        Pcu_hf = mean(delta_i_pp^2/12) * Rac(fsw, geometry, T)
        """
        from ..core.line_cycle import average_over_half_cycle

        core = design.core
        a_wire_mm2 = np.pi * (design.wire_d_mm / 2) ** 2 * design.n_parallel
        a_wire_m2 = a_wire_mm2 * 1e-6
        mlt_m = core.mlt_cm / 100
        total_length_m = design.n_turns * mlt_m

        if t_winding is None:
            t_winding = spec.t_ambient + 40
        rho = rho_cu(t_winding)
        rdc = dc_resistance(rho, total_length_m, a_wire_m2)

        d_wire_m = design.wire_d_mm / 1000
        delta = skin_depth(spec.fsw, t_winding)
        f_skin = skin_effect_factor(d_wire_m, delta)
        n_layers_est = 1
        f_prox = proximity_factor(n_layers_est, d_wire_m, delta)
        rac = rdc * (f_prox if n_layers_est > 1 else f_skin)

        # LF: line-frequency RMS^2 * Rdc
        i_lf_rms_sq = average_over_half_cycle(trace.i_avg_phase ** 2, trace.theta)
        pcu_lf = float(i_lf_rms_sq * rdc)

        # HF: mean(delta_i_pp^2 / 12) * Rac
        i_hf_rms_sq = average_over_half_cycle(trace.delta_i_pp ** 2 / 12.0, trace.theta)
        pcu_hf = float(i_hf_rms_sq * rac)

        return pcu_lf, pcu_hf
