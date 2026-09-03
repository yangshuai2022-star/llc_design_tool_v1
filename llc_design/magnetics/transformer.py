"""LLC transformer synthesis with iGSE core loss and layered Litz loss.

The default physical stack is P/2-S-P/2.  Copper loss is computed from the
actual stack MMF rather than a layer-count multiplier.  Primary load ampere-
turns and secondary ampere-turns cancel inside the stack; magnetizing current,
strand skin effect, residual external field, bundle transposition and terminal
resistance remain explicitly visible in the loss breakdown.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Iterable

from ..core.operating_point import LLCOperatingPoint
from ..core.spec import LLCDesignSpec
from ..core.tank import TankDesign
from .core import CoreDatabase, CoreSpec
from .litz import (LitzWire, StackLayer, WindingLossBreakdown,
                   dc_resistance_ohm, distribute_turns,
                   layered_litz_stack_loss, select_litz_wire, winding_layers)
from .magnetic_waveforms import (transformer_current_waveforms,
                                 transformer_flux_waveform)


MU0 = 4.0e-7 * math.pi


@dataclass(frozen=True)
class TransformerLoss:
    core_w: float
    primary_copper_w: float
    secondary_copper_w: float
    total_w: float
    b_peak_t: float
    primary_ac_factor: float
    secondary_ac_factor: float
    b_peak_min_area_t: float = 0.0
    primary_dc_w: float = 0.0
    primary_skin_w: float = 0.0
    primary_proximity_w: float = 0.0
    primary_bundle_w: float = 0.0
    primary_termination_w: float = 0.0
    secondary_dc_w: float = 0.0
    secondary_skin_w: float = 0.0
    secondary_proximity_w: float = 0.0
    secondary_bundle_w: float = 0.0
    secondary_termination_w: float = 0.0
    estimated_hotspot_c: float = 0.0
    material_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class TransformerCandidateSummary:
    rank: int
    part_number: str
    family: str
    material: str
    feasible: bool
    nominal_total_loss_w: float
    nominal_core_loss_w: float
    nominal_primary_copper_w: float
    nominal_secondary_copper_w: float
    nominal_hotspot_c: float
    fill_factor: float
    radial_build_mm: float
    worst_b_peak_t: float
    gap_total_mm: float
    cost_usd: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class TransformerDesign:
    core: CoreSpec
    primary_turns: int
    secondary_turns: int
    primary_wire: LitzWire
    secondary_wire: LitzWire
    primary_layers_per_half: int
    secondary_layers: int
    primary_turns_per_layer: int
    secondary_turns_per_layer: int
    fill_factor: float
    radial_build_mm: float
    gap_total_mm: float
    primary_rdc_ohm: float
    secondary_rdc_ohm: float
    worst_b_peak_t: float
    feasible: bool
    reasons: tuple[str, ...]
    alternatives: tuple[TransformerCandidateSummary, ...] = ()

    @property
    def turns_ratio(self) -> float:
        return self.primary_turns / self.secondary_turns

    def _winding_stack(self, spec: LLCDesignSpec, op: LLCOperatingPoint) -> tuple[StackLayer, ...]:
        samples = spec.magnetic_waveform_samples
        primary, secondary, _ = transformer_current_waveforms(spec, op, samples)
        primary_tuple = tuple(float(x) for x in primary)
        secondary_tuple = tuple(float(x) for x in secondary)

        p1_turns = self.primary_turns // 2
        p2_turns = self.primary_turns - p1_turns
        p1_layers = distribute_turns(p1_turns, self.primary_turns_per_layer)
        p2_layers = distribute_turns(p2_turns, self.primary_turns_per_layer)
        s_layers = distribute_turns(self.secondary_turns, self.secondary_turns_per_layer)
        stack: list[StackLayer] = []
        for turns in p1_layers:
            stack.append(StackLayer(
                "primary", turns, turns * self.core.mlt_primary_mm * 1e-3,
                self.primary_wire, primary_tuple))
        for turns in s_layers:
            stack.append(StackLayer(
                "secondary", turns, turns * self.core.mlt_secondary_mm * 1e-3,
                self.secondary_wire, secondary_tuple))
        for turns in p2_layers:
            stack.append(StackLayer(
                "primary", turns, turns * self.core.mlt_primary_mm * 1e-3,
                self.primary_wire, primary_tuple))
        return tuple(stack)

    def _loss_at_temperature(self, spec: LLCDesignSpec, op: LLCOperatingPoint,
                             temperature_c: float) -> tuple[float, WindingLossBreakdown, WindingLossBreakdown,
                                                             float, float, tuple[str, ...]]:
        t, b = transformer_flux_waveform(
            op, self.primary_turns, self.core.ae_m2, spec.magnetic_waveform_samples)
        b_peak = float(max(abs(b.min()), abs(b.max())))
        b_peak_min = b_peak * self.core.ae_m2 / self.core.amin_m2
        p_core = self.core.core_loss_waveform_w(t, b, temperature_c)
        stack_loss = layered_litz_stack_loss(
            self._winding_stack(spec, op), op.switching_frequency_hz,
            self.core.window_width_m, temperature_c,
            max_harmonic=spec.litz_max_harmonic,
            transposition_quality=spec.litz_transposition_quality,
            sub_bundle_coupling_factor=spec.litz_sub_bundle_coupling_factor,
            termination_resistance_fraction=spec.winding_termination_resistance_fraction,
            calibration_factor=spec.litz_proximity_correction * spec.transformer_proximity_severity,
        )
        primary = stack_loss["primary"]
        secondary = stack_loss["secondary"]
        return (p_core, primary, secondary, b_peak, b_peak_min,
                self.core.loss_range_warnings(op.switching_frequency_hz, b_peak))

    def loss(self, spec: LLCDesignSpec, op: LLCOperatingPoint) -> TransformerLoss:
        temperature = max(spec.winding_temperature_c, spec.ambient_temperature_c)
        last = None
        for _ in range(spec.magnetic_thermal_max_iterations):
            last = self._loss_at_temperature(spec, op, temperature)
            p_core, primary, secondary, *_ = last
            predicted = spec.ambient_temperature_c + (
                p_core + primary.total_w + secondary.total_w) * spec.transformer_rth_k_per_w
            predicted = min(max(predicted, spec.ambient_temperature_c), 220.0)
            updated = 0.55 * temperature + 0.45 * predicted
            if abs(updated - temperature) <= spec.magnetic_thermal_tolerance_c:
                temperature = updated
                break
            temperature = updated
        assert last is not None
        p_core, primary, secondary, b_peak, b_peak_min, warnings = self._loss_at_temperature(
            spec, op, temperature)
        total = p_core + primary.total_w + secondary.total_w
        return TransformerLoss(
            core_w=p_core,
            primary_copper_w=primary.total_w,
            secondary_copper_w=secondary.total_w,
            total_w=total,
            b_peak_t=b_peak,
            primary_ac_factor=primary.effective_ac_factor,
            secondary_ac_factor=secondary.effective_ac_factor,
            b_peak_min_area_t=b_peak_min,
            primary_dc_w=primary.dc_copper_w,
            primary_skin_w=primary.skin_effect_w,
            primary_proximity_w=primary.external_proximity_w,
            primary_bundle_w=primary.bundle_circulating_w,
            primary_termination_w=primary.termination_w,
            secondary_dc_w=secondary.dc_copper_w,
            secondary_skin_w=secondary.skin_effect_w,
            secondary_proximity_w=secondary.external_proximity_w,
            secondary_bundle_w=secondary.bundle_circulating_w,
            secondary_termination_w=secondary.termination_w,
            estimated_hotspot_c=temperature,
            material_warnings=warnings,
        )


def _build_for_core(spec: LLCDesignSpec, tank: TankDesign,
                    operating_points: Iterable[LLCOperatingPoint],
                    core: CoreSpec) -> TransformerDesign:
    ops = tuple(operating_points)
    max_ip = max(op.resonant_current_rms_a for op in ops)
    max_is = max(op.secondary_current_rms_a for op in ops)
    primary_wire = select_litz_wire(
        max_ip, spec.litz_strand_copper_diameter_m,
        spec.litz_strand_outer_diameter_m, spec.litz_packing_factor,
        spec.litz_current_density_target_a_per_mm2)
    secondary_wire = select_litz_wire(
        max_is, spec.litz_strand_copper_diameter_m,
        spec.litz_strand_outer_diameter_m, spec.litz_packing_factor,
        spec.litz_current_density_target_a_per_mm2)

    half_turns = math.ceil(spec.primary_turns / 2)
    p_layers, p_tpl = winding_layers(half_turns, primary_wire,
                                     core.window_width_mm)
    s_layers, s_tpl = winding_layers(spec.secondary_turns, secondary_wire,
                                     core.window_width_mm)

    fill_area = (spec.primary_turns * primary_wire.envelope_area_m2 * 1e6
                 + spec.secondary_turns * secondary_wire.envelope_area_m2 * 1e6
                 + spec.transformer_insulation_area_mm2)
    fill_factor = fill_area / core.aw_mm2
    radial_build = (2.0 * p_layers * primary_wire.equivalent_outer_diameter_mm
                    + s_layers * secondary_wire.equivalent_outer_diameter_mm
                    + 1.2)

    length_p = spec.primary_turns * core.mlt_primary_mm * 1e-3
    length_s = spec.secondary_turns * core.mlt_secondary_mm * 1e-3
    rdc_p = dc_resistance_ohm(primary_wire, length_p,
                             spec.winding_temperature_c)
    rdc_s = dc_resistance_ohm(secondary_wire, length_s,
                             spec.winding_temperature_c)

    b_values = [op.transformer_square_equivalent_v /
                (4.0 * spec.primary_turns * core.amin_m2
                 * op.switching_frequency_hz) for op in ops]
    worst_b = max(b_values)
    gap_m = (MU0 * spec.primary_turns**2 * core.ae_m2 / tank.lm_h
             - core.le_m / core.mu_r)
    gap_mm = max(gap_m, 0.0) * 1e3

    reasons: list[str] = []
    if fill_factor > spec.transformer_max_fill_factor:
        reasons.append(f"window fill {fill_factor:.3f} exceeds {spec.transformer_max_fill_factor:.3f}")
    if radial_build > core.window_height_mm:
        reasons.append(f"radial build {radial_build:.1f} mm exceeds {core.window_height_mm:.1f} mm")
    if worst_b > spec.transformer_max_b_t:
        reasons.append(f"Bpk at minimum area {worst_b:.3f} T exceeds {spec.transformer_max_b_t:.3f} T")
    if worst_b > 0.70 * core.saturation_flux_at(spec.winding_temperature_c):
        reasons.append("Bpk exceeds 70% of temperature-adjusted saturation flux")
    if gap_m < 0:
        reasons.append("required Lm exceeds ungapped-core estimate")
    if gap_mm > spec.transformer_max_gap_mm:
        reasons.append(f"total gap {gap_mm:.2f} mm exceeds {spec.transformer_max_gap_mm:.2f} mm")

    return TransformerDesign(
        core=core, primary_turns=spec.primary_turns,
        secondary_turns=spec.secondary_turns,
        primary_wire=primary_wire, secondary_wire=secondary_wire,
        primary_layers_per_half=p_layers, secondary_layers=s_layers,
        primary_turns_per_layer=p_tpl, secondary_turns_per_layer=s_tpl,
        fill_factor=fill_factor, radial_build_mm=radial_build,
        gap_total_mm=gap_mm, primary_rdc_ohm=rdc_p,
        secondary_rdc_ohm=rdc_s, worst_b_peak_t=worst_b,
        feasible=not reasons, reasons=tuple(reasons))


def design_transformer(spec: LLCDesignSpec, tank: TankDesign,
                       operating_points: Iterable[LLCOperatingPoint],
                       database: CoreDatabase | None = None,
                       preferred_core: str | None = None) -> TransformerDesign:
    db = database or CoreDatabase()
    cores = [db.get(preferred_core)] if preferred_core else db.for_purpose("transformer", spec.transformer_core_families)
    ops = tuple(operating_points)
    designs = [_build_for_core(spec, tank, ops, core) for core in cores]
    feasible = [d for d in designs if d.feasible]
    candidates = feasible or designs
    nominal = min(ops, key=lambda op: abs(op.vbus_v - spec.vbus_nom_v)
                  + 100.0 * abs(op.load_fraction - 1.0))
    evaluated = [(design, design.loss(spec, nominal)) for design in candidates]
    evaluated.sort(key=lambda item: (not item[0].feasible, item[1].total_w,
                                     item[0].fill_factor, item[0].core.cost_usd))
    summaries = tuple(TransformerCandidateSummary(
        rank=index + 1, part_number=design.core.part_number,
        family=design.core.family, material=design.core.material,
        feasible=design.feasible, nominal_total_loss_w=loss.total_w,
        nominal_core_loss_w=loss.core_w,
        nominal_primary_copper_w=loss.primary_copper_w,
        nominal_secondary_copper_w=loss.secondary_copper_w,
        nominal_hotspot_c=loss.estimated_hotspot_c,
        fill_factor=design.fill_factor, radial_build_mm=design.radial_build_mm,
        worst_b_peak_t=design.worst_b_peak_t, gap_total_mm=design.gap_total_mm,
        cost_usd=design.core.cost_usd, reasons=design.reasons,
    ) for index, (design, loss) in enumerate(evaluated[:12]))
    return replace(evaluated[0][0], alternatives=summaries)
