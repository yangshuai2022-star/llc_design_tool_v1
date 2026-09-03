"""External LLC resonant-inductor synthesis with two-layer MMF and gap loss."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Iterable

from ..core.operating_point import LLCOperatingPoint
from ..core.spec import LLCDesignSpec
from ..core.tank import TankDesign
from .core import CoreDatabase, CoreSpec
from .litz import (LitzWire, StackLayer, dc_resistance_ohm,
                   distribute_turns, harmonic_rms_spectrum,
                   layered_litz_stack_loss, select_litz_wire, winding_layers)
from .magnetic_waveforms import resonant_inductor_waveforms


MU0 = 4.0e-7 * math.pi


@dataclass(frozen=True)
class ResonantInductorLoss:
    core_w: float
    copper_w: float
    total_w: float
    b_peak_t: float
    ac_factor: float
    b_peak_min_area_t: float = 0.0
    dc_copper_w: float = 0.0
    skin_effect_w: float = 0.0
    external_proximity_w: float = 0.0
    gap_fringing_w: float = 0.0
    bundle_circulating_w: float = 0.0
    termination_w: float = 0.0
    estimated_hotspot_c: float = 0.0
    material_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResonantInductorCandidateSummary:
    rank: int
    part_number: str
    family: str
    material: str
    turns: int
    layers: int
    feasible: bool
    nominal_total_loss_w: float
    nominal_core_loss_w: float
    nominal_copper_loss_w: float
    nominal_gap_fringing_w: float
    nominal_hotspot_c: float
    fill_factor: float
    worst_b_peak_t: float
    gap_total_mm: float
    cost_usd: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ResonantInductorDesign:
    core: CoreSpec
    inductance_h: float
    turns: int
    wire: LitzWire
    layers: int
    turns_per_layer: int
    fill_factor: float
    gap_total_mm: float
    rdc_ohm: float
    worst_b_peak_t: float
    feasible: bool
    reasons: tuple[str, ...]
    alternatives: tuple[ResonantInductorCandidateSummary, ...] = ()

    def _stack(self, spec: LLCDesignSpec, current_waveform) -> tuple[StackLayer, ...]:
        waveform = tuple(float(x) for x in current_waveform)
        turns_by_layer = distribute_turns(self.turns, self.turns_per_layer)
        return tuple(StackLayer(
            label="inductor", turns=n,
            conductor_length_m=n * self.core.mlt_primary_mm * 1e-3,
            wire=self.wire, current_waveform_a=waveform,
        ) for n in turns_by_layer)

    def _gap_field_harmonics(self, spec: LLCDesignSpec, current_waveform) -> dict[int, float]:
        if not spec.include_gap_fringing_loss or self.gap_total_mm <= 0.0:
            return {}
        spectrum = harmonic_rms_spectrum(current_waveform, spec.litz_max_harmonic)
        gap_m = self.gap_total_mm * 1e-3
        distance_m = max(spec.gap_to_winding_distance_mm * 1e-3, 0.25e-3)
        # Fringing coupling decays with distance from the gap and increases with
        # normalized gap length.  It augments, rather than replaces, the 1-D
        # layer MMF field and is exposed through a calibration input.
        coupling = spec.gap_fringing_calibration * min(
            1.5, 0.35 * math.sqrt(gap_m / distance_m))
        return {
            h: coupling * self.turns * spectrum[h] / self.core.window_width_m
            for h in range(1, min(len(spectrum), spec.litz_max_harmonic + 1))
        }

    def _loss_at_temperature(self, spec: LLCDesignSpec, op: LLCOperatingPoint,
                             temperature_c: float):
        time, current, b = resonant_inductor_waveforms(
            op, self.inductance_h, self.turns, self.core.ae_m2,
            spec.magnetic_waveform_samples)
        b_peak = float(max(abs(b.min()), abs(b.max())))
        b_peak_min = b_peak * self.core.ae_m2 / self.core.amin_m2
        p_core = self.core.core_loss_waveform_w(time, b, temperature_c)
        stack = self._stack(spec, current)
        common = dict(
            layers=stack, fundamental_frequency_hz=op.switching_frequency_hz,
            window_width_m=self.core.window_width_m,
            temperature_c=temperature_c, max_harmonic=spec.litz_max_harmonic,
            transposition_quality=spec.litz_transposition_quality,
            sub_bundle_coupling_factor=spec.litz_sub_bundle_coupling_factor,
            termination_resistance_fraction=spec.winding_termination_resistance_fraction,
            calibration_factor=spec.litz_proximity_correction * spec.inductor_proximity_severity,
        )
        base = layered_litz_stack_loss(**common)["inductor"]
        with_gap = layered_litz_stack_loss(
            **common, extra_field_harmonics_a_per_m=self._gap_field_harmonics(spec, current)
        )["inductor"]
        gap_loss = max(0.0, with_gap.external_proximity_w - base.external_proximity_w)
        return (p_core, with_gap, gap_loss, b_peak, b_peak_min,
                self.core.loss_range_warnings(op.switching_frequency_hz, b_peak))

    def loss(self, spec: LLCDesignSpec, op: LLCOperatingPoint) -> ResonantInductorLoss:
        temperature = max(spec.winding_temperature_c, spec.ambient_temperature_c)
        last = None
        for _ in range(spec.magnetic_thermal_max_iterations):
            last = self._loss_at_temperature(spec, op, temperature)
            p_core, copper, *_ = last
            predicted = spec.ambient_temperature_c + (
                p_core + copper.total_w) * spec.resonant_inductor_rth_k_per_w
            predicted = min(max(predicted, spec.ambient_temperature_c), 220.0)
            updated = 0.55 * temperature + 0.45 * predicted
            if abs(updated - temperature) <= spec.magnetic_thermal_tolerance_c:
                temperature = updated
                break
            temperature = updated
        assert last is not None
        p_core, copper, gap_loss, b_peak, b_peak_min, warnings = self._loss_at_temperature(
            spec, op, temperature)
        total = p_core + copper.total_w
        return ResonantInductorLoss(
            core_w=p_core, copper_w=copper.total_w, total_w=total,
            b_peak_t=b_peak, ac_factor=copper.effective_ac_factor,
            b_peak_min_area_t=b_peak_min,
            dc_copper_w=copper.dc_copper_w,
            skin_effect_w=copper.skin_effect_w,
            external_proximity_w=max(0.0, copper.external_proximity_w - gap_loss),
            gap_fringing_w=gap_loss,
            bundle_circulating_w=copper.bundle_circulating_w,
            termination_w=copper.termination_w,
            estimated_hotspot_c=temperature,
            material_warnings=warnings,
        )


def _candidate(spec: LLCDesignSpec, tank: TankDesign,
               ops: tuple[LLCOperatingPoint, ...], core: CoreSpec,
               turns: int) -> ResonantInductorDesign:
    max_i_rms = max(op.resonant_current_rms_a for op in ops)
    max_i_peak = max(op.resonant_current_peak_a for op in ops)
    wire = select_litz_wire(
        max_i_rms, spec.litz_strand_copper_diameter_m,
        spec.litz_strand_outer_diameter_m, spec.litz_packing_factor,
        spec.litz_current_density_target_a_per_mm2)
    layers, tpl = winding_layers(turns, wire, core.window_width_mm)
    fill = turns * wire.envelope_area_m2 * 1e6 / core.aw_mm2
    b_peak = tank.lr_h * max_i_peak / (turns * core.amin_m2)
    gap_m = (MU0 * turns**2 * core.ae_m2 / tank.lr_h
             - core.le_m / core.mu_r)
    length = turns * core.mlt_primary_mm * 1e-3
    rdc = dc_resistance_ohm(wire, length, spec.winding_temperature_c)

    reasons: list[str] = []
    if layers > spec.resonant_inductor_max_layers:
        reasons.append(f"winding requires {layers} layers")
    if fill > spec.resonant_inductor_max_fill_factor:
        reasons.append(f"window fill {fill:.3f} exceeds limit")
    if b_peak > spec.resonant_inductor_max_b_t:
        reasons.append(f"Bpk at minimum area {b_peak:.3f} T exceeds limit")
    if b_peak > 0.70 * core.saturation_flux_at(spec.winding_temperature_c):
        reasons.append("Bpk exceeds 70% of temperature-adjusted saturation flux")
    if gap_m <= 0:
        reasons.append("non-positive calculated gap")
    if max(gap_m, 0.0) * 1e3 > spec.resonant_inductor_max_gap_mm:
        reasons.append(f"total gap {max(gap_m, 0.0)*1e3:.2f} mm exceeds limit")

    return ResonantInductorDesign(
        core=core, inductance_h=tank.lr_h, turns=turns,
        wire=wire, layers=layers, turns_per_layer=tpl,
        fill_factor=fill, gap_total_mm=max(gap_m, 0.0) * 1e3,
        rdc_ohm=rdc, worst_b_peak_t=b_peak,
        feasible=not reasons, reasons=tuple(reasons))


def design_resonant_inductor(spec: LLCDesignSpec, tank: TankDesign,
                              operating_points: Iterable[LLCOperatingPoint],
                              database: CoreDatabase | None = None,
                              preferred_core: str | None = None) -> ResonantInductorDesign:
    db = database or CoreDatabase()
    cores = [db.get(preferred_core)] if preferred_core else db.for_purpose("inductor", spec.resonant_inductor_core_families)
    ops = tuple(operating_points)
    candidates = [_candidate(spec, tank, ops, core, turns)
                  for core in cores for turns in range(6, 61)]
    feasible = [c for c in candidates if c.feasible]
    pool = feasible or candidates
    nominal = min(ops, key=lambda op: abs(op.vbus_v - spec.vbus_nom_v)
                  + 100.0 * abs(op.load_fraction - 1.0))

    # Full harmonic/MMF/thermal evaluation is intentionally limited to the
    # most promising candidates.  The pre-screen uses physically monotonic
    # terms (classical core loss plus hot-Rdc copper loss) and avoids running
    # dozens of FFT/thermal iterations for obviously inferior turn counts.
    def quick_score(candidate: ResonantInductorDesign) -> tuple[float, float, float]:
        b_nom = (candidate.inductance_h * nominal.resonant_current_peak_a /
                 (candidate.turns * candidate.core.ae_m2))
        p_core = candidate.core.core_loss_w(
            nominal.switching_frequency_hz, b_nom, spec.winding_temperature_c)
        p_cu = nominal.resonant_current_rms_a**2 * candidate.rdc_ohm * 1.25
        return (p_core + p_cu, candidate.core.cost_usd, candidate.turns)

    preselected = sorted(pool, key=quick_score)[:max(1, spec.magnetic_candidate_full_evaluation_limit)]
    evaluated = [(candidate, candidate.loss(spec, nominal)) for candidate in preselected]
    evaluated.sort(key=lambda item: (not item[0].feasible, item[1].total_w,
                                     item[0].core.cost_usd, item[0].turns))
    summaries = tuple(ResonantInductorCandidateSummary(
        rank=index + 1, part_number=design.core.part_number,
        family=design.core.family, material=design.core.material,
        turns=design.turns, layers=design.layers, feasible=design.feasible,
        nominal_total_loss_w=loss.total_w, nominal_core_loss_w=loss.core_w,
        nominal_copper_loss_w=loss.copper_w,
        nominal_gap_fringing_w=loss.gap_fringing_w,
        nominal_hotspot_c=loss.estimated_hotspot_c,
        fill_factor=design.fill_factor, worst_b_peak_t=design.worst_b_peak_t,
        gap_total_mm=design.gap_total_mm, cost_usd=design.core.cost_usd,
        reasons=design.reasons,
    ) for index, (design, loss) in enumerate(evaluated[:20]))
    return replace(evaluated[0][0], alternatives=summaries)
