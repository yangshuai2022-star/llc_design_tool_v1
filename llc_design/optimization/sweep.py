"""Mixed discrete/continuous grid search for the LLC V1 model."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Iterable

import pandas as pd

from ..core.spec import LLCDesignSpec
from ..models.system import LLCSystemAnalyzer, SystemAnalysis
from .pareto import pareto_front


@dataclass(frozen=True)
class OptimizationConfig:
    ln_values: tuple[float, ...]
    q_values: tuple[float, ...]
    fr_values_hz: tuple[float, ...]
    primary_turn_values: tuple[int, ...]
    secondary_turn_values: tuple[int, ...]
    primary_devices: tuple[str, ...]
    sr_parallel_values: tuple[int, ...]

    @classmethod
    def quick(cls) -> "OptimizationConfig":
        return cls(
            ln_values=(4.0, 5.0, 6.0),
            q_values=(0.30, 0.40),
            fr_values_hz=(90_000.0, 110_000.0),
            primary_turn_values=(28, 30, 32),
            secondary_turn_values=(4,),
            primary_devices=("REF_650V_SIC_45M",),
            sr_parallel_values=(2,),
        )

    @classmethod
    def full(cls) -> "OptimizationConfig":
        return cls(
            ln_values=(3.5, 4.5, 5.5, 6.5),
            q_values=(0.25, 0.35, 0.45, 0.55),
            fr_values_hz=(80_000.0, 100_000.0, 120_000.0, 140_000.0),
            primary_turn_values=(26, 30, 34, 38),
            secondary_turn_values=(4,),
            primary_devices=("REF_650V_SIC_45M", "REF_650V_SJ_32M", "REF_650V_GAN_25M"),
            sr_parallel_values=(1, 2, 3),
        )

    @property
    def count(self) -> int:
        return math.prod(len(x) for x in (
            self.ln_values, self.q_values, self.fr_values_hz,
            self.primary_turn_values, self.secondary_turn_values,
            self.primary_devices, self.sr_parallel_values))


@dataclass
class OptimizationResult:
    table: pd.DataFrame
    pareto: pd.DataFrame
    best_analysis: SystemAnalysis | None


class LLCOptimizer:
    def __init__(self, analyzer: LLCSystemAnalyzer | None = None):
        self.analyzer = analyzer or LLCSystemAnalyzer()

    def run(self, base_spec: LLCDesignSpec,
            config: OptimizationConfig | None = None,
            maximum_candidates: int | None = None) -> OptimizationResult:
        cfg = config or OptimizationConfig.quick()
        combinations = itertools.product(
            cfg.ln_values, cfg.q_values, cfg.fr_values_hz,
            cfg.primary_turn_values, cfg.secondary_turn_values,
            cfg.primary_devices, cfg.sr_parallel_values)
        rows: list[dict] = []
        analyses: list[SystemAnalysis | None] = []
        for index, (ln, q, fr, np_turns, ns_turns, primary_device, sr_parallel) in enumerate(combinations):
            if maximum_candidates is not None and index >= maximum_candidates:
                break
            spec = base_spec.clone(
                ln_ratio=float(ln), q_full_load=float(q),
                resonant_frequency_hz=float(fr),
                primary_turns=int(np_turns), secondary_turns=int(ns_turns),
                primary_device=primary_device,
                sr_parallel_devices_per_position=int(sr_parallel))
            try:
                analysis = self.analyzer.analyze(spec)
                row = self._row(analysis)
            except Exception as exc:  # preserve failed candidates in the engineering record
                analysis = None
                row = {
                    "feasible": False, "failure": str(exc),
                    "ln": ln, "q_full": q, "fr_khz": fr / 1e3,
                    "primary_turns": np_turns, "secondary_turns": ns_turns,
                    "turns_ratio": np_turns / ns_turns,
                    "primary_device": primary_device,
                    "sr_parallel": sr_parallel,
                    "score_w": 1e9,
                }
            rows.append(row)
            analyses.append(analysis)

        table = pd.DataFrame(rows)
        if table.empty:
            return OptimizationResult(table, table, None)
        table = table.sort_values(["feasible", "score_w"], ascending=[False, True]).reset_index(drop=True)
        feasible = table[table["feasible"]].copy()
        pareto = pareto_front(feasible, ("weighted_loss_w", "magnetics_volume_cm3")) if not feasible.empty else feasible

        best_analysis = None
        if not feasible.empty:
            best_key = feasible.iloc[0]["candidate_key"]
            for row, analysis in zip(rows, analyses):
                if analysis is not None and row.get("candidate_key") == best_key:
                    best_analysis = analysis
                    break
        return OptimizationResult(table=table, pareto=pareto,
                                  best_analysis=best_analysis)

    @staticmethod
    def _point_weight(spec: LLCDesignSpec, vbus_v: float, load_fraction: float) -> float:
        """Weight of an operating point in the loss objective.

        Derived from the spec's bus levels instead of hard-coded labels so
        custom hold/normal/min/max voltages keep the intended weighting.
        """
        if load_fraction >= 0.999:
            for level, weight in ((spec.vbus_max_v, 0.10), (spec.vbus_nom_v, 0.30),
                                  (spec.vbus_min_normal_v, 0.15),
                                  (spec.vbus_hold_end_v, 0.15)):
                if abs(vbus_v - level) <= 1e-6:
                    return weight
        elif abs(vbus_v - spec.vbus_nom_v) <= 1e-6:
            for load, weight in ((0.50, 0.20), (0.25, 0.10), (0.10, 0.00)):
                if abs(load_fraction - load) <= 1e-6:
                    return weight
        return 0.0

    @staticmethod
    def _row(analysis: SystemAnalysis) -> dict:
        spec = analysis.spec
        weighted = sum(LLCOptimizer._point_weight(
            spec, p.operating_point.vbus_v, p.operating_point.load_fraction)
            * p.total_loss_w for p in analysis.operating_points)
        nominal = analysis.nominal
        minimum_zvs = min(min(p.primary.zvs_charge_margin,
                              p.primary.zvs_energy_margin)
                          for p in analysis.operating_points
                          if p.operating_point.load_fraction >= 0.25)
        volume_cm3 = ((analysis.transformer.core.ve_mm3
                       + analysis.resonant_inductor.core.ve_mm3) / 1000.0)
        penalty = 0.0 if analysis.feasible else 1e6 + 1e4 * len(analysis.feasibility_reasons)
        full_load = [p for p in analysis.operating_points
                     if p.operating_point.load_fraction >= 0.999]
        hold_point = min(full_load, key=lambda p: p.operating_point.vbus_v) if full_load else None
        key = (f"Ln{spec.ln_ratio:.3f}_Q{spec.q_full_load:.3f}_"
               f"fr{spec.resonant_frequency_hz:.0f}_Np{spec.primary_turns}_"
               f"Ns{spec.secondary_turns}_{spec.primary_device}_SR{spec.sr_parallel_devices_per_position}")
        return {
            "candidate_key": key,
            "feasible": analysis.feasible,
            "failure": "; ".join(analysis.feasibility_reasons),
            "score_w": weighted + penalty,
            "weighted_loss_w": weighted,
            "nominal_loss_w": nominal.total_loss_w,
            "nominal_efficiency_pct": nominal.efficiency * 100.0,
            "minimum_efficiency_pct": analysis.minimum_efficiency.efficiency * 100.0,
            "worst_loss_w": analysis.worst_loss.total_loss_w,
            "minimum_zvs_margin": minimum_zvs,
            "ln": spec.ln_ratio,
            "q_full": spec.q_full_load,
            "fr_khz": spec.resonant_frequency_hz / 1e3,
            "f_hold_khz": (hold_point.operating_point.switching_frequency_hz / 1e3
                           if hold_point else None),
            "primary_turns": spec.primary_turns,
            "secondary_turns": spec.secondary_turns,
            "turns_ratio": spec.turns_ratio,
            "lr_uh": analysis.tank.lr_h * 1e6,
            "cr_nf": analysis.tank.cr_f * 1e9,
            "lm_uh": analysis.tank.lm_h * 1e6,
            "primary_device": spec.primary_device,
            "sr_parallel": spec.sr_parallel_devices_per_position,
            "transformer_core": analysis.transformer.core.part_number,
            "transformer_fill": analysis.transformer.fill_factor,
            "transformer_b_t": analysis.transformer.worst_b_peak_t,
            "inductor_core": analysis.resonant_inductor.core.part_number,
            "inductor_turns": analysis.resonant_inductor.turns,
            "inductor_layers": analysis.resonant_inductor.layers,
            "inductor_b_t": analysis.resonant_inductor.worst_b_peak_t,
            "magnetics_volume_cm3": volume_cm3,
        }
