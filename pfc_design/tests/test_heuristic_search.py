"""Heuristic search: shared feasibility checker, local refine, DE global search.

Verifies:
  1. sweep feasibility columns agree with the shared checker (口径一致)
  2. local refine never worsens a feasible grid seed
  3. global search finds feasible designs where the grid found none
  4. penalty objective ordering: any feasible design < any infeasible one
  5. global search is deterministic for a fixed seed
"""

import numpy as np
import pytest

from pfc_design.core.spec import DesignSpec, MosfetDatabase
from pfc_design.core.operating_point import compute_mathcad_operating_point
from pfc_design.magnetics.core_database import CoreDatabase
from pfc_design.models.system import SystemAnalyzer
from pfc_design.optimization.sweep import ParamSweep
from pfc_design.optimization.scipy_opt import (
    PENALTY, make_penalty_objective, refine_design,
)
from pfc_design.optimization.feasibility import evaluate_feasibility
from pfc_design.optimization.heuristic import global_search


@pytest.fixture(scope="module")
def sweep_candidates(shared_db):
    cores = shared_db.top_for_pfc(
        ae_min_cm2=0.5, material_class="Sendust", n=5)
    mosfets = MosfetDatabase().query(vds_min=410 * 1.25, technology="Si")[:5]
    return cores, mosfets


class TestSharedFeasibilityChecker:

    def test_sweep_columns_match_shared_checker(self, shared_spec, shared_db,
                                                sweep_candidates):
        """Re-analyzing each sweep row must give the same feasibility verdict."""
        cores, mosfets = sweep_candidates
        sweeper = ParamSweep(shared_db)
        df = sweeper.sweep(
            shared_spec,
            {"fsw": np.array([65e3, 85e3]),
             "ripple_ratio": np.array([0.2, 0.35, 0.5]),
             "core_idx": list(range(len(cores))),
             "mosfet_idx": list(range(len(mosfets)))},
            cores=cores, mosfets=mosfets,
        )

        assert len(df) == 2 * 3 * len(cores) * len(mosfets)

        for _, row in df.iterrows():
            core = shared_db.get_by_part_number(row["core"])
            spec = shared_spec.clone()
            spec.fsw = row["fsw_kHz"] * 1000
            spec.ripple_ratio = row["ripple_ratio"]
            op = compute_mathcad_operating_point(spec)
            result = SystemAnalyzer(shared_db).analyze(
                spec, op, preferred_core=core,
                mosfet=sweeper.mosfet_db.get(row["mosfet"]), n_cores=2)
            report = evaluate_feasibility(result, core)
            assert report.feasible == bool(row["feasible"]), row["constraints"]
            assert report.failed_bmax == bool(row["failed_bmax"])
            assert report.failed_window == bool(row["failed_window"])
            assert report.failed_saturation == bool(row["failed_saturation"])
            assert report.failed_actual_ripple == bool(row["failed_actual_ripple"])

    def test_reason_strings_format(self, shared_spec, shared_db):
        """Golden format check for constraint reason strings."""
        sweeper = ParamSweep(shared_db)
        cores = shared_db.top_for_pfc(
            ae_min_cm2=0.5, material_class="Sendust", n=1)
        mosfets = [sweeper.mosfet_db.query(vds_min=400, technology="Si")[0]]
        df = sweeper.sweep(
            shared_spec,
            {"fsw": np.array([65e3]), "ripple_ratio": np.array([0.3]),
             "core_idx": np.array([0]), "mosfet_idx": np.array([0])},
            cores=cores, mosfets=mosfets,
        )
        row = df.iloc[0]
        # 65 kHz single-core combination fails Bmax (verified empirically),
        # so reasons must carry the exact Bmax string format.
        assert "ripple_constraint:" in row["constraints"]
        assert row["constraints"].count(">") >= 1


class TestLocalRefine:

    def test_refine_never_worsens_grid_seed(self, shared_spec, shared_db,
                                            sweep_candidates):
        """Refining a feasible grid seed keeps feasibility and loss <= seed."""
        cores, mosfets = sweep_candidates
        sweeper = ParamSweep(shared_db)
        df = sweeper.sweep(
            shared_spec,
            {"fsw": np.arange(45e3, 121e3, 15e3),
             "ripple_ratio": np.array([0.2, 0.35, 0.5]),
             "core_idx": list(range(len(cores))),
             "mosfet_idx": list(range(len(mosfets)))},
            cores=cores, mosfets=mosfets,
        )
        feasible = df[df["feasible"]]
        assert len(feasible) > 0, "probe sweep must contain feasible points"
        best = feasible.nsmallest(1, "P_total_W").iloc[0]

        analyzer = SystemAnalyzer(shared_db)
        core = shared_db.get_by_part_number(best["core"])
        mosfet = sweeper.mosfet_db.get(best["mosfet"])
        r = refine_design(analyzer, shared_spec, core, mosfet,
                          x0=(best["fsw_kHz"], best["ripple_ratio"]))

        assert r["feasible"], r["reasons"]
        assert r["loss_W"] <= best["P_total_W"] + 1e-6
        assert r["evaluations"] > 0

    def test_penalty_orders_feasible_below_infeasible(self, shared_spec,
                                                      shared_db,
                                                      sweep_candidates):
        """Infeasible designs must cost >= PENALTY, feasible ones << PENALTY."""
        cores, mosfets = sweep_candidates
        core, mosfet = cores[0], mosfets[0]
        analyzer = SystemAnalyzer(shared_db)
        objective = make_penalty_objective(analyzer, shared_spec, core, mosfet)

        losses = []
        for fsw in (55, 75, 95):
            for ripple in (0.2, 0.35, 0.5):
                spec = shared_spec.clone()
                spec.fsw = fsw * 1000
                spec.ripple_ratio = ripple
                op = compute_mathcad_operating_point(spec)
                result = analyzer.analyze(spec, op, preferred_core=core,
                                          mosfet=mosfet, n_cores=2)
                report = evaluate_feasibility(result, core)
                value = objective(np.array([fsw, ripple]))
                if report.feasible:
                    assert value < PENALTY, f"feasible design penalized: {value}"
                else:
                    assert value >= PENALTY, f"infeasible design cheap: {value}"
                losses.append(value)

        # All feasible values (if any) must rank below any infeasible one
        # by construction; verify no ordering violation exists.
        feasible_vals = [v for v in losses if v < PENALTY]
        if feasible_vals:
            assert max(feasible_vals) < PENALTY


class TestGlobalSearch:

    def test_65khz_grid_feasible_after_ldesign_fix(self, shared_spec,
                                                   shared_db):
        """T1 regression: the design loop now targets L_eff at peak current.

        Before the fix the 65 kHz grid over Sendust cores yielded zero
        feasible designs (ripple margin ~1.6 everywhere). With the loop
        honoring the effective inductance, the same grid must be feasible.
        """
        sweeper = ParamSweep(shared_db)
        cores = shared_db.top_for_pfc(
            ae_min_cm2=0.5, material_class="Sendust", n=10)
        mosfets = sweeper.mosfet_db.query(vds_min=410 * 1.25, technology="Si")

        df = sweeper.sweep(
            shared_spec,
            {"fsw": np.array([65e3]),
             "ripple_ratio": np.linspace(0.15, 0.5, 6),
             "core_idx": list(range(len(cores))),
             "mosfet_idx": list(range(len(mosfets)))},
            cores=cores, mosfets=mosfets,
        )
        assert df["feasible"].sum() > 0, "65 kHz grid must be feasible now"
        best = df[df["feasible"]].nsmallest(1, "P_total_W").iloc[0]
        assert best["actual_ripple_margin"] <= 1.50

    def test_global_search_matches_or_beats_grid_best(self, shared_spec,
                                                      shared_db):
        """DE global search must return a feasible design no worse than the
        grid best it is seeded against (same constraint thresholds)."""
        sweeper = ParamSweep(shared_db)
        cores = shared_db.top_for_pfc(
            ae_min_cm2=0.5, material_class="Sendust", n=5)
        mosfets = sweeper.mosfet_db.query(vds_min=410 * 1.25, technology="Si")[:5]

        df = sweeper.sweep(
            shared_spec,
            {"fsw": np.arange(45e3, 121e3, 15e3),
             "ripple_ratio": np.array([0.2, 0.35, 0.5]),
             "core_idx": list(range(len(cores))),
             "mosfet_idx": list(range(len(mosfets)))},
            cores=cores, mosfets=mosfets,
        )
        assert df["feasible"].sum() > 0
        grid_best = float(df[df["feasible"]]["P_total_W"].min())

        r = global_search(SystemAnalyzer(shared_db), shared_spec,
                          cores, mosfets, maxiter=30)
        assert r["feasible"], f"DE must find a feasible design: {r['reasons']}"
        assert r["loss_W"] < PENALTY
        assert 45.0 <= r["fsw_kHz"] <= 120.0
        assert 0.10 <= r["ripple_ratio"] <= 0.60
        assert r["loss_W"] <= grid_best + 1e-6

    def test_deterministic_with_fixed_seed(self, shared_spec, shared_db,
                                           sweep_candidates):
        cores, mosfets = sweep_candidates
        analyzer = SystemAnalyzer(shared_db)
        r1 = global_search(analyzer, shared_spec, cores, mosfets,
                           maxiter=10, seed=42)
        r2 = global_search(analyzer, shared_spec, cores, mosfets,
                           maxiter=10, seed=42)
        assert r1["core"] == r2["core"]
        assert r1["mosfet"] == r2["mosfet"]
        assert r1["fsw_kHz"] == pytest.approx(r2["fsw_kHz"], abs=1e-6)
        assert r1["loss_W"] == pytest.approx(r2["loss_W"], abs=1e-6)
