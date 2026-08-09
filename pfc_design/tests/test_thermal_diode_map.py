"""T2/T4/T5: diode database, thermal loop, efficiency map building blocks.

Verifies:
  1. thermal off == legacy behavior (80 C fixed junction)
  2. thermal on: Tj > Tamb, self-consistent with MOSFET Rds(Tj)
  3. diode part model: Vf(T), SiC zero Qrr, Si Qrr > 0
  4. fixed-design reuse (map use case) keeps hardware identical
"""

import numpy as np
import pytest

from pfc_design.core.spec import DesignSpec, DiodeDatabase
from pfc_design.core.operating_point import OperatingPoint
from pfc_design.models.system import SystemAnalyzer
from pfc_design.models.diode import DiodeLoss


@pytest.fixture(scope="module")
def diode_db():
    return DiodeDatabase()


class TestThermalLoop:

    def test_thermal_off_is_legacy_80c(self, shared_spec):
        r = SystemAnalyzer().analyze(shared_spec)
        assert not r["thermal"]["enabled"]
        assert r["thermal"]["t_j_mosfet"] == 80.0
        assert r["thermal"]["t_j_diode"] == 80.0

    def test_thermal_on_raises_losses_and_reports_tj(self, shared_spec):
        spec = shared_spec.clone()
        spec.thermal_model = True
        r = SystemAnalyzer().analyze(spec)
        th = r["thermal"]
        assert th["enabled"]
        # Junction temperatures must sit above ambient
        assert th["t_j_mosfet"] > spec.t_ambient
        assert th["t_j_diode"] > spec.t_ambient
        # The MOSFET loss must be evaluated at the converged Tj
        assert r["losses"]["mosfet"].metadata["T_j"] == pytest.approx(
            th["t_j_mosfet"], abs=0.5)
        # Self-consistent: Tj = Ta + Rth * P_mos
        rth = th["rth_mosfet_ja"]
        expected = spec.t_ambient + rth * r["losses"]["mosfet"].power_loss_W
        assert th["t_j_mosfet"] == pytest.approx(expected, abs=1.0)

    def test_thermal_on_loss_geq_off(self, shared_spec):
        r_off = SystemAnalyzer().analyze(shared_spec)
        spec = shared_spec.clone()
        spec.thermal_model = True
        r_on = SystemAnalyzer().analyze(spec)
        # Hotter junction -> higher Rds -> conduction loss >=
        assert r_on["total_loss"] >= r_off["total_loss"] - 1e-9


class TestDiodeDatabase:

    def test_parts_load_and_query(self, diode_db):
        parts = diode_db.all
        assert len(parts) >= 6
        sic = diode_db.query(technology="SiC")
        assert all("sic" in d.technology.lower() for d in sic)
        c4d = diode_db.get("C4D10120H")
        assert c4d is not None and c4d.is_sic

    def test_vf_temperature_model(self, diode_db):
        c4d = diode_db.get("C4D10120H")
        # SiC Schottky Vf increases with temperature
        assert c4d.vf_at(150.0) > c4d.vf_at(25.0)
        stth = diode_db.get("STTH12R06")
        # Si fast-recovery Vf decreases with temperature
        assert stth.vf_at(150.0) < stth.vf_at(25.0)

    def test_sic_no_reverse_recovery(self, shared_spec, shared_op, diode_db):
        dio = diode_db.get("C4D10120H")
        r = DiodeLoss().compute(shared_spec, shared_op, diode=dio, t_j=100.0)
        assert r.sub_losses["reverse_recovery"] == 0.0
        assert r.metadata["part_number"] == "C4D10120H"

    def test_si_reverse_recovery_scales_with_fsw(self, diode_db):
        stth = diode_db.get("STTH12R06")
        assert stth.qrr > 0
        spec_lo = DesignSpec(fsw=50_000.0)
        spec_hi = DesignSpec(fsw=100_000.0)
        op = OperatingPoint(DesignSpec(), vin_rms=176.0)
        r_lo = DiodeLoss().compute(spec_lo, op, diode=stth, t_j=80.0)
        r_hi = DiodeLoss().compute(spec_hi, op, diode=stth, t_j=80.0)
        assert r_hi.sub_losses["reverse_recovery"] == pytest.approx(
            2 * r_lo.sub_losses["reverse_recovery"])

    def test_part_diode_loss_below_legacy(self, shared_spec, shared_db):
        """C4D10120H (rd=30mOhm) beats the legacy vf=1.5/rd=0.1 model."""
        analyzer = SystemAnalyzer(shared_db)
        legacy_spec = shared_spec.clone()
        legacy_spec.diode_part = None   # force the legacy Vf/Rd path
        r_legacy = analyzer.analyze(legacy_spec)
        spec = shared_spec.clone()
        spec.diode_part = "C4D10120H"
        r_part = analyzer.analyze(spec)
        assert r_part["diode"] is not None
        assert (r_part["losses"]["diode"].power_loss_W
                < r_legacy["losses"]["diode"].power_loss_W)


class TestFixedDesignReuse:

    def test_reused_design_keeps_hardware(self, shared_spec, shared_db):
        """Efficiency map re-analyzes a fixed design at other loads."""
        analyzer = SystemAnalyzer(shared_db)
        base = analyzer.analyze(shared_spec)
        design = base["inductor_design"]

        spec_50 = shared_spec.clone()
        spec_50.pout_total = shared_spec.pout_total * 0.5
        op_50 = OperatingPoint(spec_50, vin_rms=220.0)
        r50 = analyzer.analyze(spec_50, op_50, preferred_core=design.core,
                               n_cores=design.n_cores, design=design)

        # Hardware identical: same core, same turns, same effective L
        assert r50["inductor_design"] is design
        assert r50["inductor_design"].n_turns == design.n_turns
        # Half load: losses drop, efficiency vs full-load at 220V
        assert r50["total_loss"] < base["total_loss"]
        assert r50["efficiency"] > 0
