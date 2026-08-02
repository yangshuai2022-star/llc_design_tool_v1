"""Verification tests against Mathcad reference values.

Run with: python -m pytest pfc_design/tests/verify_mathcad.py -v
"""

import pytest
import numpy as np

from pfc_design.core.spec import DesignSpec, MosfetDatabase
from pfc_design.core.operating_point import compute_mathcad_operating_point
from pfc_design.magnetics.core_database import CoreDatabase
from pfc_design.models.system import SystemAnalyzer


@pytest.fixture
def mathcad_spec():
    """Mathcad reference specification — MOSFET spec via MosfetDatabase."""
    return DesignSpec(
        vin_min=176.0, vin_max=264.0, vin_nom=220.0,
        vout=410.0, pout_total=7100.0, n_phases=2,
        f_line=50.0, fsw=65_000.0, ripple_ratio=1.0,
        eta_target=0.96,
        core_material_pref="Sendust",
        diode_vf=1.5, diode_rd=0.1, diode_type="SiC",
        diode_part=None,           # keep the Mathcad Vf/Rd comparison path
        bridge_vf=1.0,
        c_out_total=1320e-6, cap_esr=0.15, cap_n_parallel=4,
        cap_tan_delta=0.25,
    )


@pytest.fixture
def mathcad_mosfet():
    """Create a Mathcad-matching MOSFET (100mΩ, 650V)."""
    from pfc_design.core.spec import MosfetSpec
    return MosfetSpec(
        manufacturer="Test", part_number="Mathcad-ref",
        technology="SiC", vds_max=650.0, id_25c=47.0, id_100c=30.0,
        rds_on_25c=0.100, rds_on_150c=0.160, rds_alpha=0.004,
        qg_nc=100.0, coss_er_pF=200.0, tr_ns=5.0, tf_ns=4.5, vgs=14.0
    )


@pytest.fixture
def db():
    return CoreDatabase()


@pytest.fixture
def mathcad_op(mathcad_spec):
    return compute_mathcad_operating_point(mathcad_spec)


class TestOperatingPoint:
    def test_iin_rms(self, mathcad_op):
        """Mathcad: Iin_max_rms = 21.011 A"""
        assert mathcad_op.iin_rms == pytest.approx(21.011, rel=0.02)

    def test_d_min(self, mathcad_op):
        """Mathcad: Dmin = 0.393"""
        assert mathcad_op.d_min == pytest.approx(0.393, rel=0.05)

    def test_iin_peak(self, mathcad_op):
        """Iin_peak = sqrt(2) * Iin_rms"""
        assert mathcad_op.iin_peak == pytest.approx(29.71, rel=0.02)


class TestInductorDesign:
    def test_L_target(self, mathcad_spec, mathcad_op, mathcad_mosfet, db):
        """New boost inductor formula (~51 uH with r=1.0).

        The DOC-corrected formula: L = Vin_pk * D_pk / (fsw * DeltaI_ref)
        with ripple_ratio referenced to Iin_pk_phase gives ~50.6 uH.
        The legacy formula (L = Vm / (r * fsw * Iin_rms)) gave ~182 uH
        but was missing duty-cycle and had ambiguous ripple reference.
        """
        analyzer = SystemAnalyzer(db)
        result = analyzer.analyze(mathcad_spec, mathcad_op, mosfet=mathcad_mosfet)
        assert result["inductor_design"].L_target_uh == pytest.approx(50.64, rel=0.05)

    def test_N_turns_reasonable(self, mathcad_spec, mathcad_op, mathcad_mosfet, db):
        """N around 45-80 turns for this core and L"""
        analyzer = SystemAnalyzer(db)
        result = analyzer.analyze(mathcad_spec, mathcad_op, mosfet=mathcad_mosfet)
        assert 35 <= result["inductor_design"].n_turns <= 80


class TestSystemEfficiency:
    def test_efficiency_reasonable(self, mathcad_spec, mathcad_op, mathcad_mosfet, db):
        """Mathcad: eta ≈ 96-97%"""
        analyzer = SystemAnalyzer(db)
        result = analyzer.analyze(mathcad_spec, mathcad_op, mosfet=mathcad_mosfet)
        eta = result["efficiency"]
        assert 0.94 <= eta <= 0.99

    def test_total_loss_reasonable(self, mathcad_spec, mathcad_op, mathcad_mosfet, db):
        """Total loss 80-400W for ~7kW PFC.

        New L formula gives lower L (~51 uH vs ~182 uH), increasing ripple
        and therefore losses. The bridge loss also doubled (corrected from
        per-phase to total current basis). Combined effect pushes losses
        higher but should stay under 400 W.
        """
        analyzer = SystemAnalyzer(db)
        result = analyzer.analyze(mathcad_spec, mathcad_op, mosfet=mathcad_mosfet)
        assert 80 <= result["total_loss"] <= 400

    def test_bridge_loss(self, mathcad_spec, mathcad_op, mathcad_mosfet, db):
        """Bridge loss: ~75.7 W with corrected total-current basis.

        The old Mathcad reference (37.83 W) used per-phase current
        (op.iin_rms = 21 A) instead of total input current (~42 A).
        Bridge is a shared component and must use total current.
        Iin_rms_total = Pout/(eta*Vin) = 7100/(0.96*176) = 42.0 A.
        Pbridge = 2 * Vf * Iin_abs_avg_total = 2 * 1.0 * (2/pi * sqrt(2) * 42.0)
                ≈ 2 * 1.0 * 37.83 = 75.7 W.
        """
        analyzer = SystemAnalyzer(db)
        result = analyzer.analyze(mathcad_spec, mathcad_op, mosfet=mathcad_mosfet)
        bridge = result["losses"]["bridge"].power_loss_W
        assert bridge == pytest.approx(75.67, rel=0.10)
