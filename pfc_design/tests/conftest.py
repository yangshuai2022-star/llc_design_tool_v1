"""Shared fixtures for pfc_design test suite.

All fixture names use 'shared_' prefix to avoid collisions with
test_verify_mathcad.py's own fixtures (mathcad_spec, mathcad_op, db).
"""

import pytest

from pfc_design.core.spec import DesignSpec
from pfc_design.core.operating_point import compute_mathcad_operating_point
from pfc_design.magnetics.core_database import CoreDatabase
from pfc_design.core.line_cycle import build_line_cycle_trace


@pytest.fixture(scope="module")
def shared_spec():
    return DesignSpec(
        vin_min=176.0, vin_max=264.0, vin_nom=220.0,
        vout=410.0, pout_total=7100.0, n_phases=2,
        f_line=50.0, fsw=65_000.0, ripple_ratio=1.0,
        eta_target=0.96, core_material_pref="Sendust",
        diode_vf=1.5, diode_rd=0.1, diode_type="SiC",
        diode_part=None,           # shared fixtures predate the diode database
        bridge_vf=1.0, bridge_rd=0.0,
        c_out_total=1320e-6, cap_esr=0.15, cap_n_parallel=4,
    )


@pytest.fixture(scope="module")
def shared_op(shared_spec):
    return compute_mathcad_operating_point(shared_spec)


@pytest.fixture(scope="module")
def shared_db():
    return CoreDatabase()


@pytest.fixture(scope="module")
def shared_trace(shared_op, shared_spec):
    return build_line_cycle_trace(shared_op, shared_spec, L_uh=182.0)
