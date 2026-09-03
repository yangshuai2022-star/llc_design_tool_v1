"""Regression tests for the Dowell formulas against the MATLAB book programs.

Targets come from
``/mnt/d/MATLAB_CODE/BOOK_TRANSFORMER_INDUCTOR/Problem_6_4.m``,
``Problem_6_6.m`` and ``Problem_6_7.m`` — reproducing them proves the formula
transcription and the unit handling are correct.
"""

import math

from llc_design.magnetics.dowell import (
    dowell_kpn,
    dowell_optimum_delta,
    dowell_optimum_thickness_mm,
)


def test_dowell_optimum_delta_rectangular_matches_problem_6_6():
    # Problem 6.6: D=0.5, p=6, no rise time.
    #   deltaopt = (pi^2 * 0.5 * 0.5 / (3 * (5*36-1)/15))^0.25
    expected = (math.pi ** 2 * 0.5 * 0.5 / (3.0 * (5 * 36 - 1) / 15.0)) ** 0.25
    got = dowell_optimum_delta(duty=0.5, layers_p=6)
    assert math.isclose(got, expected, rel_tol=1e-9)
    # Numerical value the book prints.
    assert math.isclose(got, 0.5124, rel_tol=1e-3)


def test_dowell_optimum_delta_with_rise_time_matches_problem_6_4():
    # Problem 6.4: D=0.5, trT=0.04, p=6.
    #   deltaopt = ((0.5 - 4/3*0.04) * 2*pi^2*0.04 / ((5*36-1)/15))^0.25
    expected = (
        (0.5 - 4.0 / 3.0 * 0.04) * 2.0 * math.pi ** 2 * 0.04
        / ((5 * 36 - 1) / 15.0)
    ) ** 0.25
    got = dowell_optimum_delta(duty=0.5, layers_p=6, rise_time_frac=0.04)
    assert math.isclose(got, expected, rel_tol=1e-9)
    assert math.isclose(got, 0.4146, rel_tol=1e-3)


def test_dowell_optimum_delta_decreases_with_more_layers():
    # More layers -> the proximity term (p^2-1) grows -> optimum delta shrinks.
    d1 = dowell_optimum_delta(duty=0.5, layers_p=1)
    d3 = dowell_optimum_delta(duty=0.5, layers_p=3)
    d6 = dowell_optimum_delta(duty=0.5, layers_p=6)
    assert d1 > d3 > d6


def test_dowell_kpn_matches_problem_6_6_single_point():
    # Problem 6.6 inner loop, delta=0.5, n=1, p=6.
    dn = 0.5 * math.sqrt(1.0)
    term1 = (math.sinh(2 * dn) + math.sin(2 * dn)) / (
        math.cosh(2 * dn) - math.cos(2 * dn)
    )
    term2 = (2 * (6 ** 2 - 1) / 3) * (math.sinh(dn) - math.sin(dn)) / (
        math.cosh(dn) + math.cos(dn)
    )
    expected = dn * (term1 + term2)
    got = dowell_kpn(delta=0.5, harmonic_n=1, layers_p=6)
    assert math.isclose(got, expected, rel_tol=1e-9)
    assert math.isclose(got, 1.2480, rel_tol=1e-3)


def test_dowell_kpn_dc_limit_is_one():
    # As delta -> 0 the AC resistance factor tends to the DC value: k_pn -> 1
    # (skin term -> 1/delta_n, multiplied by delta_n outside the bracket;
    # the proximity term vanishes as delta_n^3).
    # 1e-4 is well clear of float cancellation in sinh(dn)-sin(dn).
    assert math.isclose(dowell_kpn(1e-4, 1, 6), 1.0, abs_tol=1e-3)
    assert math.isclose(dowell_kpn(1e-3, 1, 6), 1.0, abs_tol=1e-3)


def test_dowell_optimum_thickness_scales_with_skin_depth():
    # At 100 kHz copper ~100C the skin depth is ~0.24 mm; for p=6, D=0.5 the
    # optimum foil thickness is deltaopt (~0.512) times that.
    t_100k = dowell_optimum_thickness_mm(0.5, 6, 100e3, temperature_c=100.0)
    t_400k = dowell_optimum_thickness_mm(0.5, 6, 400e3, temperature_c=100.0)
    # Skin depth scales as 1/sqrt(f), so optimum thickness halves at 4x freq.
    assert math.isclose(t_100k / t_400k, 2.0, rel_tol=1e-6)
    assert 0.1 < t_100k < 0.4