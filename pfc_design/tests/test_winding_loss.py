"""Regression tests for winding-loss and iGSE math fixes.

Cross-checks against exact/closed-form references (not snapshots):
  1. skin_effect_factor — exact Kelvin-function solution (was a piecewise
     approximation that returned Rac < Rdc for 2 <= d/delta < 3).
  2. proximity_factor — Dowell's F_R = phi*[G1 + (2/3)(m^2-1)G2] (was a
     broken G2-only formula that understated multi-layer copper loss).
  3. steinmetz_igse — textbook ki and 50%-duty triangle result (was off
     by a factor 40.8).
"""

import math

import numpy as np
import pytest
from scipy import integrate, special

from pfc_design.magnetics.winding import (
    skin_effect_factor, proximity_factor, winding_loss_factor)
from pfc_design.magnetics.steinmetz import (
    steinmetz_igse, steinmetz_ose, _igse_ki)


def kelvin_exact_fr(d: float, delta: float) -> float:
    """Exact round-wire Rac/Rdc from Kelvin functions (reference)."""
    q = d / (math.sqrt(2.0) * delta)
    num = special.ber(q) * special.beip(q) - special.bei(q) * special.berp(q)
    den = special.berp(q) ** 2 + special.beip(q) ** 2
    return (q / 2.0) * num / den


class TestSkinEffectFactor:

    @pytest.mark.parametrize("d_over_delta", [0.5, 1.0, 1.9, 2.0, 2.5, 3.0,
                                              4.0, 8.0, 20.0])
    def test_matches_exact_kelvin_solution(self, d_over_delta):
        got = skin_effect_factor(d_over_delta, 1.0)
        want = kelvin_exact_fr(d_over_delta, 1.0)
        assert got == pytest.approx(want, rel=1e-9)

    def test_never_below_1(self):
        """The old branch point at d/delta=2 returned 0.75 < 1 (Rac<Rdc)."""
        for x in np.linspace(0.01, 20.0, 2000):
            assert skin_effect_factor(x, 1.0) >= 1.0 - 1e-12

    def test_monotone_increasing(self):
        xs = np.linspace(0.01, 20.0, 2000)
        vals = [skin_effect_factor(x, 1.0) for x in xs]
        assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))

    def test_low_frequency_limit_is_1(self):
        assert skin_effect_factor(1e-6, 1.0) == pytest.approx(1.0, abs=1e-9)


class TestProximityFactorDowell:

    @pytest.mark.parametrize("layers,d_over_delta,want", [
        (2, 0.5, 1.01080),
        (2, 1.0, 1.17016),
        (3, 1.0, 1.39400),
        (4, 2.0, 10.10740),
    ])
    def test_matches_dowell(self, layers, d_over_delta, want):
        got = proximity_factor(layers, d_over_delta, 1.0)
        assert got == pytest.approx(want, rel=1e-3)

    def test_single_layer_is_1(self):
        assert proximity_factor(1, 2.0, 1.0) == 1.0

    def test_winding_loss_factor_does_not_double_count_skin(self):
        """Dowell F_R includes skin; for 2 layers f_total must equal f_prox."""
        f_skin, f_prox, f_total = winding_loss_factor(100e3, 1.2e-3,
                                                      n_layers=2)
        assert f_prox > f_skin
        assert f_total == f_prox
        assert f_total > 1.0


class TestIGSE:

    def test_ki_matches_textbook(self):
        """ki = k / ((2*pi)^(a-1) * 2^(b-a) * integral(|cos t|^a))."""
        k, alpha, beta = 3.2e-2, 1.45, 2.35
        I = integrate.quad(lambda t: abs(math.cos(t)) ** alpha,
                           0, 2 * math.pi)[0]
        want = k / ((2 * math.pi) ** (alpha - 1) * 2 ** (beta - alpha) * I)
        assert _igse_ki(k, alpha, beta) == pytest.approx(want, rel=1e-6)

    def test_50pct_triangle_vs_ose(self):
        """iGSE for a 50% duty triangle must be ~0.92 * OSE (was 0.023)."""
        k, alpha, beta, f, B = 3.2e-2, 1.45, 2.35, 100e3, 0.05
        p_ose = steinmetz_ose(f, B, k, alpha, beta)
        p_igse = steinmetz_igse(f, B, k, alpha, beta, duty=0.5)
        assert p_ose == pytest.approx(498.6, rel=1e-3)
        assert p_igse == pytest.approx(460.0, rel=1e-3)
        assert p_igse / p_ose == pytest.approx(0.9226, rel=1e-3)

    def test_duty_05_reduces_to_closed_form(self):
        k, alpha, beta, f, B = 3.2e-2, 1.45, 2.35, 100e3, 0.05
        ki = _igse_ki(k, alpha, beta)
        want = ki * 2.0 ** (alpha + beta) * f ** alpha * B ** beta
        assert steinmetz_igse(f, B, k, alpha, beta) == pytest.approx(want,
                                                                     rel=1e-12)
