"""DC bias saturation model for powder cores.

Models the reduction in permeability (and thus inductance) as
DC bias current increases. Based on manufacturer %ui vs H_dc curves.
"""

import numpy as np
from ..core.constants import A_PER_M_TO_Oe


def h_dc_oersted(n_turns: float, i_dc: float, le_cm: float) -> float:
    """Calculate DC magnetizing force in Oersteds.

    H = 0.4 * pi * N * I / le (CGS units)

    Args:
        n_turns: number of turns
        i_dc: DC current in Amps
        le_cm: effective magnetic path length in cm

    Returns:
        H_dc in Oersteds
    """
    return 0.4 * np.pi * n_turns * i_dc / le_cm


def h_dc_si(n_turns: float, i_dc: float, le_m: float) -> float:
    """Calculate DC magnetizing force in A/m (SI)."""
    return n_turns * i_dc / le_m


def percent_permeability(h_oe: float, coeffs: list[float]) -> float:
    """Compute % of initial permeability at given DC bias.

    Uses polynomial fit to manufacturer's %ui vs H_dc curve.

    Args:
        h_oe: DC magnetizing force in Oersteds
        coeffs: polynomial [c0, c1, c2, c3, c4] where
                %ui = c0 + c1*H + c2*H² + c3*H³ + c4*H⁴

    Returns:
        % of initial permeability (clamped to 0-100)
    """
    pct = np.polyval(list(reversed(coeffs)), h_oe)
    return float(np.clip(pct, 1.0, 100.0))


def effective_inductance(l0_uh: float, n_turns: float, i_dc: float,
                          le_cm: float, dc_bias_coeffs: list[float]) -> float:
    """Effective inductance at given DC bias current.

    Args:
        l0_uh: inductance with no DC bias (μH)
        n_turns: number of turns
        i_dc: DC bias current in Amps
        le_cm: magnetic path length in cm
        dc_bias_coeffs: polynomial coefficients for %ui vs H

    Returns:
        Effective inductance in μH
    """
    h = h_dc_oersted(n_turns, i_dc, le_cm)
    pct = percent_permeability(h, dc_bias_coeffs) / 100.0
    return l0_uh * pct


def generate_L_vs_I(l0_uh: float, n_turns: float, le_cm: float,
                    dc_bias_coeffs: list[float], i_max: float,
                    n_pts: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Generate L vs I curve for plotting.

    Args:
        l0_uh: zero-bias inductance in μH
        n_turns: turns
        le_cm: magnetic path length in cm
        dc_bias_coeffs: %ui vs H polynomial
        i_max: maximum current for curve
        n_pts: number of points

    Returns:
        (i_arr, L_arr) — current in A, inductance in μH
    """
    i_arr = np.linspace(0, i_max, n_pts)
    L_arr = np.array([effective_inductance(l0_uh, n_turns, i, le_cm, dc_bias_coeffs)
                       for i in i_arr])
    return i_arr, L_arr


def calculate_b_max(L_uh: float, i_peak: float, n_turns: float, ae_cm2: float) -> float:
    """Calculate peak flux density in Tesla.

    B = L * I / (N * Ae)

    Args:
        L_uh: inductance in μH
        i_peak: peak current in A
        n_turns: number of turns
        ae_cm2: core cross-sectional area in cm²

    Returns:
        Peak flux density in Tesla
    """
    L_h = L_uh * 1e-6
    ae_m2 = ae_cm2 * 1e-4
    return L_h * i_peak / (n_turns * ae_m2)


def calculate_b_ac(L_uh: float, delta_i: float, n_turns: float, ae_cm2: float) -> float:
    """Calculate AC flux density swing (half-amplitude) in Tesla.

    B_ac = L * delta_I / (2 * N * Ae)
    Used for core loss Steinmetz calculations.

    Args:
        L_uh: inductance in μH
        delta_i: peak-to-peak ripple current in A
        n_turns: number of turns
        ae_cm2: core cross-sectional area in cm²

    Returns:
        AC flux density half-swing in Tesla
    """
    return calculate_b_max(L_uh, delta_i / 2, n_turns, ae_cm2)
