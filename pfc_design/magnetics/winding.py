"""Winding loss calculations: skin effect and proximity effect."""

import math

import numpy as np
from scipy import special

from ..core.constants import skin_depth


def skin_effect_factor(d_wire_m: float, delta_m: float) -> float:
    """Skin effect AC/DC resistance ratio for a single round conductor.

    Exact Bessel/Kelvin-function solution for a solid round wire:

        q   = d / (sqrt(2) * delta)
        Fr  = (q/2) * [ber(q)*bei'(q) - bei(q)*ber'(q)]
                     / [ber'(q)² + bei'(q)²]

    Valid for all d/delta and always returns Fr >= 1 (no spurious
    Rac < Rdc branch as the old piecewise approximation produced).

    Args:
        d_wire_m: wire diameter in meters
        delta_m: skin depth in meters

    Returns:
        AC resistance factor Fr = Rac/Rdc
    """
    if d_wire_m <= 0 or delta_m <= 0:
        return 1.0
    q = d_wire_m / (math.sqrt(2.0) * delta_m)
    if q < 1e-6:
        return 1.0  # low-frequency limit of the exact solution
    num = special.ber(q) * special.beip(q) - special.bei(q) * special.berp(q)
    den = special.berp(q) ** 2 + special.beip(q) ** 2
    return float(max(1.0, (q / 2.0) * num / den))


def proximity_factor(layers: int, d_wire_m: float, delta_m: float,
                     porosity: float = 0.8) -> float:
    """Dowell's AC resistance factor for a multi-layer winding.

    F_R = phi * [G1 + (2/3) * (m² - 1) * G2],  phi = porosity * d/delta
    G1 = (sinh 2phi + sin 2phi) / (cosh 2phi - cos 2phi)
    G2 = (sinh phi - sin phi) / (cosh phi + cos phi)

    The G1 term is the layer's own skin effect and the G2 term the
    inter-layer proximity effect; F_R already includes both, so
    Rac = Rdc * F_R (callers must not multiply by the skin factor again).
    Single-layer windings return 1.0 and callers use the exact round-wire
    skin factor instead.

    Args:
        layers: number of winding layers
        d_wire_m: conductor thickness/diameter in meters
        delta_m: skin depth in meters
        porosity: winding porosity (conductor width / layer width)

    Returns:
        Dowell AC resistance factor F_R = Rac/Rdc
    """
    if layers <= 1:
        return 1.0  # single layer: no proximity effect
    phi = porosity * (d_wire_m / delta_m)
    g1 = (np.sinh(2 * phi) + np.sin(2 * phi)) / (np.cosh(2 * phi) - np.cos(2 * phi))
    g2 = (np.sinh(phi) - np.sin(phi)) / (np.cosh(phi) + np.cos(phi))
    return float(phi * (g1 + (2.0 / 3.0) * (layers ** 2 - 1) * g2))


def dc_resistance(rho: float, length_m: float, area_m2: float) -> float:
    """DC resistance of a wire segment.

    Args:
        rho: resistivity in Ohm·m
        length_m: wire length in meters
        area_m2: cross-sectional area in m²

    Returns:
        Resistance in Ohms
    """
    return rho * length_m / area_m2


def winding_loss_factor(freq: float, d_wire_m: float, n_layers: int = 2,
                        temperature: float = 100.0) -> tuple[float, float, float]:
    """Compute total AC loss factors for a winding.

    Args:
        freq: switching frequency in Hz
        d_wire_m: wire diameter in meters
        n_layers: number of equivalent layers
        temperature: winding temperature in °C

    Returns:
        (F_skin, F_prox, F_total) where Rac = Rdc * F_total
    """
    delta = skin_depth(freq, temperature)
    f_skin = skin_effect_factor(d_wire_m, delta)
    f_prox = proximity_factor(n_layers, d_wire_m, delta)
    # Dowell's F_R (f_prox) already contains the skin-effect term for
    # multi-layer windings; only single-layer windings use f_skin alone.
    f_total = f_skin if n_layers <= 1 else f_prox
    return f_skin, f_prox, f_total
