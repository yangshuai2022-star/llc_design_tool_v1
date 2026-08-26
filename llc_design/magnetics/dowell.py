"""Dowell 1-D winding-loss model: AC resistance factor and optimum penetration.

These are the Fourier / Dowell formulas from Chap6 of
《应用于电力电子技术的变压器和电感器理论设计和应用》 (see
``BOOK_TRANSFORMER_INDUCTOR/Problem_6_4.m`` / ``6_6.m`` / ``6_7.m``).

The model assumes a 1-D field and a foil / solid-layer conductor (not Litz).
For Litz windings the strand-level Bessel model in ``litz.py`` is more accurate;
the Dowell model is provided as an alternative for foil/flat wire and as an
analytical cross-check.

All functions are pure and frequency-general: ``delta`` is the layer
thickness divided by the skin depth (a penetration ratio), so the same
formulas apply at 50 Hz and 100 kHz once ``delta`` is scaled by
``skin_depth_m``.
"""

from __future__ import annotations

import math


def _dowell_terms(delta: float, harmonic_n: int, layers_p: int) -> tuple[float, float]:
    """Split the Dowell factor into (skin, proximity) parts.

    ``kpn = skin + proximity`` where ``skin = delta_n * term1`` is the isolated-
    layer internal-skin term and ``proximity = delta_n * 2*(p**2-1)/3 * term2``
    is the multi-layer proximity term.
    """
    dn = delta * math.sqrt(harmonic_n)
    term1 = (math.sinh(2.0 * dn) + math.sin(2.0 * dn)) / (
        math.cosh(2.0 * dn) - math.cos(2.0 * dn)
    )
    term2 = (math.sinh(dn) - math.sin(dn)) / (math.cosh(dn) + math.cos(dn))
    skin = dn * term1
    proximity = dn * (2.0 * (layers_p ** 2 - 1) / 3.0) * term2
    return skin, proximity


def dowell_kpn(delta: float, harmonic_n: int, layers_p: int) -> float:
    """Dowell AC-resistance factor ``k_pn`` for one harmonic of a ``p``-layer winding.

    Parameters
    ----------
    delta:
        Penetration ratio = layer (foil) thickness / skin depth at the
        *fundamental* frequency.
    harmonic_n:
        Harmonic index ``n >= 1``.  The effective penetration ratio at this
        harmonic is ``delta_n = delta * sqrt(n)`` because skin depth shrinks
        as ``1/sqrt(f)``.
    layers_p:
        Number of layers ``p >= 1`` carrying this harmonic field.

    Returns
    -------
    float
        The AC resistance factor ``Rac_n / Rdc`` for this harmonic, where
        ``Rdc`` is the DC resistance of the full foil layer.  ``k_pn -> 1`` as
        ``delta -> 0`` (the DC limit).

    Formula (Eq. 6.69 in the text, as implemented in Problem 6.6)::

        delta_n = delta * sqrt(n)
        k_pn = delta_n * [ (sinh(2*delta_n) + sin(2*delta_n))
                           / (cosh(2*delta_n) - cos(2*delta_n))
                           + 2*(p**2-1)/3 * (sinh(delta_n) - sin(delta_n))
                           / (cosh(delta_n) + cos(delta_n)) ]
    """
    if harmonic_n < 1:
        raise ValueError("harmonic_n must be >= 1")
    if layers_p < 1:
        raise ValueError("layers_p must be >= 1")
    if delta <= 0.0:
        return 0.0
    skin, proximity = _dowell_terms(delta, harmonic_n, layers_p)
    return skin + proximity


def dowell_optimum_delta(
    duty: float,
    layers_p: int,
    rise_time_frac: float | None = None,
) -> float:
    """Closed-form optimum penetration ratio that minimises AC copper loss.

    Parameters
    ----------
    duty:
        Waveform duty ``D`` in ``(0, 1]``.  For a square / resonant waveform
        with no dwell-time reduction use ``D = 0.5`` (half-wave symmetric).
    layers_p:
        Number of layers ``p >= 1``.
    rise_time_frac:
        Optional rise-time as a fraction of the period (``trT`` in the text).
        When ``None`` the rectangular-waveform form is used (Eq. 6.75, the
        ``trT -> 0`` limit); when provided the trapezoidal form is used
        (Problem 6.4 / 6.7).

    Returns
    -------
    float
        Optimum ``delta`` (layer thickness / skin depth at the fundamental).

    Both forms are frequency- and topology-agnostic — they depend only on the
    waveform shape (``D`` and ``trT``) and the layer count ``p``.
    """
    if not 0.0 < duty <= 1.0:
        raise ValueError("duty must be in (0, 1]")
    if layers_p < 1:
        raise ValueError("layers_p must be >= 1")
    denom_base = (5.0 * layers_p ** 2 - 1) / 15.0
    if rise_time_frac is None:
        # Eq. 6.75 (Problem 6.6): rectangular waveform, no rise time.
        #   deltaopt = ( pi^2 * D * (1-D) / (3 * (5p^2-1)/15) )^0.25
        numerator = math.pi ** 2 * duty * (1.0 - duty)
        return (numerator / (3.0 * denom_base)) ** 0.25
    if rise_time_frac < 0.0:
        raise ValueError("rise_time_frac must be >= 0")
    # Problem 6.4 / 6.7: trapezoidal waveform with rise time trT.
    #   deltaopt = ( (D - 4/3*trT) * 2*pi^2*trT / ((5p^2-1)/15) )^0.25
    numerator = (duty - 4.0 / 3.0 * rise_time_frac) * 2.0 * math.pi ** 2 * rise_time_frac
    return (numerator / denom_base) ** 0.25


def dowell_optimum_thickness_mm(
    duty: float,
    layers_p: int,
    frequency_hz: float,
    temperature_c: float = 100.0,
    rise_time_frac: float | None = None,
) -> float:
    """Optimum foil/layer thickness in mm = ``deltaopt * skin_depth``.

    Thin convenience wrapper combining :func:`dowell_optimum_delta` with the
    skin depth from :func:`llc_design.magnetics.litz.skin_depth_m`.
    """
    # Local import to avoid a circular dependency at module import time.
    from .litz import skin_depth_m

    delta_opt = dowell_optimum_delta(duty, layers_p, rise_time_frac)
    return delta_opt * skin_depth_m(frequency_hz, temperature_c) * 1e3


# ---- Foil / solid-layer winding loss ------------------------------------------

from dataclasses import dataclass
from typing import Sequence

from .litz import (
    HarmonicLoss, WindingLossBreakdown, copper_resistivity, skin_depth_m,
)


@dataclass(frozen=True)
class DowellLayer:
    """One physical foil / solid-conductor layer in a Dowell winding stack."""
    label: str                       # "primary" / "secondary" / "inductor"
    turns: int
    conductor_length_m: float        # mean-turn-length * turns
    foil_thickness_m: float          # conductor thickness in the field direction
    foil_width_m: float              # conductor width perpendicular to the field
    current_waveform_a: tuple[float, ...]
    layers_in_group: int             # p: total layers carrying this winding's field


def _fft_rms_phasors_local(waveform: Sequence[float], max_harmonic: int) -> list[complex]:
    """RMS phasor spectrum (DC + harmonics 1..max), matching litz._fft_rms_phasors."""
    import numpy as np
    values = np.asarray(waveform, dtype=float)
    if values.ndim != 1 or len(values) < 32:
        raise ValueError("waveform must contain at least 32 samples")
    coeffs = np.fft.rfft(values) / len(values)
    phasors = [complex(float(values.mean()), 0.0)]
    for h in range(1, max_harmonic + 1):
        if h >= len(coeffs):
            phasors.append(0.0j)
        else:
            phasors.append(math.sqrt(2.0) * complex(coeffs[h]))
    return phasors


def dowell_stack_loss(
    layers: Sequence[DowellLayer],
    fundamental_hz: float,
    *,
    max_harmonic: int = 15,
    temperature_c: float = 100.0,
    termination_resistance_fraction: float = 0.03,
) -> dict[str, WindingLossBreakdown]:
    """AC copper loss for a foil / solid-layer winding via the Dowell 1-D model.

    For each layer the per-harmonic AC resistance factor is
    ``kpn = skin + proximity`` (:func:`_dowell_terms`); the harmonic copper loss
    is ``Rdc * kpn * I_n^2`` where ``Rdc = rho * length / (thickness * width)``
    is the DC resistance of the full foil layer.  The proximity term carries
    the ``(p**2-1)/3`` layer-count factor, so this model captures multi-layer
    proximity but — like the textbook original — does **not** resolve
    interleaving (use ``layered_litz_stack_loss`` for that).  Returns the same
    ``WindingLossBreakdown`` shape as the Litz engine so the GUI's loss chart
    works unchanged.
    """
    if not layers:
        return {}
    sample_count = len(layers[0].current_waveform_a)
    if any(len(l.current_waveform_a) != sample_count for l in layers):
        raise ValueError("all layer waveforms must have equal length")

    rho = copper_resistivity(temperature_c)
    delta_skin = skin_depth_m(fundamental_hz, temperature_c)

    labels = sorted({l.label for l in layers})
    accum: dict[str, dict[str, object]] = {
        label: {"dc": 0.0, "skin": 0.0, "prox": 0.0, "term": 0.0,
                "rms_sq": 0.0, "rdc": 0.0, "harmonics": []}
        for label in labels
    }

    for layer in layers:
        rdc = rho * layer.conductor_length_m / max(
            layer.foil_thickness_m * layer.foil_width_m, 1e-15
        )
        delta = layer.foil_thickness_m / delta_skin if delta_skin > 0 else 0.0
        p = max(1, layer.layers_in_group)
        phasors = _fft_rms_phasors_local(layer.current_waveform_a, max_harmonic)
        import numpy as np
        rms = float(np.sqrt(np.mean(np.asarray(layer.current_waveform_a) ** 2)))
        item = accum[layer.label]
        # DC component (harmonic 0): kpn = 1.
        idc = abs(phasors[0])
        p_dc = idc ** 2 * rdc
        item["dc"] = float(item["dc"]) + p_dc
        item["term"] = float(item["term"]) + termination_resistance_fraction * p_dc
        item["rms_sq"] = float(item["rms_sq"]) + rms ** 2
        item["rdc"] = rdc
        cast_list = item["harmonics"]
        assert isinstance(cast_list, list)
        for h in range(1, max_harmonic + 1):
            i_rms = abs(phasors[h])
            if i_rms <= 0.0:
                continue
            freq = h * fundamental_hz
            skin_f, prox_f = _dowell_terms(delta, h, p)
            p_skin = rdc * skin_f * i_rms ** 2
            p_prox = rdc * prox_f * i_rms ** 2
            # skin_f alone (p=1) is the isolated-conductor AC factor; the p-dependent
            # excess over skin_f is the multi-layer proximity.
            item["skin"] = float(item["skin"]) + p_skin
            item["prox"] = float(item["prox"]) + p_prox
            cast_list.append(HarmonicLoss(
                harmonic=h, frequency_hz=freq, current_rms_a=i_rms,
                field_rms_a_per_m=0.0,  # Dowell does not expose the H field
                dc_component_w=p_dc, skin_increment_w=p_skin, proximity_w=p_prox,
            ))

    results: dict[str, WindingLossBreakdown] = {}
    for label in labels:
        it = accum[label]
        dc = float(it["dc"])
        skin = float(it["skin"])
        prox = float(it["prox"])
        term = float(it["term"])
        total = dc + skin + prox + term
        factor = total / dc if dc > 0 else 1.0
        results[label] = WindingLossBreakdown(
            dc_copper_w=dc, skin_effect_w=skin, external_proximity_w=prox,
            bundle_circulating_w=0.0, termination_w=term, total_w=total,
            effective_ac_factor=factor,
            current_rms_a=math.sqrt(float(it["rms_sq"])) if float(it["rms_sq"]) > 0 else 0.0,
            harmonics=tuple(it["harmonics"]),  # type: ignore[arg-type]
        )
    return results