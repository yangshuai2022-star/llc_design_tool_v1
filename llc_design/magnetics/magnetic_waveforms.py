"""Reconstructed LLC magnetic-component waveforms for loss calculation."""

from __future__ import annotations

import math

import numpy as np

from ..core.operating_point import LLCOperatingPoint
from ..core.spec import LLCDesignSpec


def periodic_timebase(frequency_hz: float, samples: int = 2048) -> np.ndarray:
    return np.arange(samples, dtype=float) / (samples * frequency_hz)


def symmetric_triangle(samples: int, peak: float) -> np.ndarray:
    # Starts at -peak, reaches +peak after half a cycle, then returns.
    phase = np.arange(samples, dtype=float) / samples
    unit = np.where(phase < 0.5, -1.0 + 4.0 * phase, 3.0 - 4.0 * phase)
    return peak * unit


def transformer_flux_waveform(op: LLCOperatingPoint, primary_turns: int,
                                effective_area_m2: float,
                                samples: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    b_peak = (op.transformer_square_equivalent_v /
              (4.0 * primary_turns * effective_area_m2 * op.switching_frequency_hz))
    return periodic_timebase(op.switching_frequency_hz, samples), symmetric_triangle(samples, b_peak)


def transformer_current_waveforms(spec: LLCDesignSpec, op: LLCOperatingPoint,
                                  samples: int = 2048) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Approximate primary and secondary winding currents over one period.

    The reflected-load component is sinusoidal.  The magnetizing component is
    triangular because the transformer is driven by an approximately square
    voltage.  The secondary current cancels the primary load ampere-turns;
    therefore the layer-MMF model retains the magnetizing field rather than
    incorrectly adding primary and secondary RMS magnitudes.
    """
    theta = 2.0 * math.pi * np.arange(samples, dtype=float) / samples
    load_primary = math.sqrt(2.0) * op.reflected_load_current_rms_a * np.sin(theta)
    mag_peak = math.sqrt(3.0) * op.magnetizing_current_rms_a
    # Shift the triangular current by a quarter cycle relative to load voltage.
    magnetizing = np.roll(symmetric_triangle(samples, mag_peak), samples // 4)
    primary = load_primary + magnetizing
    primary_rms = float(np.sqrt(np.mean(primary**2)))
    if primary_rms > 0.0:
        # FHA magnitudes and reconstructed shapes are not exactly identical.
        # Scale only the magnetizing residual so the terminal current matches.
        target_sq = op.resonant_current_rms_a**2
        load_sq = op.reflected_load_current_rms_a**2
        mag_target = math.sqrt(max(target_sq - load_sq, 0.0))
        mag_now = float(np.sqrt(np.mean(magnetizing**2)))
        if mag_now > 0.0:
            magnetizing *= mag_target / mag_now
        primary = load_primary + magnetizing
    secondary = -spec.turns_ratio * load_primary
    return primary, secondary, magnetizing


def resonant_inductor_waveforms(op: LLCOperatingPoint, inductance_h: float,
                                turns: int, effective_area_m2: float,
                                samples: int = 2048) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = 2.0 * math.pi * np.arange(samples, dtype=float) / samples
    current = op.resonant_current_peak_a * np.sin(theta)
    flux_density = inductance_h * current / (turns * effective_area_m2)
    return periodic_timebase(op.switching_frequency_hz, samples), current, flux_density
