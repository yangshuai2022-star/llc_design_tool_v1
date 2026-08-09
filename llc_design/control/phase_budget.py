"""Gain/phase budget helpers for interactive Bode diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class PhaseBudgetEntry:
    key: str
    label: str
    gain_db: float
    phase_deg: float


def _interp_log_frequency(frequencies_hz: np.ndarray, values: np.ndarray, frequency_hz: float) -> float:
    f = np.asarray(frequencies_hz, dtype=float)
    v = np.asarray(values, dtype=float)
    x = math.log10(max(float(frequency_hz), float(f[0])))
    return float(np.interp(x, np.log10(f), v))


def phase_budget(
    frequencies_hz: np.ndarray,
    responses: Mapping[str, np.ndarray],
    labels: Mapping[str, str],
    frequency_hz: float,
    keys: list[str] | tuple[str, ...],
) -> tuple[PhaseBudgetEntry, ...]:
    """Return per-block gain/phase at one cursor/crossover frequency."""
    out: list[PhaseBudgetEntry] = []
    for key in keys:
        if key not in responses:
            continue
        response = np.asarray(responses[key], dtype=complex)
        gain_db = 20.0 * np.log10(np.maximum(np.abs(response), 1e-300))
        phase_deg = np.unwrap(np.angle(response)) * 180.0 / math.pi
        out.append(PhaseBudgetEntry(
            key=key,
            label=labels.get(key, key),
            gain_db=_interp_log_frequency(frequencies_hz, gain_db, frequency_hz),
            phase_deg=_interp_log_frequency(frequencies_hz, phase_deg, frequency_hz),
        ))
    return tuple(out)


__all__ = ["PhaseBudgetEntry", "phase_budget"]
