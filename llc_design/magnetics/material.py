"""Ferrite material models with temperature interpolation and iGSE loss.

The material database intentionally separates geometry from material.  A core
shape can therefore be evaluated with several ferrite grades without copying
geometry records.  Core loss is evaluated from the actual B(t) waveform using
the improved generalized Steinmetz equation (iGSE); the classical Steinmetz
result remains available for validation and manufacturer-curve fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.special import gamma


@dataclass(frozen=True)
class SteinmetzCoefficients:
    temperature_c: float
    k: float
    alpha: float
    beta: float


@dataclass(frozen=True)
class CoreMaterial:
    key: str
    manufacturer: str
    grade: str
    mu_i_25: float
    bsat_points: tuple[tuple[float, float], ...]
    steinmetz_points: tuple[SteinmetzCoefficients, ...]
    frequency_range_hz: tuple[float, float]
    flux_range_t: tuple[float, float]
    core_loss_correction: float = 1.0

    def coefficients_at(self, temperature_c: float) -> SteinmetzCoefficients:
        points = sorted(self.steinmetz_points, key=lambda p: p.temperature_c)
        if temperature_c <= points[0].temperature_c:
            return points[0]
        if temperature_c >= points[-1].temperature_c:
            return points[-1]
        for lo, hi in zip(points[:-1], points[1:]):
            if lo.temperature_c <= temperature_c <= hi.temperature_c:
                x = (temperature_c - lo.temperature_c) / (hi.temperature_c - lo.temperature_c)
                return SteinmetzCoefficients(
                    temperature_c,
                    lo.k + x * (hi.k - lo.k),
                    lo.alpha + x * (hi.alpha - lo.alpha),
                    lo.beta + x * (hi.beta - lo.beta),
                )
        return points[-1]

    def bsat_at(self, temperature_c: float) -> float:
        pts = sorted(self.bsat_points)
        if temperature_c <= pts[0][0]:
            return pts[0][1]
        if temperature_c >= pts[-1][0]:
            return pts[-1][1]
        for (t0, b0), (t1, b1) in zip(pts[:-1], pts[1:]):
            if t0 <= temperature_c <= t1:
                return b0 + (temperature_c - t0) * (b1 - b0) / (t1 - t0)
        return pts[-1][1]

    def steinmetz_density_w_m3(self, frequency_hz: float, b_peak_t: float,
                               temperature_c: float) -> float:
        if frequency_hz <= 0.0 or b_peak_t <= 0.0:
            return 0.0
        c = self.coefficients_at(temperature_c)
        return (self.core_loss_correction * c.k * frequency_hz**c.alpha
                * b_peak_t**c.beta)

    def igse_density_w_m3(self, time_s: Iterable[float], flux_density_t: Iterable[float],
                          temperature_c: float) -> float:
        """Evaluate arbitrary periodic B(t) with the improved GSE.

        B(t) must cover one complete period without repeating the endpoint.
        The periodic derivative is evaluated with a wrapped first difference.
        The normalization is chosen so a sinusoid exactly reproduces the
        classical Steinmetz equation for the same coefficients.
        """
        t = np.asarray(tuple(time_s), dtype=float)
        b = np.asarray(tuple(flux_density_t), dtype=float)
        if len(t) < 16 or len(t) != len(b):
            raise ValueError("time and B arrays must contain the same >=16 samples")
        period = (t[-1] - t[0]) + float(np.median(np.diff(t)))
        if period <= 0.0:
            raise ValueError("waveform period must be positive")
        delta_b = float(b.max() - b.min())
        if delta_b <= 0.0:
            return 0.0
        dt = period / len(t)
        db = np.roll(b, -1) - b
        dbdt = db / dt
        c = self.coefficients_at(temperature_c)
        # Integral of |cos(theta)|^alpha over 0..2pi.
        cos_integral = 2.0 * math.sqrt(math.pi) * gamma((c.alpha + 1.0) / 2.0) / gamma((c.alpha + 2.0) / 2.0)
        denominator = ((2.0 * math.pi) ** (c.alpha - 1.0)
                       * 2.0 ** (c.beta - c.alpha) * cos_integral)
        k_i = c.k / denominator
        mean_rate = float(np.mean(np.abs(dbdt) ** c.alpha))
        return (self.core_loss_correction * k_i * mean_rate
                * delta_b ** (c.beta - c.alpha))

    def range_warnings(self, frequency_hz: float, b_peak_t: float) -> tuple[str, ...]:
        warnings: list[str] = []
        f0, f1 = self.frequency_range_hz
        b0, b1 = self.flux_range_t
        if not f0 <= frequency_hz <= f1:
            warnings.append(f"frequency {frequency_hz/1e3:.1f} kHz outside material fit range {f0/1e3:.1f}..{f1/1e3:.1f} kHz")
        if not b0 <= b_peak_t <= b1:
            warnings.append(f"Bpk {b_peak_t:.3f} T outside material fit range {b0:.3f}..{b1:.3f} T")
        return tuple(warnings)


class MaterialDatabase:
    def __init__(self, path: str | None = None):
        source = Path(path) if path else Path(__file__).parent.parent / "data" / "materials.json"
        data = json.loads(source.read_text(encoding="utf-8"))
        self.metadata = data.get("metadata", {})
        self.materials: list[CoreMaterial] = []
        for entry in data["materials"]:
            sm = tuple(SteinmetzCoefficients(*row) for row in entry["steinmetz_points"])
            self.materials.append(CoreMaterial(
                key=entry["key"], manufacturer=entry["manufacturer"], grade=entry["grade"],
                mu_i_25=float(entry["mu_i_25"]),
                bsat_points=tuple(tuple(map(float, row)) for row in entry["bsat_points"]),
                steinmetz_points=sm,
                frequency_range_hz=tuple(map(float, entry["frequency_range_hz"])),
                flux_range_t=tuple(map(float, entry["flux_range_t"])),
                core_loss_correction=float(entry.get("core_loss_correction", 1.0)),
            ))

    def get(self, key: str) -> CoreMaterial:
        for material in self.materials:
            if material.key.casefold() == key.casefold() or material.grade.casefold() == key.casefold():
                return material
        raise KeyError(key)
