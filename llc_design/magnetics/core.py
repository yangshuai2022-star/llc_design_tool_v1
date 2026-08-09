"""Ferrite core geometry library and waveform-aware core-loss evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .material import CoreMaterial, MaterialDatabase


@dataclass(frozen=True)
class CoreSpec:
    part_number: str
    shape: str
    family: str
    manufacturer: str
    standard: str
    material_key: str
    purposes: tuple[str, ...]
    ae_mm2: float
    amin_mm2: float
    aw_mm2: float
    ve_mm3: float
    le_mm: float
    window_width_mm: float
    window_height_mm: float
    mlt_primary_mm: float
    mlt_secondary_mm: float
    center_leg_width_mm: float
    thermal_resistance_k_per_w: float
    core_mass_g: float
    cost_usd: float
    material_spec: CoreMaterial

    @property
    def material(self) -> str:
        return self.material_spec.grade

    @property
    def mu_r(self) -> float:
        return self.material_spec.mu_i_25

    @property
    def b_sat_t(self) -> float:
        return self.material_spec.bsat_at(100.0)

    @property
    def ae_m2(self) -> float:
        return self.ae_mm2 * 1e-6

    @property
    def amin_m2(self) -> float:
        return self.amin_mm2 * 1e-6

    @property
    def ve_m3(self) -> float:
        return self.ve_mm3 * 1e-9

    @property
    def le_m(self) -> float:
        return self.le_mm * 1e-3

    @property
    def window_width_m(self) -> float:
        return self.window_width_mm * 1e-3

    def supports(self, purpose: str) -> bool:
        return purpose in self.purposes

    def core_loss_w(self, frequency_hz: float, b_peak_t: float,
                    temperature_c: float = 100.0) -> float:
        """Classical sinusoidal Steinmetz result retained for compatibility."""
        return self.material_spec.steinmetz_density_w_m3(
            frequency_hz, b_peak_t, temperature_c) * self.ve_m3

    def core_loss_waveform_w(self, time_s: Iterable[float], flux_density_t: Iterable[float],
                             temperature_c: float = 100.0) -> float:
        return self.material_spec.igse_density_w_m3(
            time_s, flux_density_t, temperature_c) * self.ve_m3

    def saturation_flux_at(self, temperature_c: float) -> float:
        return self.material_spec.bsat_at(temperature_c)

    def loss_range_warnings(self, frequency_hz: float, b_peak_t: float) -> tuple[str, ...]:
        return self.material_spec.range_warnings(frequency_hz, b_peak_t)


class CoreDatabase:
    def __init__(self, path: str | None = None, material_path: str | None = None):
        source = Path(path) if path else Path(__file__).parent.parent / "data" / "cores.json"
        data = json.loads(source.read_text(encoding="utf-8"))
        self.metadata = data.get("metadata", {})
        self.material_db = MaterialDatabase(material_path)
        self.cores: list[CoreSpec] = []
        for entry in data["cores"]:
            material = self.material_db.get(entry["material_key"])
            self.cores.append(CoreSpec(
                **{k: v for k, v in entry.items() if k not in {"purposes"}},
                purposes=tuple(entry["purposes"]), material_spec=material,
            ))

    def for_purpose(self, purpose: str, families: Iterable[str] | None = None) -> list[CoreSpec]:
        family_set = {f.casefold() for f in families} if families else None
        return [core for core in self.cores
                if core.supports(purpose)
                and (family_set is None or core.family.casefold() in family_set)]

    def by_family(self, family: str) -> list[CoreSpec]:
        return [core for core in self.cores if core.family.casefold() == family.casefold()]

    def get(self, part_number: str) -> CoreSpec:
        for core in self.cores:
            if core.part_number.casefold() == part_number.casefold():
                return core
        raise KeyError(part_number)

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(sorted({core.family for core in self.cores}))
