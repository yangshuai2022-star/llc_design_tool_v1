"""Local core database: load, query, and filter toroidal core specifications."""

import json
from pathlib import Path
from typing import Optional

import numpy as np

from .core_entry import CoreSpec
from .steinmetz import SteinmetzMaterial

DATA_DIR = Path(__file__).parent.parent / "data"


class CoreDatabase:
    """Searchable database of toroidal core specifications."""

    def __init__(self, cores_file: Optional[str] = None):
        if cores_file is None:
            cores_file = str(DATA_DIR / "cores.json")
        self._cores: list[CoreSpec] = []
        self._steinmetz: dict[str, SteinmetzMaterial] = {}
        self._load_cores(cores_file)
        self._load_steinmetz()

    def _load_cores(self, path: str):
        with open(path) as f:
            data = json.load(f)
        for entry in data["cores"]:
            self._cores.append(CoreSpec(
                manufacturer=entry["manufacturer"],
                part_number=entry["part_number"],
                material=entry["material"],
                material_class=entry["material_class"],
                od_mm=entry["od_mm"],
                id_mm=entry["id_mm"],
                ht_mm=entry["ht_mm"],
                ae_cm2=entry["ae_cm2"],
                le_cm=entry["le_cm"],
                ve_cm3=entry["ve_cm3"],
                al_nH_per_t2=entry["al_nH_per_t2"],
                ui=entry["ui"],
                bs_T=entry["bs_T"],
                density_g_per_cm3=entry.get("density_g_per_cm3", 6.5),
                dc_bias_coeffs=entry.get("dc_bias_coeffs", [100, -2.0, 0.04, -0.0003, 0.0]),
                price_usd=entry.get("price_usd"),
                source_url=entry.get("source_url"),
            ))

    def _load_steinmetz(self):
        path = str(DATA_DIR / "steinmetz_coefficients.json")
        with open(path) as f:
            data = json.load(f)
        for name, coeffs in data["materials"].items():
            self._steinmetz[name] = SteinmetzMaterial(
                name=name, k=coeffs["k"], alpha=coeffs["alpha"], beta=coeffs["beta"],
                f_min_khz=coeffs["f_range_kHz"][0], f_max_khz=coeffs["f_range_kHz"][1]
            )

    @property
    def cores(self) -> list[CoreSpec]:
        return self._cores

    def get_steinmetz(self, material_name: str) -> Optional[SteinmetzMaterial]:
        """Get Steinmetz material model by name, with fuzzy matching."""
        if material_name in self._steinmetz:
            return self._steinmetz[material_name]
        # Try case-insensitive
        for name, mat in self._steinmetz.items():
            if name.lower() == material_name.lower():
                return mat
        # Try substring match
        for name, mat in self._steinmetz.items():
            if material_name.lower() in name.lower():
                return mat
        return None

    def query(self,
              material_class: Optional[str] = None,
              od_min_mm: Optional[float] = None,
              od_max_mm: Optional[float] = None,
              ae_min_cm2: Optional[float] = None,
              ae_max_cm2: Optional[float] = None,
              al_min: Optional[float] = None,
              al_max: Optional[float] = None,
              ui_min: Optional[float] = None,
              ui_max: Optional[float] = None,
              bs_min_T: Optional[float] = None,
              ap_min_cm4: Optional[float] = None,
              max_results: int = 50) -> list[CoreSpec]:
        """Search cores by criteria. All parameters are optional filters."""
        results = []
        for core in self._cores:
            if material_class and core.material_class.lower() != material_class.lower():
                continue
            if od_min_mm is not None and core.od_mm < od_min_mm:
                continue
            if od_max_mm is not None and core.od_mm > od_max_mm:
                continue
            if ae_min_cm2 is not None and core.ae_cm2 < ae_min_cm2:
                continue
            if ae_max_cm2 is not None and core.ae_cm2 > ae_max_cm2:
                continue
            if al_min is not None and core.al_nH_per_t2 < al_min:
                continue
            if al_max is not None and core.al_nH_per_t2 > al_max:
                continue
            if ui_min is not None and core.ui < ui_min:
                continue
            if ui_max is not None and core.ui > ui_max:
                continue
            if bs_min_T is not None and core.bs_T < bs_min_T:
                continue
            if ap_min_cm4 is not None and core.ap_cm4 < ap_min_cm4:
                continue
            results.append(core)
        return results[:max_results]

    def find_similar(self, od_mm: float, id_mm: float, ht_mm: float,
                     material_class: Optional[str] = None, tolerance: float = 0.15) -> list[CoreSpec]:
        """Find cores with similar dimensions (within tolerance fraction)."""
        results = []
        for core in self._cores:
            if material_class and core.material_class.lower() != material_class.lower():
                continue
            od_err = abs(core.od_mm - od_mm) / od_mm
            id_err = abs(core.id_mm - id_mm) / id_mm if id_mm > 0 else 1.0
            ht_err = abs(core.ht_mm - ht_mm) / ht_mm if ht_mm > 0 else 1.0
            avg_err = (od_err + id_err + ht_err) / 3
            if avg_err < tolerance:
                results.append((core, avg_err))
        results.sort(key=lambda x: x[1])
        return [r[0] for r in results]

    def get_by_part_number(self, part_number: str) -> Optional[CoreSpec]:
        for core in self._cores:
            if core.part_number.lower() == part_number.lower():
                return core
        return None

    def top_for_pfc(self, ae_min_cm2: float = 0.5, ap_min_cm4: float = 1.0,
                    material_class: str = "Sendust", n: int = 10) -> list[CoreSpec]:
        """Quick access to top cores for PFC inductor design."""
        return self.query(material_class=material_class, ae_min_cm2=ae_min_cm2,
                          ap_min_cm4=ap_min_cm4, max_results=n)
