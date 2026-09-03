"""Toroidal core specification dataclass."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CoreSpec:
    """Single toroidal core geometry and material properties."""

    manufacturer: str
    part_number: str
    material: str            # e.g. "Kool Mu 60μ", "Sendust CS"
    material_class: str      # "Sendust", "Ferrite", "MPP", "HighFlux", "Amorphous", "Nanocrystalline"
    od_mm: float             # outer diameter
    id_mm: float             # inner diameter
    ht_mm: float             # height
    ae_cm2: float            # effective cross-section area
    le_cm: float             # effective magnetic path length
    ve_cm3: float            # effective volume
    al_nH_per_t2: float      # inductance factor (nH/turn²)
    ui: float                # initial permeability
    bs_T: float              # saturation flux density
    density_g_per_cm3: float = 6.5
    # DC bias curve: polynomial coefficients [c0, c1, c2, c3, c4]
    # %ui = c0 + c1*H + c2*H² + c3*H³ + c4*H⁴  (H in Oe)
    dc_bias_coeffs: list[float] = field(default_factory=lambda: [100, -2.0, 0.04, -0.0003, 0.0])
    price_usd: Optional[float] = None
    source_url: Optional[str] = None

    @property
    def aw_cm2(self) -> float:
        """Window area in cm² (area inside toroid)."""
        return 3.14159 * (self.id_mm / 20) ** 2

    @property
    def ap_cm4(self) -> float:
        """Area product Aw * Ae in cm⁴."""
        return self.aw_cm2 * self.ae_cm2

    @property
    def mlt_cm(self) -> float:
        """Mean length per turn for toroid winding (cm).

        For a rectangular cross-section toroid, the wire wraps around
        the core body. Includes winding buildup allowance (~3x geometric).
        """
        # Geometric MLT: 2*HT + (OD - ID) around rectangular cross-section
        geo_cm = (2 * self.ht_mm + self.od_mm - self.id_mm) / 10
        # Apply buildup factor to account for winding layers and lead-outs
        return geo_cm * 2.8

    @property
    def surface_area_cm2(self) -> float:
        """Approximate surface area for thermal calculations."""
        od_cm = self.od_mm / 10
        ht_cm = self.ht_mm / 10
        id_cm = self.id_mm / 10
        # Outer surface + inner surface + 2 × top/bottom ring area
        outer = 3.14159 * od_cm * ht_cm
        inner = 3.14159 * id_cm * ht_cm
        rings = 2 * 3.14159 * ((od_cm / 2) ** 2 - (id_cm / 2) ** 2)
        return outer + inner + rings

    def __hash__(self):
        return hash((self.manufacturer, self.part_number))

    def __str__(self):
        return f"{self.manufacturer} {self.part_number} ({self.material}, OD:{self.od_mm}mm, Ae:{self.ae_cm2}cm², AL:{self.al_nH_per_t2}nH/t²)"
