"""Magnetics Inc. High Flux toroid material/core models.

The built-in preset is the High Flux toroid Core Data 254 highlighted in the
user's reference table.  Geometry and AL values are taken from Magnetics'
High Flux core table.  DC-bias and core-loss curve-fit coefficients are from
Magnetics' High Flux material-curves page.

Equations published by Magnetics:
    %ui(H) = 1 / (a + b * H**c), H in Oersted
    Pv     = a * B**b * f**c, B in tesla, f in kHz, Pv in mW/cm^3
"""
from __future__ import annotations

from dataclasses import dataclass
import math


CORE_TABLE_SOURCE_URL = "https://www.mag-inc.com/zh-cn/products/powder-cores/high-flux-cores"
MATERIAL_CURVE_SOURCE_URL = "https://www.mag-inc.com/zh-cn/products/powder-cores/high-flux-cores/high-flux-material-curves"


@dataclass(frozen=True)
class HighFluxCoreGeometry:
    core_data: str
    le_mm: float
    ae_mm2: float
    ve_mm3: float
    od_mm: float
    id_mm: float
    ht_mm: float
    bs_t: float = 1.5

    @property
    def window_area_mm2(self) -> float:
        return math.pi * (0.5 * self.id_mm) ** 2


@dataclass(frozen=True)
class HighFluxMaterial:
    permeability: int
    al_nh_per_t2: float
    dc_a: float
    dc_b: float
    dc_c: float
    loss_a: float
    loss_b: float
    loss_c: float

    def permeability_percent(self, h_oe: float) -> float:
        h = max(float(h_oe), 0.0)
        pct = 1.0 / (self.dc_a + self.dc_b * h ** self.dc_c)
        return min(max(pct, 1.0), 100.0)

    def core_loss_density_mw_cm3(self, b_pk_t: float, frequency_hz: float) -> float:
        b = max(float(b_pk_t), 0.0)
        f_khz = max(float(frequency_hz), 0.0) * 1.0e-3
        if b <= 0.0 or f_khz <= 0.0:
            return 0.0
        return self.loss_a * b ** self.loss_b * f_khz ** self.loss_c


# Magnetics High Flux Core Data 254 (toroid), highlighted in the supplied table.
HIGH_FLUX_254 = HighFluxCoreGeometry(
    core_data="254",
    le_mm=98.4,
    ae_mm2=110.6,
    ve_mm3=10880.0,
    od_mm=40.77,
    id_mm=23.32,
    ht_mm=15.37,
    bs_t=1.5,
)

# AL map for Core Data 254.  Only permeability grades for which the official
# material-curves page also publishes both DC-bias and core-loss fits are made
# selectable by the automatic designer.
_AL_254 = {
    14: 19.0,
    26: 35.0,
    40: 54.0,
    60: 81.0,
    90: 121.0,
    125: 168.0,
    147: 198.0,
}

# DC-bias fit: %ui = 1/(a+b*H^c), H in Oe.
_DC = {
    14: (0.01, 1.657e-09, 2.441),
    26: (0.01, 4.205e-09, 2.426),
    40: (0.01, 1.843e-08, 2.358),
    60: (0.01, 6.413e-08, 2.291),
    90: (0.01, 9.693e-08, 2.391),
    125: (0.01, 1.403e-07, 2.465),
    147: (0.01, 8.155e-08, 2.714),
}

# Core-loss density fit: Pv[mW/cm^3] = a*B[T]^b*f[kHz]^c.
_LOSS = {
    14: (968.56, 2.218, 1.189),
    26: (492.31, 2.218, 1.240),
    40: (185.02, 2.218, 1.398),
    60: (246.54, 2.218, 1.311),
    90: (440.08, 2.218, 1.210),
    125: (184.44, 2.218, 1.428),
    147: (184.44, 2.218, 1.428),
}

HIGH_FLUX_254_MATERIALS: dict[int, HighFluxMaterial] = {
    mu: HighFluxMaterial(mu, _AL_254[mu], *_DC[mu], *_LOSS[mu])
    for mu in _AL_254
}


def high_flux_254_material(permeability: int = 60) -> HighFluxMaterial:
    try:
        return HIGH_FLUX_254_MATERIALS[int(permeability)]
    except KeyError as exc:
        raise ValueError(
            f"High Flux Core Data 254 has no complete built-in loss/bias fit for μ={permeability}. "
            f"Available: {sorted(HIGH_FLUX_254_MATERIALS)}"
        ) from exc


__all__ = [
    "CORE_TABLE_SOURCE_URL",
    "MATERIAL_CURVE_SOURCE_URL",
    "HighFluxCoreGeometry",
    "HighFluxMaterial",
    "HIGH_FLUX_254",
    "HIGH_FLUX_254_MATERIALS",
    "high_flux_254_material",
]
