"""Design variable and sweep space definitions."""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class DesignVariable:
    """A single design variable with bounds for sweeping."""
    name: str
    values: np.ndarray | list[float]

    @classmethod
    def from_range(cls, name: str, start: float, stop: float, n_pts: int):
        return cls(name=name, values=np.linspace(start, stop, n_pts))

    @classmethod
    def from_arange(cls, name: str, start: float, stop: float, step: float):
        return cls(name=name, values=np.arange(start, stop + step/2, step))


def default_sweep_space() -> dict[str, np.ndarray]:
    """Default sweep space for PFC optimization.

    Grid: fsw × ripple_ratio × (core candidates)
    """
    from ..magnetics.core_database import CoreDatabase

    db = CoreDatabase()
    candidates = db.top_for_pfc(ae_min_cm2=0.5, ap_min_cm4=1.0,
                                material_class="Sendust", n=5)

    return {
        "fsw": np.arange(40e3, 130e3, 10e3),
        "ripple_ratio": np.linspace(0.15, 0.50, 8),
        "core_idx": list(range(len(candidates))),
    }
