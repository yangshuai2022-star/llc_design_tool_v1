"""Physical constants for PFC design calculations."""

import numpy as np

# Magnetic constant
MU0 = 4 * np.pi * 1e-7  # H/m

# Copper resistivity at 20°C
RHO_CU_20 = 1.72e-8  # Ohm·m

# Copper temperature coefficient
ALPHA_CU = 0.00393  # 1/K

# Conversion factors
Oe_TO_A_PER_M = 79.57747  # 1 Oe = 79.577 A/m
A_PER_M_TO_Oe = 1 / Oe_TO_A_PER_M
T_TO_GAUSS = 10000.0
GAUSS_TO_T = 1e-4
CM2_TO_M2 = 1e-4
MM2_TO_M2 = 1e-6
UH_TO_H = 1e-6
NH_TO_H = 1e-9


def rho_cu(temperature: float = 100.0) -> float:
    """Copper resistivity at given temperature (°C)."""
    return RHO_CU_20 * (1 + ALPHA_CU * (temperature - 20))


def skin_depth(freq: float, temperature: float = 100.0) -> float:
    """Skin depth in meters at given frequency and temperature."""
    rho = rho_cu(temperature)
    return np.sqrt(rho / (np.pi * freq * MU0))
