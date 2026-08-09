"""Inductor characteristic curves: L vs I, B vs H."""

import matplotlib.pyplot as plt
import numpy as np

from ..magnetics.saturation import generate_L_vs_I


def plot_L_vs_I(l0_uh: float, n_turns: float, le_cm: float,
                dc_bias_coeffs: list[float], i_max: float,
                save_path: str | None = None):
    """Plot inductance vs DC bias current curve."""
    i_arr, L_arr = generate_L_vs_I(l0_uh, n_turns, le_cm, dc_bias_coeffs, i_max)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(i_arr, L_arr, 'b-', linewidth=2)
    ax.axhline(y=l0_uh * 0.8, color='r', linestyle='--', alpha=0.5, label='80% L0')
    ax.axhline(y=l0_uh * 0.5, color='orange', linestyle='--', alpha=0.5, label='50% L0')
    ax.fill_between(i_arr, 0, L_arr, alpha=0.1, color='blue')

    ax.set_xlabel("DC Current (A)")
    ax.set_ylabel("Inductance (μH)")
    ax.set_title(f"Inductance vs DC Bias (L₀ = {l0_uh:.0f}μH, {n_turns}T)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_core_loss_vs_freq(steinmetz_material, b_ac: float,
                           f_min_khz: float = 10, f_max_khz: float = 200,
                           save_path: str | None = None):
    """Plot core loss density vs frequency."""
    f_arr = np.linspace(f_min_khz * 1000, f_max_khz * 1000, 100)

    pv_ose = [steinmetz_material.pv_ose(f, b_ac) for f in f_arr]
    pv_igse = [steinmetz_material.pv_igse(f, b_ac) for f in f_arr]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(f_arr / 1000, pv_ose, 'b-', label='OSE', linewidth=2)
    ax.loglog(f_arr / 1000, pv_igse, 'r--', label='iGSE', linewidth=2)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Core Loss Density (W/m³)")
    ax.set_title(f"Core Loss vs Frequency (B_ac = {b_ac*1000:.0f}mT, {steinmetz_material.name})")
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
