"""Efficiency plots."""

import matplotlib.pyplot as plt
import numpy as np


def plot_efficiency_vs_load(power_range_pct: np.ndarray, efficiency_pct: np.ndarray,
                             save_path: str | None = None):
    """Plot efficiency vs load power (%)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(power_range_pct, efficiency_pct, 'b-', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel("Load (%)")
    ax.set_ylabel("Efficiency (%)")
    ax.set_title("Efficiency vs Load")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=max(80, min(efficiency_pct) - 2))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_efficiency_vs_fsw(fsw_range: np.ndarray, efficiency_pct: np.ndarray,
                            save_path: str | None = None):
    """Plot efficiency vs switching frequency."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fsw_range / 1000, efficiency_pct, 'g-', linewidth=2, marker='s', markersize=4)
    ax.set_xlabel("Switching Frequency (kHz)")
    ax.set_ylabel("Efficiency (%)")
    ax.set_title("Efficiency vs Switching Frequency")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
