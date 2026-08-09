"""Sweep visualization: heatmaps and trade-off contours."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def heatmap(df: pd.DataFrame, x_col: str, y_col: str, z_col: str,
            title: str = "Design Sweep Heatmap",
            cmap: str = "RdYlGn_r", save_path: str | None = None):
    """Heatmap of a sweep parameter over two design variables."""
    pivot = df.pivot_table(values=z_col, index=y_col, columns=x_col, aggfunc='mean')

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(pivot.values, aspect='auto', origin='lower', cmap=cmap)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.0f}" for v in pivot.columns], rotation=45)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.0f}" for v in pivot.index])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)

    cbar = plt.colorbar(im)
    cbar.set_label(z_col)
    ax.set_title(title, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def pareto_frontier(x: np.ndarray, y: np.ndarray,
                    x_label: str = "Objective 1", y_label: str = "Objective 2",
                    title: str = "Pareto Frontier",
                    minimize: tuple[bool, bool] = (True, True),
                    save_path: str | None = None):
    """Plot Pareto frontier for two-objective optimization."""
    # Find Pareto frontier
    n = len(x)
    pareto_mask = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            x_better = x[j] < x[i] if minimize[0] else x[j] > x[i]
            y_better = y[j] < y[i] if minimize[1] else y[j] > y[i]
            if x_better and y_better:
                pareto_mask[i] = False
                break

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, alpha=0.3, s=20, label='Designs')
    ax.scatter(x[pareto_mask], y[pareto_mask], c='red', s=40, label='Pareto Front')
    # Sort Pareto points and connect
    idx = np.argsort(x[pareto_mask])
    ax.plot(x[pareto_mask][idx], y[pareto_mask][idx], 'r-', linewidth=2)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
