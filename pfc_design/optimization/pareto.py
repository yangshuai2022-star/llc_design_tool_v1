"""Pareto frontier extraction for multi-objective optimization."""

import numpy as np


def find_pareto(x: np.ndarray, y: np.ndarray,
                minimize: tuple[bool, bool] = (True, True)) -> np.ndarray:
    """Find Pareto-optimal points (non-dominated set).

    Args:
        x, y: objective arrays
        minimize: (minimize_x, minimize_y)

    Returns:
        Boolean mask of Pareto-optimal points
    """
    n = len(x)
    mask = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j or not mask[i]:
                continue
            x_better = (x[j] < x[i]) if minimize[0] else (x[j] > x[i])
            y_better = (y[j] < y[i]) if minimize[1] else (y[j] > y[i])
            x_eq = np.isclose(x[j], x[i])
            y_eq = np.isclose(y[j], y[i])

            if (x_better and y_better) or (x_better and y_eq) or (x_eq and y_better):
                mask[i] = False
                break

    return mask


def tradeoff_ratio(x1: float, y1: float, x2: float, y2: float) -> float:
    """Marginal trade-off ratio between two designs.

    Returns dY/dX (how much Y changes per unit X).
    """
    dx = x2 - x1
    if abs(dx) < 1e-12:
        return float('inf')
    return (y2 - y1) / dx
