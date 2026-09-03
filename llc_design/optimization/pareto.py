"""Small Pareto-front helper for tabular LLC optimization results."""

from __future__ import annotations

import pandas as pd


def pareto_front(df: pd.DataFrame, objectives: tuple[str, ...]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    values = df.loc[:, objectives].to_numpy(float)
    keep = []
    for i, candidate in enumerate(values):
        dominated = False
        for j, other in enumerate(values):
            if i == j:
                continue
            if (other <= candidate).all() and (other < candidate).any():
                dominated = True
                break
        keep.append(not dominated)
    result = df.loc[keep].copy()
    return result.sort_values(list(objectives)).reset_index(drop=True)
