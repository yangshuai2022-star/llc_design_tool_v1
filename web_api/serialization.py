"""Conversion of engineering values into strict JSON-compatible values."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel


def _finite_float(value: float) -> float | dict[str, str]:
    if math.isfinite(value):
        return value
    if math.isnan(value):
        return {"nonfinite": "nan"}
    return {"nonfinite": "inf" if value > 0 else "-inf"}


def jsonable(value: Any) -> Any:
    """Return a recursively JSON-compatible representation of ``value``.

    JSON has no representation for complex numbers or non-finite IEEE-754
    values.  Those values use explicit marker objects so that an exported
    engineering result is loss-aware instead of silently changing meaning.
    """

    if isinstance(value, Enum):
        return jsonable(value.value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return _finite_float(value)
    if isinstance(value, BaseModel):
        return jsonable(value.model_dump(mode="python"))
    if isinstance(value, pd.DataFrame):
        return {
            "columns": [str(column) for column in value.columns],
            "records": jsonable(value.to_dict(orient="records")),
        }
    if isinstance(value, pd.Series):
        return {"name": jsonable(value.name), "values": jsonable(value.tolist())}
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, complex):
        return {"real": jsonable(float(value.real)), "imag": jsonable(float(value.imag))}
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        if np.issubdtype(value.dtype, np.complexfloating):
            return jsonable(complex(value))
        if np.issubdtype(value.dtype, np.floating):
            return _finite_float(float(value))
        if np.issubdtype(value.dtype, np.integer):
            return int(value)
        if np.issubdtype(value.dtype, np.bool_):
            return bool(value)
        return jsonable(value.item())
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"Unsupported JSON value: mapping key {type(key).__name__}")
            result[key] = jsonable(item)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [jsonable(item) for item in value]
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


__all__ = ["jsonable"]
