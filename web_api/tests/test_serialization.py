from concurrent.futures import CancelledError
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pytest

from llc_design.core.tank import GainNotReachableError
from web_api.errors import map_exception
from web_api.serialization import jsonable


class Mode(Enum):
    FHA = "fha"


@dataclass(frozen=True)
class Sample:
    mode: Mode
    values: np.ndarray


def test_jsonable_handles_engineering_types_recursively():
    value = {
        "sample": Sample(Mode.FHA, np.array([1.0, np.inf])),
        "complex": 1 + 2j,
        "path": Path("plots/bode.png"),
        "scalar": np.float64(3.5),
    }

    assert jsonable(value) == {
        "sample": {"mode": "fha", "values": [1.0, {"nonfinite": "inf"}]},
        "complex": {"real": 1.0, "imag": 2.0},
        "path": "plots/bode.png",
        "scalar": 3.5,
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (float("nan"), {"nonfinite": "nan"}),
        (float("inf"), {"nonfinite": "inf"}),
        (float("-inf"), {"nonfinite": "-inf"}),
    ],
)
def test_jsonable_marks_nonfinite_values(value, expected):
    assert jsonable(value) == expected


def test_jsonable_rejects_unknown_types():
    with pytest.raises(TypeError, match="Unsupported JSON value"):
        jsonable(object())


@pytest.mark.parametrize(
    ("exception", "code", "retryable"),
    [
        (ValueError("bad limit"), "invalid_parameters", False),
        (GainNotReachableError("no point"), "gain_not_reachable", False),
        (KeyError("part"), "resource_not_found", False),
        (TimeoutError("slow"), "task_timeout", True),
        (CancelledError(), "task_cancelled", False),
        (
            ValueError("LLC code-generation stability gate failed"),
            "stability_gate_rejected",
            False,
        ),
    ],
)
def test_map_exception_returns_stable_error_codes(exception, code, retryable):
    detail = map_exception(exception, stage="analysis")

    assert detail.code == code
    assert detail.stage == "analysis"
    assert detail.retryable is retryable
    assert detail.details["exception_type"] == type(exception).__name__
