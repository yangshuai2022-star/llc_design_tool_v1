from concurrent.futures import CancelledError
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import Enum, IntEnum
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

from llc_design.core.tank import GainNotReachableError
from web_api.errors import map_exception
from web_api.serialization import jsonable


class Mode(Enum):
    FHA = "fha"


class Channel(IntEnum):
    CURRENT = 1


class SampleModel(BaseModel):
    name: str
    value: float


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


def test_jsonable_supports_mapping_sequence_dates_pandas_and_models():
    timestamp = pd.Timestamp("2026-01-02T03:04:05Z")
    value = {
        "enum": Channel.CURRENT,
        "mapping": MappingProxyType({"x": 1}),
        "sequence": range(3),
        "date": date(2026, 1, 2),
        "datetime": datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        "timestamp": timestamp,
        "frame": pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}),
        "series": pd.Series([2.0, 3.0], name="gain"),
        "model": SampleModel(name="Lr", value=1.0),
        "path": Path("relative/report.pdf"),
        "absolute_path": Path("/Users/secret/report.pdf"),
    }

    output = jsonable(value)
    assert output["enum"] == 1
    assert output["mapping"] == {"x": 1}
    assert output["sequence"] == [0, 1, 2]
    assert output["date"] == "2026-01-02"
    assert output["datetime"] == "2026-01-02T03:04:05+00:00"
    assert output["timestamp"] == "2026-01-02T03:04:05+00:00"
    assert output["frame"] == {"columns": ["a", "b"], "records": [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]}
    assert output["series"] == {"name": "gain", "values": [2.0, 3.0]}
    assert output["model"] == {"name": "Lr", "value": 1.0}
    assert output["path"] == "relative/report.pdf"
    assert output["absolute_path"] == "/Users/secret/report.pdf"


def test_jsonable_rejects_unknown_mapping_keys():
    with pytest.raises(TypeError, match="Unsupported JSON value"):
        jsonable({object(): "not safe"})


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


def test_map_exception_covers_engineering_failure_taxonomy_without_leaking_paths():
    from llc_design.dynamics.plant import PlantModelError

    class ExpiredError(RuntimeError):
        pass

    cases = [
        (ValueError("No feasible design found"), "core_no_solution"),
        (KeyError("device MOSFET-X missing"), "device_not_found"),
        (RuntimeError("No inductor candidate found"), "inductor_no_solution"),
        (ValueError("No suitable core found in database"), "magnetics_no_solution"),
        (PlantModelError("steady-state solve failed: normalized residual=1e-2"), "simulation_non_convergence"),
        (ValueError("closed-loop is unstable"), "unstable_loop"),
        (ExpiredError("job expired"), "task_expired"),
        (RuntimeError("internal failure at /Users/secret/report.json"), "internal_error"),
    ]
    for exception, code in cases:
        detail = map_exception(exception, stage="analysis")
        assert detail.code == code
        assert detail.details["exception_type"] == type(exception).__name__
        assert "/Users/secret" not in str(detail.details)
        assert "message" in detail.details


def test_map_exception_redacts_windows_absolute_paths():
    detail = map_exception(RuntimeError(r"failed to read C:\Users\secret\file.json"), stage="export")

    assert detail.code == "internal_error"
    assert "C:\\Users\\secret" not in str(detail.details)
    assert "<path>" in detail.details["message"]
