import json

import pytest

from llc_design.core.spec import LLCDesignSpec
from llc_design.models.system import LLCSystemAnalyzer
from llc_design.report.export import export_calculation_book


def test_export_produces_complete_strict_json_book(tmp_path):
    paths = export_calculation_book(LLCSystemAnalyzer().analyze(LLCDesignSpec()),
                                    tmp_path)
    raw = paths["result_json"].read_text(encoding="utf-8")
    assert "Infinity" not in raw and "NaN" not in raw
    data = json.loads(raw)
    assert data["feasible"]
    assert len(data["operating_points"]) == 7
    assert data["operating_points"][0]["total_loss_w"] > 0.0
    assert paths["report"].exists()
    assert paths["operating_points"].exists()
    assert paths["loss_breakdown_csv"].exists()
    assert len(list(tmp_path.glob("*.png"))) == 4


def test_export_handles_infinite_required_capacitance(tmp_path):
    spec = LLCDesignSpec().clone(output_cap_esr_ohm=0.2,
                                 output_ripple_limit_vpp=0.05)
    paths = export_calculation_book(LLCSystemAnalyzer().analyze(spec), tmp_path)
    raw = paths["result_json"].read_text(encoding="utf-8")
    assert "Infinity" not in raw and "NaN" not in raw
    data = json.loads(raw)
    assert data["operating_points"][0]["output_ripple_vpp"] > 0.05
