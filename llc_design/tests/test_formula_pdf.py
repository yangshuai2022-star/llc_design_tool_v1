from dataclasses import replace

import pytest

from llc_design.core.spec import LLCDesignSpec, TankParameterMode
from llc_design.core.tank import design_tank
from llc_design.models.system import LLCSystemAnalyzer
from llc_design.report.formula_pdf import build_formula_pdf, build_formula_steps


@pytest.fixture(scope="module")
def baseline_analysis():
    return LLCSystemAnalyzer().analyze(LLCDesignSpec())


def test_formula_trace_exposes_numerical_and_loss_methods(baseline_analysis):
    sections = build_formula_steps(baseline_analysis)
    steps = [step for group in sections.values() for step in group]
    text = "\n".join(
        (step.name + step.formula + step.substitution + step.result + step.method)
        for step in steps
    )

    assert len(steps) >= 40
    assert "Brent's method" in text
    assert "4096 samples" in text
    assert "iGSE" in text
    assert "FFT" in text
    assert "thermal fixed-point" in text


def test_user_defined_tank_report_does_not_claim_auto_synthesis(baseline_analysis):
    tank = baseline_analysis.tank
    user_spec = baseline_analysis.spec.clone(
        parameter_mode=TankParameterMode.USER_DEFINED,
        user_lr_h=tank.lr_h,
        user_cr_f=tank.cr_f,
        user_lm_h=tank.lm_h,
    )
    user_analysis = replace(
        baseline_analysis,
        spec=user_spec,
        tank=design_tank(user_spec),
    )
    tank_steps = build_formula_steps(user_analysis)["Resonant tank synthesis"]
    text = "\n".join(step.name + step.method for step in tank_steps)

    assert "entered tank" in text
    assert "not optimized" in text
    assert "Resonant inductance" not in text


def test_formula_pdf_is_created(baseline_analysis, tmp_path):
    path = build_formula_pdf(baseline_analysis, tmp_path / "worksheet.pdf")

    content = path.read_bytes()
    assert content.startswith(b"%PDF")
    assert len(content) > 10_000
