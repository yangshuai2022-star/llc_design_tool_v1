from dataclasses import replace
from pathlib import Path
import shutil
import subprocess

import pytest

from llc_design.control.analysis import build_small_signal_analysis
from llc_design.control.digital_loop import PIControllerConfig, build_digital_loop_analysis
from llc_design.core.spec import LLCDesignSpec
from llc_design.models.system import LLCSystemAnalyzer
from pfc_design.control import PFCControlLabConfig, build_pfc_control_lab_analysis
from pfc_design.vienna import ViennaControlLabConfig, build_vienna_control_lab_analysis
from power_codegen import generate_llc_control_code, generate_ttpl_control_code, generate_vienna_control_code


def _compile(folder: Path, sources: list[str]) -> None:
    compiler = shutil.which("gcc") or shutil.which("clang")
    if compiler is None:
        pytest.skip("no C compiler available")
    for source in sources:
        subprocess.run(
            [compiler, "-std=c99", "-Wall", "-Wextra", "-Werror", "-Wdouble-promotion", "-c", source],
            cwd=folder,
            check=True,
            capture_output=True,
            text=True,
        )


def test_ttpl_codegen_compiles_and_uses_stable_autotuned_loop(tmp_path: Path):
    base = PFCControlLabConfig()
    cfg = replace(base, current_controller=PIControllerConfig(kp=0.00854059, ti_s=0.00064608, sample_time_s=20e-6, output_min=-2.0, output_max=0.98), frequency_points=220)
    analysis = build_pfc_control_lab_analysis(cfg)
    result = generate_ttpl_control_code(analysis, tmp_path / "ttpl")
    assert result.validation.passed
    assert (result.directory / "ttpl_control.c").exists()
    text = (result.directory / "ttpl_control.c").read_text(encoding="utf-8")
    assert "TTPL_DUTY_FF_ENABLED" in text
    assert "state->i_ref = state->gcmd * vac_abs" in text
    _compile(result.directory, ["control_runtime.c", "ttpl_control.c", "isr_template.c"])


def test_vienna_codegen_compiles(tmp_path: Path):
    cfg = replace(ViennaControlLabConfig(), frequency_points=220)
    analysis = build_vienna_control_lab_analysis(cfg)
    result = generate_vienna_control_code(analysis, tmp_path / "vienna")
    assert result.validation.passed
    text = (result.directory / "vienna_control.c").read_text(encoding="utf-8")
    assert "Vienna_ModulatePhase" in text
    assert "output->duty_a" in text
    _compile(result.directory, ["control_runtime.c", "vienna_control.c", "isr_template.c"])


def test_llc_codegen_compiles(tmp_path: Path):
    spec = LLCDesignSpec()
    system = LLCSystemAnalyzer().analyze(spec)
    small = build_small_signal_analysis(spec, system_analysis=system, sample_time_s=20e-6)
    loop = build_digital_loop_analysis(
        small,
        controller_config=PIControllerConfig(kp=0.002, ti_s=3e-3, sample_time_s=20e-6),
    )
    result = generate_llc_control_code(loop, tmp_path / "llc", require_stable=False)
    assert (result.directory / "llc_control.c").exists()
    text = (result.directory / "llc_control.c").read_text(encoding="utf-8")
    assert "LLC_LUT" in text
    assert "output->fsw_hz" in text
    _compile(result.directory, ["control_runtime.c", "llc_control.c", "isr_template.c"])
