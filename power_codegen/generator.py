"""Portable C99/float32 code generation for validated digital-power control loops.

The generated code intentionally stops at the semantic control boundary:

* inputs are engineering-unit measurements already provided by the BSP;
* outputs are semantic duty/frequency commands for the BSP to apply;
* no ADC/PWM/GPIO/interrupt-controller register programming is emitted.

The generated runtime is valid C99 and uses single-precision arithmetic only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from llc_design.control.digital_loop import (
    ControllerKind,
    DigitalLoopAnalysis,
    FMLUTMode,
    PIControllerConfig,
    PIFControllerConfig,
    TwoP2ZControllerConfig,
    controller_kind,
)
from pfc_design.control.analysis import PFCControlLabAnalysis
from pfc_design.control.config import PFCControlLabConfig
from pfc_design.vienna.analysis import ViennaControlLabAnalysis
from pfc_design.vienna.config import ViennaControlLabConfig


ControllerConfig = PIControllerConfig | PIFControllerConfig | TwoP2ZControllerConfig


@dataclass(frozen=True)
class CodegenValidation:
    passed: bool
    checks: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class CodegenResult:
    directory: Path
    files: dict[str, Path]
    validation: CodegenValidation


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _cf(value: float) -> str:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"cannot emit non-finite C constant: {value}")
    text = f"{value:.9g}"
    if "e" not in text.lower() and "." not in text:
        text += ".0"
    return text + "f"


def _cu(value: int) -> str:
    return f"{int(value)}u"


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return path


def _divider(base_hz: float, sub_hz: float, label: str) -> int:
    if base_hz <= 0.0 or sub_hz <= 0.0:
        raise ValueError(f"{label}: rates must be positive")
    ratio = base_hz / sub_hz
    rounded = int(round(ratio))
    if rounded < 1 or not math.isclose(ratio, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{label}: {base_hz:g}/{sub_hz:g} is not an integer multi-rate divider")
    return rounded


def _controller_initializer(name: str, config: ControllerConfig) -> str:
    kind = controller_kind(config)
    if kind == ControllerKind.PI:
        cfg = config
        return f'''    CTRL_InitPI(&state->{name}, {_cf(cfg.kp)}, {_cf(cfg.sample_time_s/(2.0*cfg.ti_s))}, {_cf(cfg.output_min)}, {_cf(cfg.output_max)});'''
    if kind == ControllerKind.PIF:
        cfg = config
        return f'''    CTRL_InitPIF(&state->{name}, {_cf(cfg.kp)}, {_cf(cfg.sample_time_s/(2.0*cfg.ti_s))}, {_cf(cfg.alpha)}, {_cf(cfg.output_min)}, {_cf(cfg.output_max)});'''
    cfg = config
    return f'''    CTRL_Init2P2Z(&state->{name}, {_cf(cfg.b0)}, {_cf(cfg.b1)}, {_cf(cfg.b2)}, {_cf(cfg.a1)}, {_cf(cfg.a2)}, {_cf(cfg.output_min)}, {_cf(cfg.output_max)});'''


def _controller_kind_macro(config: ControllerConfig) -> str:
    kind = controller_kind(config)
    return {
        ControllerKind.PI: "CTRL_KIND_PI",
        ControllerKind.PIF: "CTRL_KIND_PIF",
        ControllerKind.TWO_P_TWO_Z: "CTRL_KIND_2P2Z",
    }[kind]


def _runtime_header() -> str:
    return r'''#ifndef CONTROL_RUNTIME_H
#define CONTROL_RUNTIME_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef float real32_t;

typedef enum
{
    CTRL_KIND_PI = 0,
    CTRL_KIND_PIF = 1,
    CTRL_KIND_2P2Z = 2
} CTRL_Kind_t;

typedef struct
{
    CTRL_Kind_t kind;
    real32_t kp;
    real32_t ki2;
    real32_t alpha;
    real32_t out_min;
    real32_t out_max;
    real32_t error_prev;
    real32_t i_state;
    real32_t out_prev;
    real32_t b0;
    real32_t b1;
    real32_t b2;
    real32_t a1;
    real32_t a2;
    real32_t x1;
    real32_t x2;
    real32_t y1;
    real32_t y2;
} CTRL_State_t;

real32_t CTRL_Clamp(real32_t value, real32_t minimum, real32_t maximum);
void CTRL_Reset(CTRL_State_t *state);
void CTRL_InitPI(CTRL_State_t *state, real32_t kp, real32_t ki2, real32_t out_min, real32_t out_max);
void CTRL_InitPIF(CTRL_State_t *state, real32_t kp, real32_t ki2, real32_t alpha, real32_t out_min, real32_t out_max);
void CTRL_Init2P2Z(CTRL_State_t *state, real32_t b0, real32_t b1, real32_t b2, real32_t a1, real32_t a2, real32_t out_min, real32_t out_max);
real32_t CTRL_Step(CTRL_State_t *state, real32_t error);

#ifdef __cplusplus
}
#endif

#endif /* CONTROL_RUNTIME_H */
'''


def _runtime_source() -> str:
    return r'''#include "control_runtime.h"

#include <stddef.h>
#include <string.h>

real32_t CTRL_Clamp(real32_t value, real32_t minimum, real32_t maximum)
{
    if(value > maximum) { return maximum; }
    if(value < minimum) { return minimum; }
    return value;
}

void CTRL_Reset(CTRL_State_t *state)
{
    CTRL_Kind_t kind;
    real32_t kp, ki2, alpha, out_min, out_max;
    real32_t b0, b1, b2, a1, a2;
    if(state == NULL) { return; }
    kind = state->kind;
    kp = state->kp; ki2 = state->ki2; alpha = state->alpha;
    out_min = state->out_min; out_max = state->out_max;
    b0 = state->b0; b1 = state->b1; b2 = state->b2;
    a1 = state->a1; a2 = state->a2;
    (void)memset(state, 0, sizeof(*state));
    state->kind = kind;
    state->kp = kp; state->ki2 = ki2; state->alpha = alpha;
    state->out_min = out_min; state->out_max = out_max;
    state->b0 = b0; state->b1 = b1; state->b2 = b2;
    state->a1 = a1; state->a2 = a2;
}

void CTRL_InitPI(CTRL_State_t *state, real32_t kp, real32_t ki2, real32_t out_min, real32_t out_max)
{
    if(state == NULL) { return; }
    (void)memset(state, 0, sizeof(*state));
    state->kind = CTRL_KIND_PI;
    state->kp = kp;
    state->ki2 = ki2;
    state->alpha = 1.0f;
    state->out_min = out_min;
    state->out_max = out_max;
}

void CTRL_InitPIF(CTRL_State_t *state, real32_t kp, real32_t ki2, real32_t alpha, real32_t out_min, real32_t out_max)
{
    CTRL_InitPI(state, kp, ki2, out_min, out_max);
    if(state != NULL)
    {
        state->kind = CTRL_KIND_PIF;
        state->alpha = alpha;
    }
}

void CTRL_Init2P2Z(CTRL_State_t *state, real32_t b0, real32_t b1, real32_t b2, real32_t a1, real32_t a2, real32_t out_min, real32_t out_max)
{
    if(state == NULL) { return; }
    (void)memset(state, 0, sizeof(*state));
    state->kind = CTRL_KIND_2P2Z;
    state->b0 = b0; state->b1 = b1; state->b2 = b2;
    state->a1 = a1; state->a2 = a2;
    state->out_min = out_min; state->out_max = out_max;
}

static real32_t CTRL_StepPI(CTRL_State_t *state, real32_t error, int filtered)
{
    real32_t i_new;
    real32_t raw;
    real32_t saturated;
    real32_t output;

    i_new = state->i_state + state->ki2 * (error + state->error_prev);
    raw = state->kp * (error + i_new);
    saturated = CTRL_Clamp(raw, state->out_min, state->out_max);

    if(!((raw > state->out_max && error > 0.0f) ||
         (raw < state->out_min && error < 0.0f)))
    {
        state->i_state = i_new;
    }

    if(filtered != 0)
    {
        output = (1.0f - state->alpha) * state->out_prev + state->alpha * saturated;
        state->out_prev = output;
    }
    else
    {
        output = saturated;
    }
    state->error_prev = error;
    return output;
}

static real32_t CTRL_Step2P2Z(CTRL_State_t *state, real32_t error)
{
    real32_t output;
    output = -state->a1 * state->y1 - state->a2 * state->y2
             + state->b0 * error + state->b1 * state->x1 + state->b2 * state->x2;
    output = CTRL_Clamp(output, state->out_min, state->out_max);
    state->x2 = state->x1;
    state->x1 = error;
    state->y2 = state->y1;
    state->y1 = output;
    return output;
}

real32_t CTRL_Step(CTRL_State_t *state, real32_t error)
{
    if(state == NULL) { return 0.0f; }
    switch(state->kind)
    {
        case CTRL_KIND_PIF:
            return CTRL_StepPI(state, error, 1);
        case CTRL_KIND_2P2Z:
            return CTRL_Step2P2Z(state, error);
        case CTRL_KIND_PI:
        default:
            return CTRL_StepPI(state, error, 0);
    }
}
'''


def _common_files(out: Path) -> dict[str, Path]:
    return {
        "runtime_h": _write(out / "control_runtime.h", _runtime_header()),
        "runtime_c": _write(out / "control_runtime.c", _runtime_source()),
    }


def _margin_ok(value: float | None, minimum: float) -> bool:
    return value is not None and math.isfinite(float(value)) and float(value) >= minimum


def validate_ttpl_codegen(analysis: PFCControlLabAnalysis) -> CodegenValidation:
    checks: list[str] = []
    warnings: list[str] = []
    cm = analysis.current_loop.margins
    vm = analysis.voltage_loop.margins
    current_ok = _margin_ok(cm.phase_margin_deg, 45.0)
    voltage_ok = _margin_ok(vm.phase_margin_deg, 0.0)
    checks.append(f"current-loop phase margin = {cm.phase_margin_deg!r} deg ({'PASS' if current_ok else 'FAIL'})")
    checks.append(f"voltage-loop phase margin = {vm.phase_margin_deg!r} deg ({'PASS' if voltage_ok else 'FAIL'})")
    if vm.phase_margin_deg is not None and vm.phase_margin_deg < 30.0:
        warnings.append(f"voltage-loop phase margin is only {vm.phase_margin_deg:.3f} deg; generate as a stable baseline, then tune the outer loop")
    if cm.gain_margin_db is not None and cm.gain_margin_db < 6.0:
        warnings.append(f"current-loop gain margin is only {cm.gain_margin_db:.3f} dB")
    warnings.extend(analysis.warnings)
    return CodegenValidation(current_ok and voltage_ok, tuple(checks), tuple(warnings))


def validate_vienna_codegen(analysis: ViennaControlLabAnalysis) -> CodegenValidation:
    loops = (
        ("current", analysis.current_loop.margins.phase_margin_deg, 40.0),
        ("voltage", analysis.voltage_loop.margins.phase_margin_deg, 30.0),
        ("balance", analysis.balance_loop.margins.phase_margin_deg, 30.0),
    )
    checks = []
    passed = True
    for name, margin, minimum in loops:
        ok = _margin_ok(margin, minimum)
        passed &= ok
        checks.append(f"{name}-loop phase margin = {margin!r} deg ({'PASS' if ok else 'FAIL'})")
    return CodegenValidation(bool(passed), tuple(checks), tuple(analysis.warnings))


def validate_llc_codegen(analysis: DigitalLoopAnalysis) -> CodegenValidation:
    margin = analysis.margins_nominal_delay.phase_margin_deg
    phase_ok = _margin_ok(margin, 30.0)
    poles_ok = analysis.discrete_approximation.stable
    checks = (
        f"nominal phase margin = {margin!r} deg ({'PASS' if phase_ok else 'FAIL'})",
        f"all-discrete closed-loop poles stable = {poles_ok} ({'PASS' if poles_ok else 'FAIL'})",
    )
    return CodegenValidation(bool(phase_ok and poles_ok), checks, tuple(analysis.warnings))


def _emit_validation(out: Path, validation: CodegenValidation) -> Path:
    lines = ["CODE GENERATION VALIDATION", "", f"Overall: {'PASS' if validation.passed else 'FAIL'}", "", "Checks:"]
    lines.extend(f"- {item}" for item in validation.checks)
    if validation.warnings:
        lines += ["", "Warnings:"] + [f"- {item}" for item in validation.warnings]
    return _write(out / "stability_report.txt", "\n".join(lines))


def _emit_readme(out: Path, topology: str, base_rate_hz: float, outputs: Iterable[str]) -> Path:
    output_lines = "\n".join(f"- `{item}`" for item in outputs)
    return _write(out / "README.md", f'''# Generated {topology} control core

This folder contains **C99 / single-precision float** control code generated by Power Design Tool.

## Boundary

The generated code intentionally does **not** configure ADC, PWM, GPIO, interrupt controllers, clocks, DMA, CMPSS or protection peripherals. The BSP supplies engineering-unit measurements to `*_ControlStep()` and applies the semantic output commands to hardware.

Base control-step rate: **{base_rate_hz:g} Hz**.

## Semantic outputs

{output_lines}

## Files

- `control_runtime.[ch]`: PI/PIF/2P2Z runtime, C99 float32.
- topology `*_control.[ch]`: generated algorithm and constants.
- `isr_template.c`: compile-ready integration skeleton; replace `USER_*` stubs in the BSP.
- `design_snapshot.json`: design provenance.
- `stability_report.txt`: code-generation gate result.

The generated sources contain no dynamic allocation and no vendor MCU headers.
''')


def generate_ttpl_control_code(
    analysis: PFCControlLabAnalysis,
    directory: str | Path,
    *,
    duty_feedforward_enabled: bool = True,
    require_stable: bool = True,
) -> CodegenResult:
    """Generate TTPL control-step C99 code from the analyzed GUI configuration."""
    cfg: PFCControlLabConfig = analysis.config
    cfg.validate()
    validation = validate_ttpl_codegen(analysis)
    if require_stable and not validation.passed:
        raise ValueError("TTPL code-generation stability gate failed; auto-tune/apply a stable loop first")
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    files = _common_files(out)
    stage, fw = cfg.power_stage, cfg.firmware
    amc_div = _divider(fw.current_loop_rate_hz, fw.amc_rate_hz, "TTPL AMC")
    voltage_div = _divider(fw.current_loop_rate_hz, fw.voltage_loop_rate_hz, "TTPL voltage loop")
    effective_duty_min = max(stage.duty_min, stage.minimum_effective_pulse_s * stage.switching_frequency_hz)

    header = f'''#ifndef TTPL_CONTROL_H
#define TTPL_CONTROL_H

#include <stdint.h>
#include "control_runtime.h"

#ifdef __cplusplus
extern "C" {{
#endif

#define TTPL_CONTROL_RATE_HZ {_cf(fw.current_loop_rate_hz)}
#define TTPL_AMC_DIVIDER {_cu(amc_div)}
#define TTPL_VOLTAGE_DIVIDER {_cu(voltage_div)}
#define TTPL_DUTY_FF_ENABLED {_cu(1 if duty_feedforward_enabled else 0)}

typedef struct
{{
    real32_t vac;
    real32_t iL;
    real32_t vbus;
    real32_t vbus_ref;
    uint16_t enable;
}} TTPL_Input_t;

typedef struct
{{
    real32_t duty;
    real32_t duty_ff;
    real32_t duty_pi;
    real32_t i_ref;
    real32_t gcmd;
    real32_t vloop;
    real32_t vac_rms;
    real32_t indu_comp;
    int16_t lf_polarity;
    uint16_t hf_enable;
}} TTPL_Output_t;

typedef struct
{{
    CTRL_State_t current_ctrl;
    CTRL_State_t voltage_ctrl;
    uint16_t amc_div;
    uint16_t voltage_div;
    real32_t vac_sq_lpf;
    real32_t vac_rms;
    real32_t vloop;
    real32_t gcmd;
    real32_t i_ref;
    real32_t duty_ff;
    real32_t indu_comp;
    real32_t vbus_ref_inv;
}} TTPL_State_t;

void TTPL_Init(TTPL_State_t *state);
void TTPL_Reset(TTPL_State_t *state);
void TTPL_ControlStep(const TTPL_Input_t *input, TTPL_Output_t *output, TTPL_State_t *state);

#ifdef __cplusplus
}}
#endif

#endif /* TTPL_CONTROL_H */
'''
    source = f'''#include "ttpl_control.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#define TTPL_DUTY_MIN {_cf(effective_duty_min)}
#define TTPL_DUTY_MAX {_cf(stage.duty_max)}
#define TTPL_VAC_RMS_ALPHA {_cf(fw.vac_rms_lpf_alpha)}
#define TTPL_VAC_RMS_FF_GAIN {_cf(fw.vac_rms_feedforward_gain)}
#define TTPL_GCMD_MAX {_cf(fw.gcmd_max_a_per_v)}
#define TTPL_INDU_COMP_GAIN {_cf(fw.indu_comp_gain)}
#define TTPL_INDU_COMP_MIN {_cf(fw.indu_comp_min)}
#define TTPL_INDU_COMP_MAX {_cf(fw.indu_comp_max)}
#define TTPL_VFF_BYPASS {_cu(1 if fw.vff_bypass else 0)}

void TTPL_Init(TTPL_State_t *state)
{{
    if(state == NULL) {{ return; }}
    (void)memset(state, 0, sizeof(*state));
{_controller_initializer('current_ctrl', cfg.current_controller)}
{_controller_initializer('voltage_ctrl', cfg.voltage_controller)}
    state->indu_comp = 1.0f;
    state->vbus_ref_inv = 1.0f;
}}

void TTPL_Reset(TTPL_State_t *state)
{{
    if(state == NULL) {{ return; }}
    CTRL_Reset(&state->current_ctrl);
    CTRL_Reset(&state->voltage_ctrl);
    state->amc_div = 0u;
    state->voltage_div = 0u;
    state->vac_sq_lpf = 0.0f;
    state->vac_rms = 0.0f;
    state->vloop = 0.0f;
    state->gcmd = 0.0f;
    state->i_ref = 0.0f;
    state->duty_ff = 0.0f;
    state->indu_comp = 1.0f;
    state->vbus_ref_inv = 1.0f;
}}

void TTPL_ControlStep(const TTPL_Input_t *input, TTPL_Output_t *output, TTPL_State_t *state)
{{
    real32_t vac_abs;
    real32_t current_error;
    real32_t duty_pi;
    real32_t duty;
    real32_t vin_rms_div;

    if(input == NULL || output == NULL || state == NULL) {{ return; }}
    if(input->enable == 0u)
    {{
        TTPL_Reset(state);
        (void)memset(output, 0, sizeof(*output));
        return;
    }}

    vac_abs = fabsf(input->vac);
    state->amc_div++;
    state->voltage_div++;

    if(state->voltage_div >= TTPL_VOLTAGE_DIVIDER)
    {{
        real32_t voltage_error;
        state->voltage_div = 0u;
        state->vac_sq_lpf += TTPL_VAC_RMS_ALPHA * (input->vac * input->vac - state->vac_sq_lpf);
        if(state->vac_sq_lpf < 0.0f) {{ state->vac_sq_lpf = 0.0f; }}
        state->vac_rms = sqrtf(state->vac_sq_lpf);
        voltage_error = input->vbus_ref - input->vbus;
        state->vloop = CTRL_Step(&state->voltage_ctrl, voltage_error);
        if(TTPL_VFF_BYPASS == 0u)
        {{
            vin_rms_div = TTPL_VAC_RMS_FF_GAIN * state->vac_rms;
            vin_rms_div *= vin_rms_div;
            if(vin_rms_div < 1.0f) {{ vin_rms_div = 1.0f; }}
            state->vloop /= vin_rms_div;
            state->vloop = CTRL_Clamp(state->vloop, {_cf(cfg.voltage_controller.output_min)}, {_cf(cfg.voltage_controller.output_max)});
        }}
        state->gcmd = state->vloop * TTPL_VAC_RMS_FF_GAIN;
        if(state->gcmd > TTPL_GCMD_MAX) {{ state->gcmd = TTPL_GCMD_MAX; }}
        state->vbus_ref_inv = (input->vbus_ref > 1.0f) ? (1.0f / input->vbus_ref) : 1.0f;
    }}

    if(state->amc_div >= TTPL_AMC_DIVIDER)
    {{
        state->amc_div = 0u;
        state->i_ref = state->gcmd * vac_abs;
        state->duty_ff = (TTPL_DUTY_FF_ENABLED != 0u) ? (1.0f - vac_abs * state->vbus_ref_inv) : 0.0f;
        state->indu_comp = CTRL_Clamp(TTPL_INDU_COMP_GAIN * state->i_ref, TTPL_INDU_COMP_MIN, TTPL_INDU_COMP_MAX);
    }}

    current_error = state->i_ref - fabsf(input->iL);
    duty_pi = CTRL_Step(&state->current_ctrl, current_error);
    duty = state->duty_ff + duty_pi * state->indu_comp;
    duty = CTRL_Clamp(duty, TTPL_DUTY_MIN, TTPL_DUTY_MAX);

    output->duty = duty;
    output->duty_ff = state->duty_ff;
    output->duty_pi = duty_pi;
    output->i_ref = state->i_ref;
    output->gcmd = state->gcmd;
    output->vloop = state->vloop;
    output->vac_rms = state->vac_rms;
    output->indu_comp = state->indu_comp;
    output->lf_polarity = (input->vac >= 0.0f) ? (int16_t)1 : (int16_t)-1;
    output->hf_enable = 1u;
}}
'''
    files["control_h"] = _write(out / "ttpl_control.h", header)
    files["control_c"] = _write(out / "ttpl_control.c", source)
    isr = r'''#include "ttpl_control.h"

static TTPL_State_t g_ttpl_control;

/* Replace these USER_* stubs with the target BSP. */
static TTPL_Input_t USER_ReadControlInputs(void)
{
    TTPL_Input_t input = {0};
    return input;
}

static void USER_ApplyControlOutputs(const TTPL_Output_t *output)
{
    (void)output;
}

void TTPL_ISR_Init(void)
{
    TTPL_Init(&g_ttpl_control);
}

void TTPL_ISR_Template(void)
{
    TTPL_Input_t input = USER_ReadControlInputs();
    TTPL_Output_t output = {0};
    TTPL_ControlStep(&input, &output, &g_ttpl_control);
    USER_ApplyControlOutputs(&output);
}
'''
    files["isr_template"] = _write(out / "isr_template.c", isr)
    snapshot = {
        "topology": "single_phase_ttpl_pfc",
        "numeric_format": "C99 float32",
        "bsp_generated": False,
        "duty_feedforward_enabled": duty_feedforward_enabled,
        "config": _jsonable(cfg),
        "analysis": analysis.summary_dict(),
        "controller_kinds": {
            "current": controller_kind(cfg.current_controller).value,
            "voltage": controller_kind(cfg.voltage_controller).value,
        },
    }
    files["snapshot"] = _write(out / "design_snapshot.json", json.dumps(snapshot, indent=2, ensure_ascii=False))
    files["validation"] = _emit_validation(out, validation)
    files["readme"] = _emit_readme(out, "Single-Phase TTPL PFC", fw.current_loop_rate_hz,
                                    ("duty", "hf_enable", "lf_polarity", "i_ref", "duty_ff", "duty_pi", "gcmd", "vloop"))
    return CodegenResult(out, files, validation)


def generate_vienna_control_code(
    analysis: ViennaControlLabAnalysis,
    directory: str | Path,
    *,
    require_stable: bool = True,
) -> CodegenResult:
    cfg: ViennaControlLabConfig = analysis.config
    cfg.validate()
    validation = validate_vienna_codegen(analysis)
    if require_stable and not validation.passed:
        raise ValueError("Vienna code-generation stability gate failed")
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    files = _common_files(out)
    stage, fw = cfg.power_stage, cfg.firmware
    ref_div = _divider(fw.current_loop_rate_hz, fw.reference_rate_hz, "Vienna reference")
    voltage_div = _divider(fw.current_loop_rate_hz, fw.voltage_loop_rate_hz, "Vienna voltage loop")
    balance_div = _divider(fw.current_loop_rate_hz, fw.balance_loop_rate_hz, "Vienna balance loop")
    min_frac = stage.minimum_effective_pulse_s * stage.switching_frequency_hz

    header = f'''#ifndef VIENNA_CONTROL_H
#define VIENNA_CONTROL_H

#include <stdint.h>
#include "control_runtime.h"

#ifdef __cplusplus
extern "C" {{
#endif

#define VIENNA_CONTROL_RATE_HZ {_cf(fw.current_loop_rate_hz)}
#define VIENNA_REFERENCE_DIVIDER {_cu(ref_div)}
#define VIENNA_VOLTAGE_DIVIDER {_cu(voltage_div)}
#define VIENNA_BALANCE_DIVIDER {_cu(balance_div)}

typedef struct
{{
    real32_t va, vb, vc;
    real32_t ia, ib, ic;
    real32_t vdc_pos, vdc_neg;
    real32_t vdc_ref;
    uint16_t enable;
}} Vienna_Input_t;

typedef struct
{{
    real32_t duty_a, duty_b, duty_c;
    real32_t modulation_a, modulation_b, modulation_c;
    real32_t ia_ref, ib_ref, ic_ref;
    real32_t pi_a, pi_b, pi_c;
    real32_t gcmd;
    real32_t balance_cmd;
    real32_t third_harmonic;
    uint16_t enable;
}} Vienna_Output_t;

typedef struct
{{
    CTRL_State_t current_a;
    CTRL_State_t current_b;
    CTRL_State_t current_c;
    CTRL_State_t voltage_ctrl;
    CTRL_State_t balance_ctrl;
    uint16_t reference_div;
    uint16_t voltage_div;
    uint16_t balance_div;
    real32_t gcmd;
    real32_t balance;
    real32_t ia_ref, ib_ref, ic_ref;
    real32_t ia_ref_prev, ib_ref_prev, ic_ref_prev;
    real32_t ff_a, ff_b, ff_c;
    real32_t pi_a, pi_b, pi_c;
}} Vienna_State_t;

void Vienna_Init(Vienna_State_t *state);
void Vienna_Reset(Vienna_State_t *state);
void Vienna_ControlStep(const Vienna_Input_t *input, Vienna_Output_t *output, Vienna_State_t *state);

#ifdef __cplusplus
}}
#endif

#endif /* VIENNA_CONTROL_H */
'''

    source = f'''#include "vienna_control.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#define VIENNA_MOD_LIMIT {_cf(stage.modulation_limit)}
#define VIENNA_MIN_PULSE_FRACTION {_cf(min_frac)}
#define VIENNA_L_H {_cf(stage.boost_inductance_h)}
#define VIENNA_R_OHM {_cf(stage.phase_series_resistance_ohm)}
#define VIENNA_REFERENCE_TS {_cf(1.0/fw.reference_rate_hz)}
#define VIENNA_BALANCE_LIMIT {_cf(fw.balance_injection_limit)}
#define VIENNA_THIRD_HARMONIC_ENABLED {_cu(1 if fw.third_harmonic_injection_enabled else 0)}
#define VIENNA_INDUCTOR_FF_ENABLED {_cu(1 if fw.inductor_voltage_drop_feedforward_enabled else 0)}

static real32_t max3(real32_t a, real32_t b, real32_t c)
{{
    real32_t x = (a > b) ? a : b;
    return (x > c) ? x : c;
}}

static real32_t min3(real32_t a, real32_t b, real32_t c)
{{
    real32_t x = (a < b) ? a : b;
    return (x < c) ? x : c;
}}

static void Vienna_ModulatePhase(real32_t raw, real32_t *modulation, real32_t *zero_duty)
{{
    real32_t m = CTRL_Clamp(raw, -VIENNA_MOD_LIMIT, VIENNA_MOD_LIMIT);
    real32_t active = fabsf(m);
    real32_t sign = (m >= 0.0f) ? 1.0f : -1.0f;
    if(VIENNA_MIN_PULSE_FRACTION > 0.0f)
    {{
        if(active > 0.0f && active < VIENNA_MIN_PULSE_FRACTION)
        {{
            active = 0.0f;
        }}
        else if((1.0f - active) > 0.0f && (1.0f - active) < VIENNA_MIN_PULSE_FRACTION)
        {{
            active = 1.0f;
        }}
    }}
    *modulation = (active == 0.0f) ? 0.0f : sign * active;
    *zero_duty = 1.0f - active;
}}

void Vienna_Init(Vienna_State_t *state)
{{
    if(state == NULL) {{ return; }}
    (void)memset(state, 0, sizeof(*state));
{_controller_initializer('current_a', cfg.current_controller)}
{_controller_initializer('current_b', cfg.current_controller)}
{_controller_initializer('current_c', cfg.current_controller)}
{_controller_initializer('voltage_ctrl', cfg.voltage_controller)}
{_controller_initializer('balance_ctrl', cfg.balance_controller)}
    state->gcmd = {_cf(stage.input_conductance_a_per_v)};
}}

void Vienna_Reset(Vienna_State_t *state)
{{
    if(state == NULL) {{ return; }}
    CTRL_Reset(&state->current_a); CTRL_Reset(&state->current_b); CTRL_Reset(&state->current_c);
    CTRL_Reset(&state->voltage_ctrl); CTRL_Reset(&state->balance_ctrl);
    state->reference_div = 0u; state->voltage_div = 0u; state->balance_div = 0u;
    state->gcmd = {_cf(stage.input_conductance_a_per_v)};
    state->balance = 0.0f;
    state->ia_ref = state->ib_ref = state->ic_ref = 0.0f;
    state->ia_ref_prev = state->ib_ref_prev = state->ic_ref_prev = 0.0f;
    state->ff_a = state->ff_b = state->ff_c = 0.0f;
    state->pi_a = state->pi_b = state->pi_c = 0.0f;
}}

void Vienna_ControlStep(const Vienna_Input_t *input, Vienna_Output_t *output, Vienna_State_t *state)
{{
    real32_t vdc;
    real32_t half_bus;
    real32_t third;
    real32_t mff_a, mff_b, mff_c;
    real32_t mod_a, mod_b, mod_c;

    if(input == NULL || output == NULL || state == NULL) {{ return; }}
    if(input->enable == 0u)
    {{
        Vienna_Reset(state);
        (void)memset(output, 0, sizeof(*output));
        return;
    }}

    state->reference_div++;
    state->voltage_div++;
    state->balance_div++;

    if(state->voltage_div >= VIENNA_VOLTAGE_DIVIDER)
    {{
        state->voltage_div = 0u;
        state->gcmd = CTRL_Step(&state->voltage_ctrl, input->vdc_ref - (input->vdc_pos + input->vdc_neg));
        if(state->gcmd < 0.0f) {{ state->gcmd = 0.0f; }}
    }}

    if(state->balance_div >= VIENNA_BALANCE_DIVIDER)
    {{
        state->balance_div = 0u;
        state->balance = CTRL_Step(&state->balance_ctrl, input->vdc_pos - input->vdc_neg);
        state->balance = CTRL_Clamp(state->balance, -VIENNA_BALANCE_LIMIT, VIENNA_BALANCE_LIMIT);
    }}

    if(state->reference_div >= VIENNA_REFERENCE_DIVIDER)
    {{
        real32_t dia, dib, dic;
        state->reference_div = 0u;
        state->ia_ref_prev = state->ia_ref;
        state->ib_ref_prev = state->ib_ref;
        state->ic_ref_prev = state->ic_ref;
        state->ia_ref = state->gcmd * input->va;
        state->ib_ref = state->gcmd * input->vb;
        state->ic_ref = state->gcmd * input->vc;
        if(VIENNA_INDUCTOR_FF_ENABLED != 0u)
        {{
            dia = (state->ia_ref - state->ia_ref_prev) / VIENNA_REFERENCE_TS;
            dib = (state->ib_ref - state->ib_ref_prev) / VIENNA_REFERENCE_TS;
            dic = (state->ic_ref - state->ic_ref_prev) / VIENNA_REFERENCE_TS;
            state->ff_a = VIENNA_R_OHM * state->ia_ref + VIENNA_L_H * dia;
            state->ff_b = VIENNA_R_OHM * state->ib_ref + VIENNA_L_H * dib;
            state->ff_c = VIENNA_R_OHM * state->ic_ref + VIENNA_L_H * dic;
        }}
        else
        {{
            state->ff_a = state->ff_b = state->ff_c = 0.0f;
        }}
    }}

    state->pi_a = CTRL_Step(&state->current_a, state->ia_ref - input->ia);
    state->pi_b = CTRL_Step(&state->current_b, state->ib_ref - input->ib);
    state->pi_c = CTRL_Step(&state->current_c, state->ic_ref - input->ic);

    vdc = input->vdc_pos + input->vdc_neg;
    half_bus = 0.5f * vdc;
    if(half_bus < 25.0f) {{ half_bus = 25.0f; }}
    third = (VIENNA_THIRD_HARMONIC_ENABLED != 0u) ? 0.5f * (max3(input->va, input->vb, input->vc) + min3(input->va, input->vb, input->vc)) : 0.0f;

    mff_a = (input->va - third - state->ff_a) / half_bus;
    mff_b = (input->vb - third - state->ff_b) / half_bus;
    mff_c = (input->vc - third - state->ff_c) / half_bus;
    Vienna_ModulatePhase(mff_a - state->pi_a - state->balance, &mod_a, &output->duty_a);
    Vienna_ModulatePhase(mff_b - state->pi_b - state->balance, &mod_b, &output->duty_b);
    Vienna_ModulatePhase(mff_c - state->pi_c - state->balance, &mod_c, &output->duty_c);

    output->modulation_a = mod_a; output->modulation_b = mod_b; output->modulation_c = mod_c;
    output->ia_ref = state->ia_ref; output->ib_ref = state->ib_ref; output->ic_ref = state->ic_ref;
    output->pi_a = state->pi_a; output->pi_b = state->pi_b; output->pi_c = state->pi_c;
    output->gcmd = state->gcmd;
    output->balance_cmd = state->balance;
    output->third_harmonic = third;
    output->enable = 1u;
}}
'''
    files["control_h"] = _write(out / "vienna_control.h", header)
    files["control_c"] = _write(out / "vienna_control.c", source)
    isr = r'''#include "vienna_control.h"

static Vienna_State_t g_vienna_control;

static Vienna_Input_t USER_ReadControlInputs(void)
{
    Vienna_Input_t input = {0};
    return input;
}

static void USER_ApplyControlOutputs(const Vienna_Output_t *output)
{
    (void)output;
}

void Vienna_ISR_Init(void)
{
    Vienna_Init(&g_vienna_control);
}

void Vienna_ISR_Template(void)
{
    Vienna_Input_t input = USER_ReadControlInputs();
    Vienna_Output_t output = {0};
    Vienna_ControlStep(&input, &output, &g_vienna_control);
    USER_ApplyControlOutputs(&output);
}
'''
    files["isr_template"] = _write(out / "isr_template.c", isr)
    snapshot = {
        "topology": "three_phase_vienna_pfc",
        "numeric_format": "C99 float32",
        "bsp_generated": False,
        "config": _jsonable(cfg),
        "controller_kinds": {
            "current_abc": controller_kind(cfg.current_controller).value,
            "voltage": controller_kind(cfg.voltage_controller).value,
            "midpoint_balance": controller_kind(cfg.balance_controller).value,
        },
        "margins": {
            "current_pm_deg": analysis.current_loop.margins.phase_margin_deg,
            "voltage_pm_deg": analysis.voltage_loop.margins.phase_margin_deg,
            "balance_pm_deg": analysis.balance_loop.margins.phase_margin_deg,
        },
    }
    files["snapshot"] = _write(out / "design_snapshot.json", json.dumps(snapshot, indent=2, ensure_ascii=False))
    files["validation"] = _emit_validation(out, validation)
    files["readme"] = _emit_readme(out, "Three-Phase Vienna PFC", fw.current_loop_rate_hz,
                                    ("duty_a/duty_b/duty_c (center-switch zero-state duty)", "modulation_a/b/c", "ia_ref/ib_ref/ic_ref", "gcmd", "balance_cmd"))
    return CodegenResult(out, files, validation)


def generate_llc_control_code(
    analysis: DigitalLoopAnalysis,
    directory: str | Path,
    *,
    require_stable: bool = True,
) -> CodegenResult:
    validation = validate_llc_codegen(analysis)
    if require_stable and not validation.passed:
        raise ValueError("LLC code-generation stability gate failed")
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    files = _common_files(out)
    cfg = analysis.controller_config
    lut = analysis.fm_lut
    pcmd = ", ".join(_cf(v) for v in lut.pcmd)
    values = ", ".join(_cf(v) for v in lut.values)
    count_divisor = lut.count_mode.frequency_divisor
    sample_rate = 1.0 / cfg.sample_time_s

    header = f'''#ifndef LLC_CONTROL_H
#define LLC_CONTROL_H

#include <stdint.h>
#include "control_runtime.h"

#ifdef __cplusplus
extern "C" {{
#endif

#define LLC_CONTROL_RATE_HZ {_cf(sample_rate)}
#define LLC_FM_LUT_POINTS {_cu(len(lut.pcmd))}

typedef struct
{{
    real32_t vout;
    real32_t vref;
    uint16_t enable;
}} LLC_Input_t;

typedef struct
{{
    real32_t pcmd;
    real32_t fsw_hz;
    real32_t tbprd_equivalent;
    real32_t controller_output;
    uint16_t enable;
}} LLC_Output_t;

typedef struct
{{
    CTRL_State_t voltage_ctrl;
}} LLC_State_t;

void LLC_Init(LLC_State_t *state);
void LLC_Reset(LLC_State_t *state);
void LLC_ControlStep(const LLC_Input_t *input, LLC_Output_t *output, LLC_State_t *state);

#ifdef __cplusplus
}}
#endif

#endif /* LLC_CONTROL_H */
'''
    mode_tbprd = lut.mode == FMLUTMode.PCMD_TO_TBPRD
    source = f'''#include "llc_control.h"

#include <stddef.h>
#include <string.h>

#define LLC_TIMER_CLOCK_HZ {_cf(lut.timer_clock_hz)}
#define LLC_COUNT_DIVISOR {_cf(count_divisor)}
#define LLC_LUT_IS_TBPRD {_cu(1 if mode_tbprd else 0)}

static const real32_t k_pcmd[LLC_FM_LUT_POINTS] = {{{pcmd}}};
static const real32_t k_value[LLC_FM_LUT_POINTS] = {{{values}}};

static real32_t LLC_LUT(real32_t command)
{{
    uint32_t i;
    real32_t p;
    real32_t fraction;
    p = CTRL_Clamp(command, k_pcmd[0], k_pcmd[LLC_FM_LUT_POINTS - 1u]);
    for(i = 0u; i < LLC_FM_LUT_POINTS - 1u; ++i)
    {{
        if(p <= k_pcmd[i + 1u])
        {{
            fraction = (p - k_pcmd[i]) / (k_pcmd[i + 1u] - k_pcmd[i]);
            return k_value[i] + fraction * (k_value[i + 1u] - k_value[i]);
        }}
    }}
    return k_value[LLC_FM_LUT_POINTS - 1u];
}}

void LLC_Init(LLC_State_t *state)
{{
    if(state == NULL) {{ return; }}
    (void)memset(state, 0, sizeof(*state));
{_controller_initializer('voltage_ctrl', cfg)}
}}

void LLC_Reset(LLC_State_t *state)
{{
    if(state == NULL) {{ return; }}
    CTRL_Reset(&state->voltage_ctrl);
}}

void LLC_ControlStep(const LLC_Input_t *input, LLC_Output_t *output, LLC_State_t *state)
{{
    real32_t command;
    real32_t value;
    if(input == NULL || output == NULL || state == NULL) {{ return; }}
    if(input->enable == 0u)
    {{
        LLC_Reset(state);
        (void)memset(output, 0, sizeof(*output));
        return;
    }}
    command = CTRL_Step(&state->voltage_ctrl, input->vref - input->vout);
    command = CTRL_Clamp(command, 0.0f, 1.0f);
    value = LLC_LUT(command);
    output->pcmd = command;
    output->controller_output = command;
    if(LLC_LUT_IS_TBPRD != 0u)
    {{
        output->tbprd_equivalent = value;
        output->fsw_hz = LLC_TIMER_CLOCK_HZ / (LLC_COUNT_DIVISOR * value);
    }}
    else
    {{
        output->fsw_hz = value;
        output->tbprd_equivalent = LLC_TIMER_CLOCK_HZ / (LLC_COUNT_DIVISOR * value);
    }}
    output->enable = 1u;
}}
'''
    files["control_h"] = _write(out / "llc_control.h", header)
    files["control_c"] = _write(out / "llc_control.c", source)
    isr = r'''#include "llc_control.h"

static LLC_State_t g_llc_control;

static LLC_Input_t USER_ReadControlInputs(void)
{
    LLC_Input_t input = {0};
    return input;
}

static void USER_ApplyControlOutputs(const LLC_Output_t *output)
{
    (void)output;
}

void LLC_ISR_Init(void)
{
    LLC_Init(&g_llc_control);
}

void LLC_ISR_Template(void)
{
    LLC_Input_t input = USER_ReadControlInputs();
    LLC_Output_t output = {0};
    LLC_ControlStep(&input, &output, &g_llc_control);
    USER_ApplyControlOutputs(&output);
}
'''
    files["isr_template"] = _write(out / "isr_template.c", isr)
    snapshot = {
        "topology": "llc_frequency_control",
        "numeric_format": "C99 float32",
        "bsp_generated": False,
        "controller_kind": controller_kind(cfg).value,
        "controller_config": _jsonable(cfg),
        "fm_lut": {
            "mode": lut.mode.value,
            "pcmd": lut.pcmd.tolist(),
            "values": lut.values.tolist(),
            "timer_clock_hz": lut.timer_clock_hz,
            "count_mode": lut.count_mode.value,
        },
        "operating_point": _jsonable(analysis.fm_operating_point),
        "nominal_phase_margin_deg": analysis.margins_nominal_delay.phase_margin_deg,
        "nominal_gain_margin_db": analysis.margins_nominal_delay.gain_margin_db,
        "discrete_stable": analysis.discrete_approximation.stable,
    }
    files["snapshot"] = _write(out / "design_snapshot.json", json.dumps(snapshot, indent=2, ensure_ascii=False))
    files["validation"] = _emit_validation(out, validation)
    files["readme"] = _emit_readme(out, "LLC frequency-control loop", sample_rate,
                                    ("pcmd", "fsw_hz", "tbprd_equivalent", "controller_output"))
    return CodegenResult(out, files, validation)
