#include "ttpl_control.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#define TTPL_DUTY_MIN 0.01f
#define TTPL_DUTY_MAX 0.98f
#define TTPL_VAC_RMS_ALPHA 0.006283f
#define TTPL_VAC_RMS_FF_GAIN 0.01f
#define TTPL_GCMD_MAX 0.18f
#define TTPL_INDU_COMP_GAIN 0.085f
#define TTPL_INDU_COMP_MIN 0.7f
#define TTPL_INDU_COMP_MAX 1.0f
#define TTPL_VFF_BYPASS 0u

void TTPL_Init(TTPL_State_t *state)
{
    if(state == NULL) { return; }
    (void)memset(state, 0, sizeof(*state));
    CTRL_InitPI(&state->current_ctrl, 0.00854059477f, 0.0154779537f, -2.0f, 0.98f);
    CTRL_InitPI(&state->voltage_ctrl, 0.1f, 0.025f, -1.0f, 40.0f);
    state->indu_comp = 1.0f;
    state->vbus_ref_inv = 1.0f;
}

void TTPL_Reset(TTPL_State_t *state)
{
    if(state == NULL) { return; }
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
}

void TTPL_ControlStep(const TTPL_Input_t *input, TTPL_Output_t *output, TTPL_State_t *state)
{
    real32_t vac_abs;
    real32_t current_error;
    real32_t duty_pi;
    real32_t duty;
    real32_t vin_rms_div;

    if(input == NULL || output == NULL || state == NULL) { return; }
    if(input->enable == 0u)
    {
        TTPL_Reset(state);
        (void)memset(output, 0, sizeof(*output));
        return;
    }

    vac_abs = fabsf(input->vac);
    state->amc_div++;
    state->voltage_div++;

    if(state->voltage_div >= TTPL_VOLTAGE_DIVIDER)
    {
        real32_t voltage_error;
        state->voltage_div = 0u;
        state->vac_sq_lpf += TTPL_VAC_RMS_ALPHA * (input->vac * input->vac - state->vac_sq_lpf);
        if(state->vac_sq_lpf < 0.0f) { state->vac_sq_lpf = 0.0f; }
        state->vac_rms = sqrtf(state->vac_sq_lpf);
        voltage_error = input->vbus_ref - input->vbus;
        state->vloop = CTRL_Step(&state->voltage_ctrl, voltage_error);
        if(TTPL_VFF_BYPASS == 0u)
        {
            vin_rms_div = TTPL_VAC_RMS_FF_GAIN * state->vac_rms;
            vin_rms_div *= vin_rms_div;
            if(vin_rms_div < 1.0f) { vin_rms_div = 1.0f; }
            state->vloop /= vin_rms_div;
            state->vloop = CTRL_Clamp(state->vloop, -1.0f, 40.0f);
        }
        state->gcmd = state->vloop * TTPL_VAC_RMS_FF_GAIN;
        if(state->gcmd > TTPL_GCMD_MAX) { state->gcmd = TTPL_GCMD_MAX; }
        state->vbus_ref_inv = (input->vbus_ref > 1.0f) ? (1.0f / input->vbus_ref) : 1.0f;
    }

    if(state->amc_div >= TTPL_AMC_DIVIDER)
    {
        state->amc_div = 0u;
        state->i_ref = state->gcmd * vac_abs;
        state->duty_ff = (TTPL_DUTY_FF_ENABLED != 0u) ? (1.0f - vac_abs * state->vbus_ref_inv) : 0.0f;
        state->indu_comp = CTRL_Clamp(TTPL_INDU_COMP_GAIN * state->i_ref, TTPL_INDU_COMP_MIN, TTPL_INDU_COMP_MAX);
    }

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
}
