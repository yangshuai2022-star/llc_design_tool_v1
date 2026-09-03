#include "vienna_control.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#define VIENNA_MOD_LIMIT 0.96f
#define VIENNA_MIN_PULSE_FRACTION 0.0f
#define VIENNA_L_H 0.0006f
#define VIENNA_R_OHM 0.08f
#define VIENNA_REFERENCE_TS 4e-05f
#define VIENNA_BALANCE_LIMIT 0.08f
#define VIENNA_THIRD_HARMONIC_ENABLED 1u
#define VIENNA_INDUCTOR_FF_ENABLED 1u

static real32_t max3(real32_t a, real32_t b, real32_t c)
{
    real32_t x = (a > b) ? a : b;
    return (x > c) ? x : c;
}

static real32_t min3(real32_t a, real32_t b, real32_t c)
{
    real32_t x = (a < b) ? a : b;
    return (x < c) ? x : c;
}

static void Vienna_ModulatePhase(real32_t raw, real32_t *modulation, real32_t *zero_duty)
{
    real32_t m = CTRL_Clamp(raw, -VIENNA_MOD_LIMIT, VIENNA_MOD_LIMIT);
    real32_t active = fabsf(m);
    real32_t sign = (m >= 0.0f) ? 1.0f : -1.0f;
    if(VIENNA_MIN_PULSE_FRACTION > 0.0f)
    {
        if(active > 0.0f && active < VIENNA_MIN_PULSE_FRACTION)
        {
            active = 0.0f;
        }
        else if((1.0f - active) > 0.0f && (1.0f - active) < VIENNA_MIN_PULSE_FRACTION)
        {
            active = 1.0f;
        }
    }
    *modulation = (active == 0.0f) ? 0.0f : sign * active;
    *zero_duty = 1.0f - active;
}

void Vienna_Init(Vienna_State_t *state)
{
    if(state == NULL) { return; }
    (void)memset(state, 0, sizeof(*state));
    CTRL_InitPI(&state->current_a, 0.02f, 0.04f, -0.45f, 0.45f);
    CTRL_InitPI(&state->current_b, 0.02f, 0.04f, -0.45f, 0.45f);
    CTRL_InitPI(&state->current_c, 0.02f, 0.04f, -0.45f, 0.45f);
    CTRL_InitPI(&state->voltage_ctrl, 2e-05f, 0.000625f, 0.0f, 0.3f);
    CTRL_InitPI(&state->balance_ctrl, 0.001f, 0.0025f, -0.08f, 0.08f);
    state->gcmd = 0.0637755102f;
}

void Vienna_Reset(Vienna_State_t *state)
{
    if(state == NULL) { return; }
    CTRL_Reset(&state->current_a); CTRL_Reset(&state->current_b); CTRL_Reset(&state->current_c);
    CTRL_Reset(&state->voltage_ctrl); CTRL_Reset(&state->balance_ctrl);
    state->reference_div = 0u; state->voltage_div = 0u; state->balance_div = 0u;
    state->gcmd = 0.0637755102f;
    state->balance = 0.0f;
    state->ia_ref = state->ib_ref = state->ic_ref = 0.0f;
    state->ia_ref_prev = state->ib_ref_prev = state->ic_ref_prev = 0.0f;
    state->ff_a = state->ff_b = state->ff_c = 0.0f;
    state->pi_a = state->pi_b = state->pi_c = 0.0f;
}

void Vienna_ControlStep(const Vienna_Input_t *input, Vienna_Output_t *output, Vienna_State_t *state)
{
    real32_t vdc;
    real32_t half_bus;
    real32_t third;
    real32_t mff_a, mff_b, mff_c;
    real32_t mod_a, mod_b, mod_c;

    if(input == NULL || output == NULL || state == NULL) { return; }
    if(input->enable == 0u)
    {
        Vienna_Reset(state);
        (void)memset(output, 0, sizeof(*output));
        return;
    }

    state->reference_div++;
    state->voltage_div++;
    state->balance_div++;

    if(state->voltage_div >= VIENNA_VOLTAGE_DIVIDER)
    {
        state->voltage_div = 0u;
        state->gcmd = CTRL_Step(&state->voltage_ctrl, input->vdc_ref - (input->vdc_pos + input->vdc_neg));
        if(state->gcmd < 0.0f) { state->gcmd = 0.0f; }
    }

    if(state->balance_div >= VIENNA_BALANCE_DIVIDER)
    {
        state->balance_div = 0u;
        state->balance = CTRL_Step(&state->balance_ctrl, input->vdc_pos - input->vdc_neg);
        state->balance = CTRL_Clamp(state->balance, -VIENNA_BALANCE_LIMIT, VIENNA_BALANCE_LIMIT);
    }

    if(state->reference_div >= VIENNA_REFERENCE_DIVIDER)
    {
        real32_t dia, dib, dic;
        state->reference_div = 0u;
        state->ia_ref_prev = state->ia_ref;
        state->ib_ref_prev = state->ib_ref;
        state->ic_ref_prev = state->ic_ref;
        state->ia_ref = state->gcmd * input->va;
        state->ib_ref = state->gcmd * input->vb;
        state->ic_ref = state->gcmd * input->vc;
        if(VIENNA_INDUCTOR_FF_ENABLED != 0u)
        {
            dia = (state->ia_ref - state->ia_ref_prev) / VIENNA_REFERENCE_TS;
            dib = (state->ib_ref - state->ib_ref_prev) / VIENNA_REFERENCE_TS;
            dic = (state->ic_ref - state->ic_ref_prev) / VIENNA_REFERENCE_TS;
            state->ff_a = VIENNA_R_OHM * state->ia_ref + VIENNA_L_H * dia;
            state->ff_b = VIENNA_R_OHM * state->ib_ref + VIENNA_L_H * dib;
            state->ff_c = VIENNA_R_OHM * state->ic_ref + VIENNA_L_H * dic;
        }
        else
        {
            state->ff_a = state->ff_b = state->ff_c = 0.0f;
        }
    }

    state->pi_a = CTRL_Step(&state->current_a, state->ia_ref - input->ia);
    state->pi_b = CTRL_Step(&state->current_b, state->ib_ref - input->ib);
    state->pi_c = CTRL_Step(&state->current_c, state->ic_ref - input->ic);

    vdc = input->vdc_pos + input->vdc_neg;
    half_bus = 0.5f * vdc;
    if(half_bus < 25.0f) { half_bus = 25.0f; }
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
}
