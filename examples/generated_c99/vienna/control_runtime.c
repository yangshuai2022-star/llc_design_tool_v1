#include "control_runtime.h"

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
