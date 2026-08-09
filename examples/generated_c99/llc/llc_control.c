#include "llc_control.h"

#include <stddef.h>
#include <string.h>

#define LLC_TIMER_CLOCK_HZ 120000000.0f
#define LLC_COUNT_DIVISOR 2.0f
#define LLC_LUT_IS_TBPRD 1u

static const real32_t k_pcmd[LLC_FM_LUT_POINTS] = {0.0f, 0.04f, 0.08f, 0.12f, 0.16f, 0.2f, 0.25f, 0.3f, 0.355f, 0.41f, 0.465f, 0.52f, 0.5825f, 0.645f, 0.7075f, 0.77f, 0.8275f, 0.885f, 0.9425f, 1.0f};
static const real32_t k_value[LLC_FM_LUT_POINTS] = {240.0f, 258.0f, 279.0f, 303.0f, 332.0f, 367.0f, 424.0f, 500.0f, 533.0f, 571.0f, 615.0f, 667.0f, 706.0f, 750.0f, 800.0f, 811.0f, 822.0f, 833.0f, 845.0f, 857.0f};

static real32_t LLC_LUT(real32_t command)
{
    uint32_t i;
    real32_t p;
    real32_t fraction;
    p = CTRL_Clamp(command, k_pcmd[0], k_pcmd[LLC_FM_LUT_POINTS - 1u]);
    for(i = 0u; i < LLC_FM_LUT_POINTS - 1u; ++i)
    {
        if(p <= k_pcmd[i + 1u])
        {
            fraction = (p - k_pcmd[i]) / (k_pcmd[i + 1u] - k_pcmd[i]);
            return k_value[i] + fraction * (k_value[i + 1u] - k_value[i]);
        }
    }
    return k_value[LLC_FM_LUT_POINTS - 1u];
}

void LLC_Init(LLC_State_t *state)
{
    if(state == NULL) { return; }
    (void)memset(state, 0, sizeof(*state));
    CTRL_InitPI(&state->voltage_ctrl, 0.002f, 0.00333333333f, 0.0f, 1.0f);
}

void LLC_Reset(LLC_State_t *state)
{
    if(state == NULL) { return; }
    CTRL_Reset(&state->voltage_ctrl);
}

void LLC_ControlStep(const LLC_Input_t *input, LLC_Output_t *output, LLC_State_t *state)
{
    real32_t command;
    real32_t value;
    if(input == NULL || output == NULL || state == NULL) { return; }
    if(input->enable == 0u)
    {
        LLC_Reset(state);
        (void)memset(output, 0, sizeof(*output));
        return;
    }
    command = CTRL_Step(&state->voltage_ctrl, input->vref - input->vout);
    command = CTRL_Clamp(command, 0.0f, 1.0f);
    value = LLC_LUT(command);
    output->pcmd = command;
    output->controller_output = command;
    if(LLC_LUT_IS_TBPRD != 0u)
    {
        output->tbprd_equivalent = value;
        output->fsw_hz = LLC_TIMER_CLOCK_HZ / (LLC_COUNT_DIVISOR * value);
    }
    else
    {
        output->fsw_hz = value;
        output->tbprd_equivalent = LLC_TIMER_CLOCK_HZ / (LLC_COUNT_DIVISOR * value);
    }
    output->enable = 1u;
}
