#include "llc_control.h"

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
