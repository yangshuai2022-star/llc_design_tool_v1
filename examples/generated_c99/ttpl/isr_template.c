#include "ttpl_control.h"

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
