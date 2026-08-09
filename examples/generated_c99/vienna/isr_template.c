#include "vienna_control.h"

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
