#ifndef LLC_CONTROL_H
#define LLC_CONTROL_H

#include <stdint.h>
#include "control_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LLC_CONTROL_RATE_HZ 50000.0f
#define LLC_FM_LUT_POINTS 20u

typedef struct
{
    real32_t vout;
    real32_t vref;
    uint16_t enable;
} LLC_Input_t;

typedef struct
{
    real32_t pcmd;
    real32_t fsw_hz;
    real32_t tbprd_equivalent;
    real32_t controller_output;
    uint16_t enable;
} LLC_Output_t;

typedef struct
{
    CTRL_State_t voltage_ctrl;
} LLC_State_t;

void LLC_Init(LLC_State_t *state);
void LLC_Reset(LLC_State_t *state);
void LLC_ControlStep(const LLC_Input_t *input, LLC_Output_t *output, LLC_State_t *state);

#ifdef __cplusplus
}
#endif

#endif /* LLC_CONTROL_H */
