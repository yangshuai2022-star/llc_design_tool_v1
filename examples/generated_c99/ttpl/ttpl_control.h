#ifndef TTPL_CONTROL_H
#define TTPL_CONTROL_H

#include <stdint.h>
#include "control_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TTPL_CONTROL_RATE_HZ 50000.0f
#define TTPL_AMC_DIVIDER 2u
#define TTPL_VOLTAGE_DIVIDER 5u
#define TTPL_DUTY_FF_ENABLED 1u

typedef struct
{
    real32_t vac;
    real32_t iL;
    real32_t vbus;
    real32_t vbus_ref;
    uint16_t enable;
} TTPL_Input_t;

typedef struct
{
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
} TTPL_Output_t;

typedef struct
{
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
} TTPL_State_t;

void TTPL_Init(TTPL_State_t *state);
void TTPL_Reset(TTPL_State_t *state);
void TTPL_ControlStep(const TTPL_Input_t *input, TTPL_Output_t *output, TTPL_State_t *state);

#ifdef __cplusplus
}
#endif

#endif /* TTPL_CONTROL_H */
