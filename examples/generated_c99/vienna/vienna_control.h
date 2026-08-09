#ifndef VIENNA_CONTROL_H
#define VIENNA_CONTROL_H

#include <stdint.h>
#include "control_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

#define VIENNA_CONTROL_RATE_HZ 50000.0f
#define VIENNA_REFERENCE_DIVIDER 2u
#define VIENNA_VOLTAGE_DIVIDER 5u
#define VIENNA_BALANCE_DIVIDER 5u

typedef struct
{
    real32_t va, vb, vc;
    real32_t ia, ib, ic;
    real32_t vdc_pos, vdc_neg;
    real32_t vdc_ref;
    uint16_t enable;
} Vienna_Input_t;

typedef struct
{
    real32_t duty_a, duty_b, duty_c;
    real32_t modulation_a, modulation_b, modulation_c;
    real32_t ia_ref, ib_ref, ic_ref;
    real32_t pi_a, pi_b, pi_c;
    real32_t gcmd;
    real32_t balance_cmd;
    real32_t third_harmonic;
    uint16_t enable;
} Vienna_Output_t;

typedef struct
{
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
} Vienna_State_t;

void Vienna_Init(Vienna_State_t *state);
void Vienna_Reset(Vienna_State_t *state);
void Vienna_ControlStep(const Vienna_Input_t *input, Vienna_Output_t *output, Vienna_State_t *state);

#ifdef __cplusplus
}
#endif

#endif /* VIENNA_CONTROL_H */
