#ifndef CONTROL_RUNTIME_H
#define CONTROL_RUNTIME_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef float real32_t;

typedef enum
{
    CTRL_KIND_PI = 0,
    CTRL_KIND_PIF = 1,
    CTRL_KIND_2P2Z = 2
} CTRL_Kind_t;

typedef struct
{
    CTRL_Kind_t kind;
    real32_t kp;
    real32_t ki2;
    real32_t alpha;
    real32_t out_min;
    real32_t out_max;
    real32_t error_prev;
    real32_t i_state;
    real32_t out_prev;
    real32_t b0;
    real32_t b1;
    real32_t b2;
    real32_t a1;
    real32_t a2;
    real32_t x1;
    real32_t x2;
    real32_t y1;
    real32_t y2;
} CTRL_State_t;

real32_t CTRL_Clamp(real32_t value, real32_t minimum, real32_t maximum);
void CTRL_Reset(CTRL_State_t *state);
void CTRL_InitPI(CTRL_State_t *state, real32_t kp, real32_t ki2, real32_t out_min, real32_t out_max);
void CTRL_InitPIF(CTRL_State_t *state, real32_t kp, real32_t ki2, real32_t alpha, real32_t out_min, real32_t out_max);
void CTRL_Init2P2Z(CTRL_State_t *state, real32_t b0, real32_t b1, real32_t b2, real32_t a1, real32_t a2, real32_t out_min, real32_t out_max);
real32_t CTRL_Step(CTRL_State_t *state, real32_t error);

#ifdef __cplusplus
}
#endif

#endif /* CONTROL_RUNTIME_H */
