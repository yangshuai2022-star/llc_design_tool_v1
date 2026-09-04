#ifndef POWER_CTRL_H
#define POWER_CTRL_H

#include <stdint.h>

#ifndef POWER_CTRL_FLOAT32_T_DEFINED
typedef float float32_t;
#define POWER_CTRL_FLOAT32_T_DEFINED
#endif

#define POWER_CTRL_FS_HZ (8.433350000e+04F)
#define POWER_CTRL_SECTIONS (1U)

#define POWER_CTRL_S1_B0 (1.300132760e+00F)
#define POWER_CTRL_S1_B1 (1.065939774e-01F)
#define POWER_CTRL_S1_B2 (-1.193538782e+00F)
#define POWER_CTRL_S1_A1 (-1.914327932e+00F)
#define POWER_CTRL_S1_A2 (9.143279317e-01F)

typedef struct { float32_t d1; float32_t d2; } POWER_CTRL_SECTION_T;
typedef struct { POWER_CTRL_SECTION_T section[POWER_CTRL_SECTIONS]; } POWER_CTRL_STATE_T;

static inline void POWER_CTRL_Reset(POWER_CTRL_STATE_T *state)
{
    uint32_t i;
    if (state == (void *)0) { return; }
    for (i = 0U; i < POWER_CTRL_SECTIONS; ++i)
    {
        state->section[i].d1 = 0.0F;
        state->section[i].d2 = 0.0F;
    }
}

/* DF2T/SOS implementation. Coefficient signs match displayed H(z). */
static inline float32_t POWER_CTRL_Run(POWER_CTRL_STATE_T *state, float32_t x)
{
    float32_t y = x;
    if (state == (void *)0) { return x; }
    {
        const float32_t in = y;
        y = POWER_CTRL_S1_B0 * in + state->section[0U].d1;
        state->section[0U].d1 = POWER_CTRL_S1_B1 * in - POWER_CTRL_S1_A1 * y + state->section[0U].d2;
        state->section[0U].d2 = POWER_CTRL_S1_B2 * in - POWER_CTRL_S1_A2 * y;
    }
    return y;
}

#endif
