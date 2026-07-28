#include "flexi_blas_common.h"

#include <stddef.h>

static int flexi_npu_errno;

float fabsf(float value) {
    union {
        float floating;
        unsigned int bits;
    } converted = { .floating = value };
    converted.bits &= 0x7fffffffu;
    return converted.floating;
}

float ldexpf(float value, int exponent) {
    while (exponent > 0) {
        value *= 2.0f;
        --exponent;
    }
    while (exponent < 0) {
        value *= 0.5f;
        ++exponent;
    }
    return value;
}

float expf(float value) {
    const float inverse_ln2 = 1.4426950408889634f;
    const float ln2 = 0.6931471805599453f;
    int exponent;
    float remainder;
    float result;

    if (value > 88.0f) return __builtin_inff();
    if (value < -103.0f) return 0.0f;
    exponent = (int)(value * inverse_ln2);
    if ((float)exponent > value * inverse_ln2) --exponent;
    remainder = value - (float)exponent * ln2;
    result = 1.0f + remainder * (1.0f + remainder * (0.5f + remainder * (0.1666666667f + remainder * (0.0416666667f + remainder * (0.0083333333f + remainder * 0.0013888889f)))));
    return ldexpf(result, exponent);
}

int* __errno(void) {
    return &flexi_npu_errno;
}
