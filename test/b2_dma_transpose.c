#include <stdint.h>
#include <stdio.h>

#include "flexi/flexi.h"

#define MATRIX_DIM 128
#define ONC_ADDR 0

static uint8_t src[MATRIX_DIM][MATRIX_DIM] row_align(64);
static uint8_t dst[MATRIX_DIM][MATRIX_DIM] row_align(64);

int main(void)
{
    uint32_t mismatches = 0;

    for (uint32_t row = 0; row < MATRIX_DIM; ++row) {
        for (uint32_t col = 0; col < MATRIX_DIM; ++col) {
            src[row][col] = (uint8_t)(
                (row * UINT32_C(3) + col * UINT32_C(5)) & 0xFF
            );
            dst[row][col] = 0;
        }
    }

    DMALoad(
        src, ONC_ADDR, MATRIX_DIM, MATRIX_DIM, MATRIX_DIM,
        1, INT8, 0
    );
    DMAStore(
        dst, ONC_ADDR, MATRIX_DIM, MATRIX_DIM, MATRIX_DIM,
        0, INT8, 0
    );
    FlexiFence();

    for (uint32_t row = 0; row < MATRIX_DIM; ++row) {
        for (uint32_t col = 0; col < MATRIX_DIM; ++col) {
            const uint8_t expected = src[col][row];
            if (dst[row][col] != expected) {
                if (mismatches < 8) {
                    printf(
                        "Mismatch row=%u col=%u: expected=%u actual=%u\n",
                        row, col, expected, dst[row][col]
                    );
                }
                ++mismatches;
            }
        }
    }

    if (mismatches != 0) {
        printf("DMA transpose FAILED: %u mismatches\n", mismatches);
        return 1;
    }
    printf("DMA transpose PASSED\n");
    return 0;
}
