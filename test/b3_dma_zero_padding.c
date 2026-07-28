#include <stdint.h>
#include <stdio.h>

#include "flexi/flexi.h"

#define N_ROWS 128
#define N_COLS 64
#define PAD_COLS 64
#define PADDED_COLS (N_COLS + PAD_COLS)
#define ONC_ADDR 0

static uint8_t src[N_ROWS][N_COLS] row_align(64);
static uint8_t dst[N_ROWS][PADDED_COLS] row_align(64);

int main(void)
{
    uint32_t mismatches = 0;

    for (uint32_t row = 0; row < N_ROWS; ++row) {
        for (uint32_t col = 0; col < N_COLS; ++col) {
            src[row][col] = (uint8_t)(
                ((row + 1) * UINT32_C(11) + col) & 0xFF
            );
        }
        for (uint32_t col = 0; col < PADDED_COLS; ++col)
            dst[row][col] = UINT8_C(0xA5);
    }

    DMALoad(
        src, ONC_ADDR, N_COLS, N_ROWS, N_COLS,
        0, INT8, PAD_COLS
    );

    /*
     * Store each full padded row. Passing zero_pad=0 intentionally keeps
     * the zero padding instead of stripping it from the on-chip layout.
     */
    DMAStore(
        dst, ONC_ADDR, PADDED_COLS, N_ROWS, PADDED_COLS,
        0, INT8, 0
    );
    FlexiFence();

    for (uint32_t row = 0; row < N_ROWS; ++row) {
        for (uint32_t col = 0; col < PADDED_COLS; ++col) {
            const uint8_t expected =
                col < N_COLS ? src[row][col] : UINT8_C(0);
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
        printf("DMA zero padding FAILED: %u mismatches\n", mismatches);
        return 1;
    }
    printf("DMA zero padding PASSED\n");
    return 0;
}
