#include <stdint.h>
#include <stdio.h>

#include "flexi/flexi.h"

#define MATRIX_DIM 256
#define TILE_DIM 128
#define ONC_ADDR 0

static uint8_t src[MATRIX_DIM][MATRIX_DIM] row_align(64);
static uint8_t dst[MATRIX_DIM][MATRIX_DIM] row_align(64);

int main(void)
{
    uint32_t mismatches = 0;

    for (uint32_t row = 0; row < MATRIX_DIM; ++row) {
        for (uint32_t col = 0; col < MATRIX_DIM; ++col) {
            src[row][col] = (uint8_t)(
                (row * UINT32_C(17) + col * UINT32_C(29)) & 0xFF
            );
            dst[row][col] = 0;
        }
    }

    const uint32_t tile_size = TILE_DIM * TILE_DIM * sizeof(uint8_t);
    const uint32_t n_tiles_per_row = MATRIX_DIM / TILE_DIM;
    uint32_t onc_addr = ONC_ADDR;

    for (uint32_t tile_row = 0; tile_row < MATRIX_DIM; tile_row += TILE_DIM) {
        for (uint32_t tile_col = 0; tile_col < MATRIX_DIM; tile_col += TILE_DIM) {
            onc_addr = ONC_ADDR + tile_size * (tile_row / TILE_DIM * n_tiles_per_row + tile_col / TILE_DIM);
            DMALoad (&src[tile_row][tile_col], onc_addr, TILE_DIM, TILE_DIM, MATRIX_DIM, 0, INT8, 0);
        }
    }

    for (uint32_t tile_row = 0; tile_row < MATRIX_DIM; tile_row += TILE_DIM) {
        for (uint32_t tile_col = 0; tile_col < MATRIX_DIM; tile_col += TILE_DIM) {
            onc_addr = ONC_ADDR + tile_size * (tile_row / TILE_DIM * n_tiles_per_row + tile_col / TILE_DIM);
            DMAStore(&dst[tile_row][tile_col], onc_addr, TILE_DIM, TILE_DIM, MATRIX_DIM, 0, INT8, 0);
        }
    }
    FlexiFence();

    for (uint32_t row = 0; row < MATRIX_DIM; ++row) {
        for (uint32_t col = 0; col < MATRIX_DIM; ++col) {
            if (dst[row][col] != src[row][col]) {
                if (mismatches < 8) {
                    printf(
                        "Mismatch row=%u col=%u: expected=%u actual=%u\n",
                        row, col, src[row][col], dst[row][col]
                    );
                }
                ++mismatches;
            }
        }
    }

    if (mismatches != 0) {
        printf("Tiled DRAM loopback FAILED: %u mismatches\n", mismatches);
        return 1;
    }
    printf("Tiled DRAM loopback PASSED\n");
    return 0;
}
