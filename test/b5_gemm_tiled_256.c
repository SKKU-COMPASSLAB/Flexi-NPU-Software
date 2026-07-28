#include <stdint.h>
#include <stdio.h>

#include "flexi/flexi.h"

#define DIM 256
#define TILE 128
#define TILE_BYTES (TILE * TILE * sizeof(int32_t))
#define ONC_A 0
#define ONC_B (ONC_A + TILE_BYTES)
#define ONC_C (ONC_B + TILE_BYTES)
#define ONC_D (ONC_C + TILE_BYTES)

static int32_t a[DIM][DIM] row_align(64);
static int32_t b_t[DIM][DIM] row_align(64);
static int32_t zero_tile[TILE][TILE] row_align(64);
static int32_t result[DIM][DIM] row_align(64);

int main(void)
{
    uint32_t mismatches = 0;

    for (uint32_t row = 0; row < DIM; ++row) {
        for (uint32_t col = 0; col < DIM; ++col) {
            a[row][col] = row == col ? 1 : 0;
            b_t[row][col] = (int32_t)(row * 5 + col * 7);
            result[row][col] = 0;
        }
    }

    for (uint32_t out_row = 0; out_row < DIM; out_row += TILE) {
        for (uint32_t out_col = 0; out_col < DIM; out_col += TILE) {
            for (uint32_t reduction = 0;
                 reduction < DIM;
                 reduction += TILE) {
                DMALoad(
                    &a[out_row][reduction],
                    ONC_A, TILE, TILE, DIM, 0, INT32, 0
                );
                DMALoad(
                    &b_t[out_col][reduction],
                    ONC_B, TILE, TILE, DIM, 0, INT32, 0
                );
                if (reduction == 0) {
                    DMALoad(
                        zero_tile,
                        ONC_C, TILE, TILE, TILE, 0, INT32, 0
                    );
                } else {
                    DMALoad(
                        &result[out_row][out_col],
                        ONC_C, TILE, TILE, DIM, 0, INT32, 0
                    );
                }

                GEMMPreload(ONC_B, TILE, TILE, MAT_B, INT32);
                GEMMPreload(ONC_C, TILE, TILE, MAT_C, INT32);
                GEMMExecuteS1(
                    ONC_A, TILE, TILE, MAT_A, INT32, INT32
                );
                GEMMFlush(ONC_D, TILE, TILE, MAT_D, INT32);
                DMAStore(
                    &result[out_row][out_col],
                    ONC_D, TILE, TILE, DIM, 0, INT32, 0
                );
            }
        }
    }
    FlexiFence();

    for (uint32_t row = 0; row < DIM; ++row) {
        for (uint32_t col = 0; col < DIM; ++col) {
            const int32_t expected = b_t[col][row];
            if (result[row][col] != expected) {
                if (mismatches < 8) {
                    printf(
                        "Mismatch row=%u col=%u: expected=%d actual=%d\n",
                        row, col, expected, result[row][col]
                    );
                }
                ++mismatches;
            }
        }
    }

    if (mismatches != 0) {
        printf("Tiled 256x256 GEMM FAILED: %u mismatches\n", mismatches);
        return 1;
    }
    printf("Tiled 256x256 GEMM PASSED\n");
    return 0;
}
