#include <stdint.h>
#include <stdio.h>

#include "flexi/flexi.h"

#define DIM 128
#define TILE_BYTES (DIM * DIM * sizeof(int32_t))
#define ONC_A 0
#define ONC_B (ONC_A + TILE_BYTES)
#define ONC_C (ONC_B + TILE_BYTES)
#define ONC_D (ONC_C + TILE_BYTES)

static int32_t a[DIM][DIM] row_align(64);
static int32_t b_t[DIM][DIM] row_align(64);
static int32_t bias[DIM][DIM] row_align(64);
static int32_t result[DIM][DIM] row_align(64);

int main(void)
{
    uint32_t mismatches = 0;

    for (uint32_t row = 0; row < DIM; ++row) {
        for (uint32_t col = 0; col < DIM; ++col) {
            a[row][col] = row == col ? 1 : 0;
            b_t[row][col] = (int32_t)(row * 3 + col * 2);
            bias[row][col] = 7;
            result[row][col] = 0;
        }
    }

    DMALoad(a, ONC_A, DIM, DIM, DIM, 0, INT32, 0);
    DMALoad(b_t, ONC_B, DIM, DIM, DIM, 0, INT32, 0);
    DMALoad(bias, ONC_C, DIM, DIM, DIM, 0, INT32, 0);
    GEMMPreload(ONC_B, DIM, DIM, MAT_B, INT32);
    GEMMPreload(ONC_C, DIM, DIM, MAT_C, INT32);
    GEMMExecuteS1(ONC_A, DIM, DIM, MAT_A, INT32, INT32);
    GEMMFlush(ONC_D, DIM, DIM, MAT_D, INT32);
    DMAStore(result, ONC_D, DIM, DIM, DIM, 0, INT32, 0);
    FlexiFence();

    for (uint32_t row = 0; row < DIM; ++row) {
        for (uint32_t col = 0; col < DIM; ++col) {
            const int32_t expected = b_t[col][row] + 7;
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
        printf("128x128 GEMM FAILED: %u mismatches\n", mismatches);
        return 1;
    }
    printf("128x128 GEMM PASSED\n");
    return 0;
}
