#include <stdint.h>
#include <stdio.h>

#include "flexi/flexi.h"

#define LOG_N 4
#define N (1U << LOG_N)
#define N_VECS 2

static uint16_t src[N_VECS][N] row_align(64);
static uint16_t dst[N_VECS][N] row_align(64);

int main(void)
{
    uint32_t mismatches = 0;
    uint64_t status = 0;

    /* BF16: 4.0 = 0x4080, 1.0 = 0x3f80. */
    src[0][0] = UINT16_C(0x4080);
    for (uint32_t index = 0; index < N; ++index)
        src[1][index] = UINT16_C(0x3F80);

    HX_FWHT_RAW(src, dst, LOG_N, N_VECS);
    HX_SYNC_RAW(status);
    FlexiFence();

    if (status != 0) {
        printf("HX sync returned status=0x%lx\n", status);
        return 1;
    }

    for (uint32_t index = 0; index < N; ++index) {
        const uint16_t expected = UINT16_C(0x3F80);
        if (dst[0][index] != expected) {
            if (mismatches < 8) {
                printf(
                    "Vector 0 mismatch index=%u: expected=0x%x actual=0x%x\n",
                    index, expected, dst[0][index]
                );
            }
            ++mismatches;
        }
    }

    for (uint32_t index = 0; index < N; ++index) {
        const uint16_t expected =
            index == 0 ? UINT16_C(0x4080) : UINT16_C(0);
        if (dst[1][index] != expected) {
            if (mismatches < 8) {
                printf(
                    "Vector 1 mismatch index=%u: expected=0x%x actual=0x%x\n",
                    index, expected, dst[1][index]
                );
            }
            ++mismatches;
        }
    }

    if (mismatches != 0) {
        printf("HX FWHT FAILED: %u mismatches\n", mismatches);
        return 1;
    }
    printf("HX FWHT PASSED\n");
    return 0;
}
