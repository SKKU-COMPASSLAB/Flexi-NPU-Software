#include <stdint.h>
#include <stdio.h>

#include "flexi/flexi.h"

#define ROW_NUMEL 32
#define TOTAL_ROWS 16
#define DRAM_STRIDE 40
#define ZERO_PAD 8
#define ONC_ADDR 0
#define GUARD_VALUE UINT8_C(0xA5)

static uint8_t src_data[TOTAL_ROWS * DRAM_STRIDE]
    __attribute__((aligned(64)));
static uint8_t dst_data[TOTAL_ROWS * DRAM_STRIDE]
    __attribute__((aligned(64)));

int main(void)
{
    uint32_t mismatches = 0;

    for (uint32_t row = 0; row < TOTAL_ROWS; ++row) {
        for (uint32_t col = 0; col < DRAM_STRIDE; ++col) {
            const uint32_t index = row * DRAM_STRIDE + col;
            src_data[index] = col < ROW_NUMEL
                ? (uint8_t)((row * ROW_NUMEL + col) & UINT8_C(0xFF))
                : GUARD_VALUE;
            dst_data[index] = GUARD_VALUE;
        }
    }

    DMALoad(src_data, ONC_ADDR, ROW_NUMEL, TOTAL_ROWS, DRAM_STRIDE, 0, INT8, ZERO_PAD);
    DMAStore(dst_data, ONC_ADDR, ROW_NUMEL, TOTAL_ROWS, DRAM_STRIDE, 0, INT8, ZERO_PAD);
    FlexiFence();

    for (uint32_t row = 0; row < TOTAL_ROWS; ++row) {
        for (uint32_t col = 0; col < DRAM_STRIDE; ++col) {
            const uint32_t index = row * DRAM_STRIDE + col;
            const uint8_t expected = col < ROW_NUMEL
                ? src_data[index]
                : GUARD_VALUE;
            if (dst_data[index] != expected) {
                if (mismatches < 8) {
                    printf(
                        "Mismatch row=%u col=%u: expected=%u actual=%u\n",
                        row,
                        col,
                        expected,
                        dst_data[index]
                    );
                }
                ++mismatches;
            }
        }
    }

    if (mismatches != 0) {
        printf("DRAM loopback FAILED: %u mismatches\n", mismatches);
        return 1;
    }

    printf("DRAM loopback PASSED\n");
    return 0;
}
