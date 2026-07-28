#ifndef FLEXI_NPU_ISA_H
#define FLEXI_NPU_ISA_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>

#include "rocc-software/src/xcustom.h"
#ifndef BAREMETAL
    #include <sys/mman.h>
#endif

#define row_align(bytes) __attribute__((aligned(bytes)))

// typedef enum {
//     INT32 = 0,
//     FP4 = 1,
//     INT4 = 3,
//     FP8 = 5,
//     BF16 = 7
// } flexi_npu_dtype_t;

// typedef enum {
//     MAT_A = 0,
//     MAT_B,
//     MAT_C,
//     MAT_D
// } flexi_npu_mat_type_t;

#define FLEXI_NPU_MAT_TYPE_A 0
#define FLEXI_NPU_MAT_TYPE_B 1
#define FLEXI_NPU_MAT_TYPE_C 2
#define FLEXI_NPU_MAT_TYPE_D 3

void flexi_npu_dma_load(uint64_t ofc_addr, uint64_t onc_addr, int rows, int cols, int row_stride, bool transposed, int data_type, int pad);
void flexi_npu_dma_store(uint64_t ofc_addr, uint64_t onc_addr, int rows, int cols, int row_stride, bool transposed, int data_type, int pad);
void flexi_npu_gemm_preload(uint64_t addr, int rows, int cols, int mat_type, int data_type);
void flexi_npu_gemm_execute_s1(uint64_t addr, int rows, int cols, int mat_type, int data_type, int res_data_type);
void flexi_npu_gemm_flush(uint64_t addr, int rows, int cols, int mat_type, int data_type);
void flexi_npu_sync();

int  flexi_hx_fwht_bf16(const uint16_t *src, uint16_t *dst, int logn, int num_vecs);
int  flexi_hx_fwht_bf16_async(const uint16_t *src, uint16_t *dst, int logn, int num_vecs);
int  flexi_hx_sync();

void flexi_fence();

#endif  // FLEXI_NPU_ISA_H