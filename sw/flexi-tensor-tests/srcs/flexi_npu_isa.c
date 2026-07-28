#include "flexi_npu_isa.h"

#include "rocc-software/src/xcustom.h"


/*
 * Flexi NPU Instructions
 */

#define XCUSTOM_ACC 3

#define OP_GEMM_PRELOAD 3
#define OP_GEMM_EXECUTE_S1 5
#define OP_GEMM_FLUSH 7
#define OP_DMA_LOAD 12
#define OP_DMA_STORE 13

#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) \
    ROCC_INSTRUCTION_0_R_R(x, rs1, rs2, funct)

#define ROCC_INSTRUCTION_RD_RS1_RS2(x, rd, rs1, rs2, funct) \
    ROCC_INSTRUCTION_R_R_R(x, rd, rs1, rs2, funct)

#define GEMMPreload(addr, row_numel, total_row, mat_type, dtype) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(addr) << 38), \
                                            ((uint64_t)(dtype)     << 38) | \
                                            ((uint64_t)(0x0)       << 37) | \
                                            ((uint64_t)(mat_type)  << 27) | \
                                            ((uint64_t)(total_row) << 17) | \
                                            ((uint64_t)(row_numel) << 7)  | \
                                            ((uint64_t)(0x0)       << 6)  | \
                                            ((uint64_t)(OP_GEMM_PRELOAD)), 0x0)

#define GEMMExecuteS1(addr, row_numel, total_row, mat_type, dtype, res_dtype) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(addr) << 38), \
                                            ((uint64_t)(dtype)     << 38) | \
                                            ((uint64_t)(0x0)       << 37) | \
                                            ((uint64_t)(res_dtype) << 30) | \
                                            ((uint64_t)(mat_type)  << 27) | \
                                            ((uint64_t)(total_row) << 17) | \
                                            ((uint64_t)(row_numel) << 7)  | \
                                            ((uint64_t)(0x0)       << 6)  | \
                                            ((uint64_t)(OP_GEMM_EXECUTE_S1)), 0x0)

#define GEMMFlush(addr, row_numel, total_row, mat_type, dtype) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(addr) << 38), \
                                            ((uint64_t)(dtype)     << 38) | \
                                            ((uint64_t)(0x0)       << 37) | \
                                            ((uint64_t)(mat_type)  << 27) | \
                                            ((uint64_t)(total_row) << 17) | \
                                            ((uint64_t)(row_numel) << 7)  | \
                                            ((uint64_t)(0x0)       << 6)  | \
                                            ((uint64_t)(OP_GEMM_FLUSH)), 0x0)
                                            
#define DMALoad(ofc_addr, onc_addr, row_numel, total_row, stride, trans, dtype, zero_pad) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(onc_addr) << 38) | ((uint64_t)(ofc_addr) >> 10), \
                                            ((uint64_t)(ofc_addr)  << 54) | \
                                            ((uint64_t)(zero_pad)  << 44) | \
                                            ((uint64_t)(dtype)     << 41) | \
                                            ((uint64_t)(trans)     << 40) | \
                                            ((uint64_t)(stride)    << 27) | \
                                            ((uint64_t)(total_row) << 17) | \
                                            ((uint64_t)(row_numel) << 7)  | \
                                            ((uint64_t)(0x0)       << 6)  | \
                                            ((uint64_t)(OP_DMA_LOAD)), 0x0)

#define DMAStore(ofc_addr, onc_addr, row_numel, total_row, stride, trans, dtype, zero_pad) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(onc_addr) << 38) | ((uint64_t)(ofc_addr) >> 10), \
                                            ((uint64_t)(ofc_addr)  << 54) | \
                                            ((uint64_t)(zero_pad)  << 44) | \
                                            ((uint64_t)(dtype)     << 41) | \
                                            ((uint64_t)(trans)     << 40) | \
                                            ((uint64_t)(stride)    << 27) | \
                                            ((uint64_t)(total_row) << 17) | \
                                            ((uint64_t)(row_numel) << 7)  | \
                                            ((uint64_t)(0x0)       << 6)  | \
                                            ((uint64_t)(OP_DMA_STORE)), 0x0)

#define FlexiFence() \
    asm volatile("fence")

#define FlexiSync(rd) \
    ROCC_INSTRUCTION_RD_RS1_RS2(XCUSTOM_ACC, rd, 0x0, 0x0, 0x0)

void flexi_npu_dma_load(uint64_t ofc_addr, uint64_t onc_addr, int row_numel, int total_row, int stride, bool trans, int data_type, int zero_pad) {
    DMALoad(ofc_addr, onc_addr, row_numel, total_row, stride, trans, data_type, zero_pad);
}

void flexi_npu_dma_store(uint64_t ofc_addr, uint64_t onc_addr, int row_numel, int total_row, int stride, bool trans, int data_type, int zero_pad) {
    DMAStore(ofc_addr, onc_addr, row_numel, total_row, stride, trans, data_type, zero_pad);
}

void flexi_npu_gemm_preload(uint64_t addr, int row_numel, int total_row, int mat_type, int data_type) {
    GEMMPreload(addr, row_numel, total_row, mat_type, data_type);
}

void flexi_npu_gemm_execute_s1(uint64_t addr, int row_numel, int total_row, int mat_type, int data_type, int res_data_type) {
    GEMMExecuteS1(addr, row_numel, total_row, mat_type, data_type, res_data_type);
}

void flexi_npu_gemm_flush(uint64_t addr, int row_numel, int total_row, int mat_type, int data_type) {
    GEMMFlush(addr, row_numel, total_row, mat_type, data_type);
}

void flexi_npu_sync() {
    volatile int done = 0;
    FlexiSync(done);
    __asm__ volatile("" : : "r"(done) : "memory");
}

void flexi_fence() {
    FlexiFence();
}

/*
 * Flexi HX Instructions
 */

#define XCUSTOM_HX 2
#define HX_FUNCT_SYNC 0
#define HX_FUNCT_FWHT 1

// status bits returned by hx_sync()
#define HX_ERR_LOGN  (1 << 0)
#define HX_ERR_NVECS (1 << 1)
#define HX_ERR_ALIGN (1 << 2)
#define HX_ERR_FAULT (1 << 3)

#define HX_RS2(dst, logn, nvecs)                                   \
    ((((uint64_t)(uintptr_t)(dst)) & 0x7FFFFFFFFFULL) |              \
     (((uint64_t)((logn) & 0xF)) << 40) |                            \
     (((uint64_t)((nvecs) & 0xFFFF)) << 44))

#define HX_FWHT_RAW(src, dst, logn, nvecs)                         \
    ROCC_INSTRUCTION_0_R_R(XCUSTOM_HX, ((uint64_t)(uintptr_t)(src)), \
                         HX_RS2(dst, logn, nvecs), HX_FUNCT_FWHT)

#define HX_SYNC_RAW(res) \
    ROCC_INSTRUCTION_R_R_R(XCUSTOM_HX, res, 0, 0, HX_FUNCT_SYNC)

int flexi_hx_fwht_bf16(const uint16_t *src, uint16_t *dst, int logn, int num_vecs) {
    flexi_hx_fwht_bf16_async(src, dst, logn, num_vecs);
    return flexi_hx_sync();
}

int flexi_hx_fwht_bf16_async(const uint16_t *src, uint16_t *dst, int logn, int num_vecs) {
    asm volatile("fence"); // make CPU-written source data visible to the accelerator
    HX_FWHT_RAW(src, dst, logn, num_vecs);
    return 0;
}

int flexi_hx_sync(void) {
    uint64_t r;
    HX_SYNC_RAW(r);
    return (int)r;
}
