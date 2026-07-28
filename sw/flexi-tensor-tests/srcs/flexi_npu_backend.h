#ifndef FLEXI_NPU_BACKEND_H
#define FLEXI_NPU_BACKEND_H

#include "flexi_blas_common.h"
#include "flexi_tensor_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t          mxu_dim;         // dimension of the MXU
    size_t          mxu_max_stride;  // maximum stride supported by the MXU
    fnblas_dtype_t  mxu_dtype;       // input data type of the MXU
    fnblas_dtype_t  mxu_res_dtype;   // output data type of the MXU
    uint64_t        onc_mem_size;    // size of the on-chip memory (in bytes)
    uint8_t         status;          // status of the NPU context (e.g., initialized or not)
} flexi_npu_context_t;

#ifdef __cplusplus
#define FLEXI_NPU_FP4_DEFAULT_CONTEXT flexi_npu_context_t{ \
    .mxu_dim = 128,                 \
    .mxu_max_stride = 512,          \
    .mxu_dtype = FP4,               \
    .mxu_res_dtype = BF16,          \
    .onc_mem_size = 512 * 1024,     \
    .status = 0                     \
}
#define FLEXI_NPU_BF16_DEFAULT_CONTEXT flexi_npu_context_t{ \
    .mxu_dim = 32,                 \
    .mxu_max_stride = 512,          \
    .mxu_dtype = BF16,               \
    .mxu_res_dtype = BF16,          \
    .onc_mem_size = 512 * 1024,     \
    .status = 0                     \
}
#else
#define FLEXI_NPU_FP4_DEFAULT_CONTEXT ((flexi_npu_context_t){ \
    .mxu_dim = 128,                 \
    .mxu_max_stride = 512,          \
    .mxu_dtype = FP4,               \
    .mxu_res_dtype = BF16,          \
    .onc_mem_size = 512 * 1024,     \
    .status = 0                     \
})
#define FLEXI_NPU_BF16_DEFAULT_CONTEXT ((flexi_npu_context_t){ \
    .mxu_dim = 32,                 \
    .mxu_max_stride = 512,          \
    .mxu_dtype = BF16,               \
    .mxu_res_dtype = BF16,          \
    .onc_mem_size = 512 * 1024,     \
    .status = 0                     \
})
#endif

extern flexi_npu_context_t _flexi_npu_current_context;
extern const flexi_tensor_backend_t flexi_npu_backend;

flexi_tensor_error_t flexi_npu_init(flexi_npu_context_t ctx);
int flexi_npu_is_initialized();
flexi_tensor_error_t flexi_npu_op_mm_matmul(fnblas_matrix_t* result, const fnblas_matrix_t* lhs, const fnblas_matrix_t* rhs_t);

#ifdef __cplusplus
}
#endif

#endif  // FLEXI_NPU_BACKEND_H