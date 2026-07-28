#include "flexi_npu_backend.h"
#include "flexi_npu_isa.h"
#include "flexi_blas_internal.h"

#include <string.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

flexi_npu_context_t _flexi_npu_current_context = {0};

const flexi_tensor_backend_t flexi_npu_backend = {
    .op_mm_matmul = flexi_npu_op_mm_matmul
};

static size_t flexi_npu_round_up(size_t value, size_t alignment) {
    if (alignment == 0 || value > SIZE_MAX - (alignment - 1)) return 0;
    return ((value + alignment - 1) / alignment) * alignment;
}

static size_t flexi_npu_packed_row_size(size_t n_elements, fnblas_dtype_t dtype) {
    const size_t dtype_size = fnblas_dtype_size_of(dtype);
    const size_t pack_size = fnblas_dtype_pack_size_of(dtype);
    if (dtype_size == 0 || pack_size == 0 || n_elements > SIZE_MAX / dtype_size) return 0;
    return (n_elements * dtype_size + pack_size - 1) / pack_size;
}

static size_t flexi_npu_packed_offset(size_t row, size_t row_stride, size_t col, fnblas_dtype_t dtype) {
    const size_t dtype_size = fnblas_dtype_size_of(dtype);
    const size_t pack_size = fnblas_dtype_pack_size_of(dtype);
    return row * flexi_npu_packed_row_size(row_stride, dtype) + col * dtype_size / pack_size;
}

flexi_tensor_error_t flexi_npu_init(flexi_npu_context_t ctx) {
    _flexi_npu_current_context.mxu_dim = ctx.mxu_dim;
    _flexi_npu_current_context.mxu_max_stride = ctx.mxu_max_stride;
    _flexi_npu_current_context.mxu_dtype = ctx.mxu_dtype;
    _flexi_npu_current_context.mxu_res_dtype = ctx.mxu_res_dtype;
    _flexi_npu_current_context.onc_mem_size = ctx.onc_mem_size;
    _flexi_npu_current_context.status = 1;  // Mark the NPU as initialized
    return FLEXI_TENSOR_SUCCESS;
}

int flexi_npu_is_initialized() {
    return _flexi_npu_current_context.status == 1;
}

flexi_tensor_error_t flexi_npu_op_mm_matmul(fnblas_matrix_t* mat_d, const fnblas_matrix_t* mat_a, const fnblas_matrix_t* mat_b_t) {
    flexi_tensor_error_t result = FLEXI_TENSOR_ERR_BACKEND;
    fnblas_error_t blas_error;
    fnblas_matrix_t mat_a_shard = FNBLAS_MATRIX_INITIALIZER;
    fnblas_matrix_t mat_b_shard = FNBLAS_MATRIX_INITIALIZER;
    fnblas_matrix_t mat_d_shard = FNBLAS_MATRIX_INITIALIZER;
    byte_t* mat_a_packed = NULL;
    byte_t* mat_b_packed = NULL;
    byte_t* mat_d_packed = NULL;
    size_t input_element_size;
    size_t result_element_size;
    size_t input_pack_size;
    size_t result_pack_size;
    size_t N_shard;
    size_t K_shard;
    size_t N_storage;
    size_t K_storage;
    size_t num_N_shards;
    size_t num_K_shards;
    size_t onc_region_size;
    uint64_t onc_addr_mat_a;
    uint64_t onc_addr_mat_b;
    uint64_t onc_addr_mat_c;
    uint64_t onc_addr_mat_d;
    int output_created = 0;
    if (!flexi_npu_is_initialized() || mat_d == NULL || mat_a == NULL || mat_b_t == NULL) return FLEXI_TENSOR_ERR_BACKEND;
    const size_t M = fnblas_matrix_n_rows(mat_a);
    const size_t N = fnblas_matrix_n_rows(mat_b_t);
    const size_t K = fnblas_matrix_n_cols(mat_a);
    const fnblas_dtype_t input_dtype = fnblas_matrix_dtype(mat_a);
    const fnblas_dtype_t result_dtype = _flexi_npu_current_context.mxu_res_dtype;
    const size_t M_tile  = _flexi_npu_current_context.mxu_dim;
    const size_t N_tile  = _flexi_npu_current_context.mxu_dim;
    const size_t K_tile  = _flexi_npu_current_context.mxu_dim;
    if (M == 0 || N == 0 || K == 0 || M_tile == 0 || _flexi_npu_current_context.mxu_max_stride == 0) return FLEXI_TENSOR_ERR_MISMATCH;
    if (!_fnblas_buffer_is_allocated(mat_a->_status) || !_fnblas_buffer_is_allocated(mat_b_t->_status)) return FLEXI_TENSOR_ERR_MISMATCH;
    if (fnblas_matrix_n_cols(mat_b_t) != K || fnblas_matrix_dtype(mat_b_t) != input_dtype || input_dtype != _flexi_npu_current_context.mxu_dtype) return FLEXI_TENSOR_ERR_MISMATCH;
    if (M_tile > 1023 || _flexi_npu_current_context.mxu_max_stride > 8191 || _flexi_npu_current_context.onc_mem_size > (UINT64_C(1) << 26)) return FLEXI_TENSOR_ERR_MISMATCH;
    if (_fnblas_buffer_is_allocated(mat_d->_status)) {
        if (fnblas_matrix_n_rows(mat_d) != M || fnblas_matrix_n_cols(mat_d) != N || fnblas_matrix_dtype(mat_d) != result_dtype) return FLEXI_TENSOR_ERR_MISMATCH;
    } else {
        blas_error = fnblas_matrix_create(mat_d, M, N, result_dtype);
        if (blas_error != FNBLAS_SUCCESS) return blas_error == FNBLAS_ERR_MALLOC_FAIL ? FLEXI_TENSOR_ERR_MALLOC_FAIL : FLEXI_TENSOR_ERR_BACKEND;
        output_created = 1;
    }
    if (mat_d == mat_a || mat_d == mat_b_t || mat_d->_buffer == mat_a->_buffer || mat_d->_buffer == mat_b_t->_buffer) {
        result = FLEXI_TENSOR_ERR_MISMATCH;
        goto _flexi_npu_op_mm_matmul_cleanup;
    }
    input_element_size = _fnblas_dtype_unpacked_size_of(input_dtype);
    result_element_size = _fnblas_dtype_unpacked_size_of(result_dtype);
    input_pack_size = fnblas_dtype_pack_size_of(input_dtype);
    result_pack_size = fnblas_dtype_pack_size_of(result_dtype);
    if (input_element_size == 0 || result_element_size == 0 || input_pack_size == 0 || result_pack_size == 0) {
        result = FLEXI_TENSOR_ERR_MISMATCH;
        goto _flexi_npu_op_mm_matmul_cleanup;
    }
    if (K_tile % input_pack_size != 0 || N_tile % result_pack_size != 0) {
        result = FLEXI_TENSOR_ERR_MISMATCH;
        goto _flexi_npu_op_mm_matmul_cleanup;
    }
    N_shard = MIN(_flexi_npu_current_context.mxu_max_stride, N);
    K_shard = MIN(_flexi_npu_current_context.mxu_max_stride, K);
    N_storage = flexi_npu_round_up(N_shard, result_pack_size);
    K_storage = flexi_npu_round_up(K_shard, input_pack_size);
    if (N_storage == 0 || K_storage == 0 || N_storage > _flexi_npu_current_context.mxu_max_stride || K_storage > _flexi_npu_current_context.mxu_max_stride) {
        result = FLEXI_TENSOR_ERR_MISMATCH;
        goto _flexi_npu_op_mm_matmul_cleanup;
    }
    num_N_shards = (N + N_shard - 1) / N_shard;
    num_K_shards = (K + K_shard - 1) / K_shard;
    onc_region_size = _flexi_npu_current_context.onc_mem_size / 4;
    if (onc_region_size == 0 || flexi_npu_packed_row_size(M_tile, input_dtype) > SIZE_MAX / M_tile || flexi_npu_packed_row_size(N_tile, result_dtype) > SIZE_MAX / M_tile) {
        result = FLEXI_TENSOR_ERR_MISMATCH;
        goto _flexi_npu_op_mm_matmul_cleanup;
    }
    if (flexi_npu_packed_row_size(K_tile, input_dtype) * M_tile > onc_region_size || flexi_npu_packed_row_size(K_tile, input_dtype) * N_tile > onc_region_size || flexi_npu_packed_row_size(N_tile, result_dtype) * M_tile > onc_region_size) {
        result = FLEXI_TENSOR_ERR_MISMATCH;
        goto _flexi_npu_op_mm_matmul_cleanup;
    }
    onc_addr_mat_a = onc_region_size * 0;
    onc_addr_mat_b = onc_region_size * 1;
    onc_addr_mat_c = onc_region_size * 2;
    onc_addr_mat_d = onc_region_size * 3;
    blas_error = fnblas_matrix_create(&mat_a_shard, M, K_storage, input_dtype);
    if (blas_error != FNBLAS_SUCCESS) goto _flexi_npu_op_mm_matmul_blas_failure;
    blas_error = fnblas_matrix_create(&mat_b_shard, N_storage, K_storage, input_dtype);
    if (blas_error != FNBLAS_SUCCESS) goto _flexi_npu_op_mm_matmul_blas_failure;
    blas_error = fnblas_matrix_create(&mat_d_shard, M, N_storage, result_dtype);
    if (blas_error != FNBLAS_SUCCESS) goto _flexi_npu_op_mm_matmul_blas_failure;
    blas_error = _fnblas_allocate_buffer(&mat_a_packed, M * K_storage, input_dtype);
    if (blas_error != FNBLAS_SUCCESS) goto _flexi_npu_op_mm_matmul_blas_failure;
    blas_error = _fnblas_allocate_buffer(&mat_b_packed, N_storage * K_storage, input_dtype);
    if (blas_error != FNBLAS_SUCCESS) goto _flexi_npu_op_mm_matmul_blas_failure;
    blas_error = _fnblas_allocate_buffer(&mat_d_packed, M * N_storage, result_dtype);
    if (blas_error != FNBLAS_SUCCESS) goto _flexi_npu_op_mm_matmul_blas_failure;
    for (size_t n_s = 0; n_s < num_N_shards; ++n_s) {
        const size_t n_begin = n_s * N_shard;
        const size_t n_access_shard = MIN(N_shard, N - n_begin);
        memset(mat_d_packed, 0, _fnblas_packed_buffer_size(M * N_storage, result_dtype));
        for (size_t k_s = 0; k_s < num_K_shards; ++k_s) {
            const size_t k_begin = k_s * K_shard;
            const size_t k_access_shard = MIN(K_shard, K - k_begin);
            memset(mat_a_shard._buffer, 0, M * K_storage * input_element_size);
            memset(mat_b_shard._buffer, 0, N_storage * K_storage * input_element_size);
            for (size_t row = 0; row < M; ++row) memcpy(mat_a_shard._buffer + row * K_storage * input_element_size, mat_a->_buffer + (row * K + k_begin) * input_element_size, k_access_shard * input_element_size);
            for (size_t row = 0; row < n_access_shard; ++row) memcpy(mat_b_shard._buffer + row * K_storage * input_element_size, mat_b_t->_buffer + ((n_begin + row) * K + k_begin) * input_element_size, k_access_shard * input_element_size);
            blas_error = fnblas_matrix_get_packed_buffer(&mat_a_shard, mat_a_packed);
            if (blas_error != FNBLAS_SUCCESS) goto _flexi_npu_op_mm_matmul_blas_failure;
            blas_error = fnblas_matrix_get_packed_buffer(&mat_b_shard, mat_b_packed);
            if (blas_error != FNBLAS_SUCCESS) goto _flexi_npu_op_mm_matmul_blas_failure;
            flexi_fence();
            for (size_t m_begin = 0; m_begin < M; m_begin += M_tile) {
                const size_t m_access = MIN(M_tile, M - m_begin);
                for (size_t n_tile_begin = 0; n_tile_begin < n_access_shard; n_tile_begin += N_tile) {
                    const size_t n_access = MIN(N_tile, n_access_shard - n_tile_begin);
                    const size_t n_pad = N_tile - n_access;
                    for (size_t k_tile_begin = 0; k_tile_begin < k_access_shard; k_tile_begin += K_tile) {
                        const size_t k_access = MIN(K_tile, k_access_shard - k_tile_begin);
                        const size_t k_pad = K_tile - k_access;
                        const uint64_t ofc_addr_mat_a = (uint64_t)(uintptr_t)(mat_a_packed + flexi_npu_packed_offset(m_begin, K_storage, k_tile_begin, input_dtype));
                        const uint64_t ofc_addr_mat_b = (uint64_t)(uintptr_t)(mat_b_packed + flexi_npu_packed_offset(n_tile_begin, K_storage, k_tile_begin, input_dtype));
                        const uint64_t ofc_addr_mat_c = (uint64_t)(uintptr_t)(mat_d_packed + flexi_npu_packed_offset(m_begin, N_storage, n_tile_begin, result_dtype));
                        flexi_npu_dma_load(ofc_addr_mat_c, onc_addr_mat_c, (int)n_access, (int)m_access, (int)N_storage, false, result_dtype, (int)n_pad);
                        flexi_npu_dma_load(ofc_addr_mat_b, onc_addr_mat_b, (int)k_access, (int)n_access, (int)K_storage, false, input_dtype, (int)k_pad);
                        flexi_npu_dma_load(ofc_addr_mat_a, onc_addr_mat_a, (int)k_access, (int)m_access, (int)K_storage, false, input_dtype, (int)k_pad);
                        flexi_npu_gemm_preload(onc_addr_mat_c, (int)N_tile, (int)m_access, FLEXI_NPU_MAT_TYPE_C, result_dtype);
                        flexi_npu_gemm_preload(onc_addr_mat_b, (int)K_tile, (int)n_access, FLEXI_NPU_MAT_TYPE_B, input_dtype);
                        flexi_npu_gemm_execute_s1(onc_addr_mat_a, (int)K_tile, (int)m_access, FLEXI_NPU_MAT_TYPE_A, input_dtype, result_dtype);
                        flexi_npu_gemm_flush(onc_addr_mat_d, (int)N_tile, (int)m_access, FLEXI_NPU_MAT_TYPE_D, result_dtype);
                        flexi_npu_dma_store(ofc_addr_mat_c, onc_addr_mat_d, (int)n_access, (int)m_access, (int)N_storage, false, result_dtype, (int)n_pad);
                        flexi_npu_sync();
                    }
                }
            }
        }
        flexi_fence();
        blas_error = fnblas_matrix_initialize_from_packed_buffer(&mat_d_shard, mat_d_packed);
        if (blas_error != FNBLAS_SUCCESS) goto _flexi_npu_op_mm_matmul_blas_failure;
        for (size_t row = 0; row < M; ++row) memcpy(mat_d->_buffer + (row * N + n_begin) * result_element_size, mat_d_shard._buffer + row * N_storage * result_element_size, n_access_shard * result_element_size);
    }
    result = FLEXI_TENSOR_SUCCESS;
    goto _flexi_npu_op_mm_matmul_cleanup;

_flexi_npu_op_mm_matmul_blas_failure:
    result = blas_error == FNBLAS_ERR_MALLOC_FAIL ? FLEXI_TENSOR_ERR_MALLOC_FAIL : blas_error == FNBLAS_ERR_MISMATCH ? FLEXI_TENSOR_ERR_MISMATCH : FLEXI_TENSOR_ERR_BACKEND;

_flexi_npu_op_mm_matmul_cleanup:
    _fnblas_deallocate_buffer(mat_d_packed);
    _fnblas_deallocate_buffer(mat_b_packed);
    _fnblas_deallocate_buffer(mat_a_packed);
    fnblas_matrix_destroy(&mat_d_shard);
    fnblas_matrix_destroy(&mat_b_shard);
    fnblas_matrix_destroy(&mat_a_shard);
    if (result != FLEXI_TENSOR_SUCCESS && output_created) fnblas_matrix_destroy(mat_d);
    return result;
}
