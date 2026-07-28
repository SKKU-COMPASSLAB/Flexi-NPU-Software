#include "flexi_tensor_backend_internal.h"
#include "flexi_blas_internal.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

static flexi_tensor_error_t flexi_tensor_from_fnblas_error(fnblas_error_t error) {
    switch (error) {
        case FNBLAS_SUCCESS:
            return FLEXI_TENSOR_SUCCESS;
        case FNBLAS_ERR_MALLOC_FAIL:
            return FLEXI_TENSOR_ERR_MALLOC_FAIL;
        case FNBLAS_ERR_MISMATCH:
            return FLEXI_TENSOR_ERR_MISMATCH;
        case FNBLAS_ERR_UNKNOWN:
        default:
            return FLEXI_TENSOR_ERR_BACKEND;
    }
}

static flexi_tensor_error_t fnblas_backend_op_sv_add(fnblas_vector_t* result, const fnblas_vector_t* lhs, const fnblas_scalar_t* rhs) {
    return flexi_tensor_from_fnblas_error(fnblas_op_sv_add(result, lhs, rhs));
}

static flexi_tensor_error_t fnblas_backend_op_sv_sub(fnblas_vector_t* result, const fnblas_vector_t* lhs, const fnblas_scalar_t* rhs) {
    return flexi_tensor_from_fnblas_error(fnblas_op_sv_sub(result, lhs, rhs));
}

static flexi_tensor_error_t fnblas_backend_op_sv_mul(fnblas_vector_t* result, const fnblas_vector_t* lhs, const fnblas_scalar_t* rhs) {
    return flexi_tensor_from_fnblas_error(fnblas_op_sv_mul(result, lhs, rhs));
}

static flexi_tensor_error_t fnblas_backend_op_sv_div(fnblas_vector_t* result, const fnblas_vector_t* lhs, const fnblas_scalar_t* rhs) {
    return flexi_tensor_from_fnblas_error(fnblas_op_sv_div(result, lhs, rhs));
}

static flexi_tensor_error_t fnblas_backend_op_sv_max(fnblas_vector_t* result, const fnblas_vector_t* lhs, const fnblas_scalar_t* rhs) {
    return flexi_tensor_from_fnblas_error(fnblas_op_sv_max(result, lhs, rhs));
}

static flexi_tensor_error_t fnblas_backend_op_sv_min(fnblas_vector_t* result, const fnblas_vector_t* lhs, const fnblas_scalar_t* rhs) {
    return flexi_tensor_from_fnblas_error(fnblas_op_sv_min(result, lhs, rhs));
}

static flexi_tensor_error_t fnblas_backend_op_ve_add(fnblas_vector_t* result, const fnblas_vector_t* lhs, const fnblas_vector_t* rhs) {
    return flexi_tensor_from_fnblas_error(fnblas_op_ve_add(result, lhs, rhs));
}

static flexi_tensor_error_t fnblas_backend_op_ve_sub(fnblas_vector_t* result, const fnblas_vector_t* lhs, const fnblas_vector_t* rhs) {
    return flexi_tensor_from_fnblas_error(fnblas_op_ve_sub(result, lhs, rhs));
}

static flexi_tensor_error_t fnblas_backend_op_ve_mul(fnblas_vector_t* result, const fnblas_vector_t* lhs, const fnblas_vector_t* rhs) {
    return flexi_tensor_from_fnblas_error(fnblas_op_ve_mul(result, lhs, rhs));
}

static flexi_tensor_error_t fnblas_backend_op_ve_div(fnblas_vector_t* result, const fnblas_vector_t* lhs, const fnblas_vector_t* rhs) {
    return flexi_tensor_from_fnblas_error(fnblas_op_ve_div(result, lhs, rhs));
}

static flexi_tensor_error_t fnblas_backend_op_ve_max(fnblas_vector_t* result, const fnblas_vector_t* lhs, const fnblas_vector_t* rhs) {
    return flexi_tensor_from_fnblas_error(fnblas_op_ve_max(result, lhs, rhs));
}

static flexi_tensor_error_t fnblas_backend_op_ve_min(fnblas_vector_t* result, const fnblas_vector_t* lhs, const fnblas_vector_t* rhs) {
    return flexi_tensor_from_fnblas_error(fnblas_op_ve_min(result, lhs, rhs));
}

static flexi_tensor_error_t fnblas_backend_op_vv_dot(fnblas_scalar_t* result, const fnblas_vector_t* lhs, const fnblas_vector_t* rhs) {
    return flexi_tensor_from_fnblas_error(fnblas_op_vv_dot(result, lhs, rhs));
}

static flexi_tensor_error_t fnblas_backend_op_vr_sum(fnblas_scalar_t* result, const fnblas_vector_t* input) {
    return flexi_tensor_from_fnblas_error(fnblas_op_vr_sum(result, input));
}

static flexi_tensor_error_t fnblas_backend_op_vr_mean(fnblas_scalar_t* result, const fnblas_vector_t* input) {
    return flexi_tensor_from_fnblas_error(fnblas_op_vr_mean(result, input));
}

static flexi_tensor_error_t fnblas_backend_op_vr_max(fnblas_scalar_t* result, const fnblas_vector_t* input) {
    return flexi_tensor_from_fnblas_error(fnblas_op_vr_max(result, input));
}

static flexi_tensor_error_t fnblas_backend_op_vr_min(fnblas_scalar_t* result, const fnblas_vector_t* input) {
    return flexi_tensor_from_fnblas_error(fnblas_op_vr_min(result, input));
}

static flexi_tensor_error_t fnblas_backend_op_vi_exp(fnblas_vector_t* result, const fnblas_vector_t* input) {
    return flexi_tensor_from_fnblas_error(fnblas_op_vi_exp(result, input));
}

static flexi_tensor_error_t fnblas_backend_op_mi_transpose(fnblas_matrix_t* result, const fnblas_matrix_t* input) {
    return flexi_tensor_from_fnblas_error(fnblas_op_mi_transpose(result, input));
}

static flexi_tensor_error_t fnblas_backend_op_me_add(fnblas_matrix_t* result, const fnblas_matrix_t* lhs, const fnblas_matrix_t* rhs) {
    return flexi_tensor_from_fnblas_error(fnblas_op_me_add(result, lhs, rhs));
}

static flexi_tensor_error_t fnblas_backend_op_me_sub(fnblas_matrix_t* result, const fnblas_matrix_t* lhs, const fnblas_matrix_t* rhs) {
    return flexi_tensor_from_fnblas_error(fnblas_op_me_sub(result, lhs, rhs));
}

static flexi_tensor_error_t fnblas_backend_op_me_mul(fnblas_matrix_t* result, const fnblas_matrix_t* lhs, const fnblas_matrix_t* rhs) {
    return flexi_tensor_from_fnblas_error(fnblas_op_me_mul(result, lhs, rhs));
}

static flexi_tensor_error_t fnblas_backend_op_me_div(fnblas_matrix_t* result, const fnblas_matrix_t* lhs, const fnblas_matrix_t* rhs) {
    return flexi_tensor_from_fnblas_error(fnblas_op_me_div(result, lhs, rhs));
}

static flexi_tensor_error_t fnblas_backend_op_mm_matmul(fnblas_matrix_t* result, const fnblas_matrix_t* lhs, const fnblas_matrix_t* rhs) {
    return flexi_tensor_from_fnblas_error(fnblas_op_mm_matmul(result, lhs, rhs));
}

static flexi_tensor_error_t fnblas_backend_op_mmv_axpy(fnblas_matrix_t* result, const fnblas_matrix_t* a, const fnblas_matrix_t* x_t, const fnblas_vector_t* y) {
    return flexi_tensor_from_fnblas_error(fnblas_op_mmv_axpy(result, a, x_t, y));
}

static flexi_tensor_error_t fnblas_backend_op_mmm_axpy(fnblas_matrix_t* result, const fnblas_matrix_t* a, const fnblas_matrix_t* x_t, const fnblas_matrix_t* y) {
    return flexi_tensor_from_fnblas_error(fnblas_op_mmm_axpy(result, a, x_t, y));
}

#define FLEXI_TENSOR_FNBLAS_BACKEND_INITIALIZER { \
    .op_sv_add = fnblas_backend_op_sv_add, \
    .op_sv_sub = fnblas_backend_op_sv_sub, \
    .op_sv_mul = fnblas_backend_op_sv_mul, \
    .op_sv_div = fnblas_backend_op_sv_div, \
    .op_sv_max = fnblas_backend_op_sv_max, \
    .op_sv_min = fnblas_backend_op_sv_min, \
    .op_ve_add = fnblas_backend_op_ve_add, \
    .op_ve_sub = fnblas_backend_op_ve_sub, \
    .op_ve_mul = fnblas_backend_op_ve_mul, \
    .op_ve_div = fnblas_backend_op_ve_div, \
    .op_ve_max = fnblas_backend_op_ve_max, \
    .op_ve_min = fnblas_backend_op_ve_min, \
    .op_vv_dot = fnblas_backend_op_vv_dot, \
    .op_vr_sum = fnblas_backend_op_vr_sum, \
    .op_vr_mean = fnblas_backend_op_vr_mean, \
    .op_vr_max = fnblas_backend_op_vr_max, \
    .op_vr_min = fnblas_backend_op_vr_min, \
    .op_vi_exp = fnblas_backend_op_vi_exp, \
    .op_mi_transpose = fnblas_backend_op_mi_transpose, \
    .op_me_add = fnblas_backend_op_me_add, \
    .op_me_sub = fnblas_backend_op_me_sub, \
    .op_me_mul = fnblas_backend_op_me_mul, \
    .op_me_div = fnblas_backend_op_me_div, \
    .op_mm_matmul = fnblas_backend_op_mm_matmul, \
    .op_mmv_axpy = fnblas_backend_op_mmv_axpy, \
    .op_mmm_axpy = fnblas_backend_op_mmm_axpy \
}

const flexi_tensor_backend_t flexi_tensor_fnblas_backend = FLEXI_TENSOR_FNBLAS_BACKEND_INITIALIZER;
flexi_tensor_backend_t _flexi_tensor_current_backend = FLEXI_TENSOR_FNBLAS_BACKEND_INITIALIZER;

flexi_tensor_error_t flexi_tensor_use_backend(const flexi_tensor_backend_t* backend) {
    if (backend == NULL) {
        return FLEXI_TENSOR_ERR_UNKNOWN;
    }

    _flexi_tensor_current_backend = flexi_tensor_fnblas_backend;

    if (backend->op_sv_add != NULL) _flexi_tensor_current_backend.op_sv_add = backend->op_sv_add;
    if (backend->op_sv_sub != NULL) _flexi_tensor_current_backend.op_sv_sub = backend->op_sv_sub;
    if (backend->op_sv_mul != NULL) _flexi_tensor_current_backend.op_sv_mul = backend->op_sv_mul;
    if (backend->op_sv_div != NULL) _flexi_tensor_current_backend.op_sv_div = backend->op_sv_div;
    if (backend->op_sv_max != NULL) _flexi_tensor_current_backend.op_sv_max = backend->op_sv_max;
    if (backend->op_sv_min != NULL) _flexi_tensor_current_backend.op_sv_min = backend->op_sv_min;
    if (backend->op_ve_add != NULL) _flexi_tensor_current_backend.op_ve_add = backend->op_ve_add;
    if (backend->op_ve_sub != NULL) _flexi_tensor_current_backend.op_ve_sub = backend->op_ve_sub;
    if (backend->op_ve_mul != NULL) _flexi_tensor_current_backend.op_ve_mul = backend->op_ve_mul;
    if (backend->op_ve_div != NULL) _flexi_tensor_current_backend.op_ve_div = backend->op_ve_div;
    if (backend->op_ve_max != NULL) _flexi_tensor_current_backend.op_ve_max = backend->op_ve_max;
    if (backend->op_ve_min != NULL) _flexi_tensor_current_backend.op_ve_min = backend->op_ve_min;
    if (backend->op_vv_dot != NULL) _flexi_tensor_current_backend.op_vv_dot = backend->op_vv_dot;
    if (backend->op_vr_sum != NULL) _flexi_tensor_current_backend.op_vr_sum = backend->op_vr_sum;
    if (backend->op_vr_mean != NULL) _flexi_tensor_current_backend.op_vr_mean = backend->op_vr_mean;
    if (backend->op_vr_max != NULL) _flexi_tensor_current_backend.op_vr_max = backend->op_vr_max;
    if (backend->op_vr_min != NULL) _flexi_tensor_current_backend.op_vr_min = backend->op_vr_min;
    if (backend->op_vi_exp != NULL) _flexi_tensor_current_backend.op_vi_exp = backend->op_vi_exp;
    if (backend->op_mi_transpose != NULL) _flexi_tensor_current_backend.op_mi_transpose = backend->op_mi_transpose;
    if (backend->op_me_add != NULL) _flexi_tensor_current_backend.op_me_add = backend->op_me_add;
    if (backend->op_me_sub != NULL) _flexi_tensor_current_backend.op_me_sub = backend->op_me_sub;
    if (backend->op_me_mul != NULL) _flexi_tensor_current_backend.op_me_mul = backend->op_me_mul;
    if (backend->op_me_div != NULL) _flexi_tensor_current_backend.op_me_div = backend->op_me_div;
    if (backend->op_mm_matmul != NULL) _flexi_tensor_current_backend.op_mm_matmul = backend->op_mm_matmul;
    if (backend->op_mmv_axpy != NULL) _flexi_tensor_current_backend.op_mmv_axpy = backend->op_mmv_axpy;
    if (backend->op_mmm_axpy != NULL) _flexi_tensor_current_backend.op_mmm_axpy = backend->op_mmm_axpy;

    return FLEXI_TENSOR_SUCCESS;
}
