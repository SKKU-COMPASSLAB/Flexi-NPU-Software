#ifndef FLEXI_TENSOR_BACKEND_H
#define FLEXI_TENSOR_BACKEND_H

#include <stddef.h>
#include <stdint.h>

#include "flexi_blas.h"
#include "flexi_tensor_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    // operator scalar-vector
    flexi_tensor_error_t (*op_sv_add)(fnblas_vector_t*, const fnblas_vector_t*, const fnblas_scalar_t*);
    flexi_tensor_error_t (*op_sv_sub)(fnblas_vector_t*, const fnblas_vector_t*, const fnblas_scalar_t*);
    flexi_tensor_error_t (*op_sv_mul)(fnblas_vector_t*, const fnblas_vector_t*, const fnblas_scalar_t*);
    flexi_tensor_error_t (*op_sv_div)(fnblas_vector_t*, const fnblas_vector_t*, const fnblas_scalar_t*);
    flexi_tensor_error_t (*op_sv_max)(fnblas_vector_t*, const fnblas_vector_t*, const fnblas_scalar_t*);
    flexi_tensor_error_t (*op_sv_min)(fnblas_vector_t*, const fnblas_vector_t*, const fnblas_scalar_t*);

    // operator vector-elementwise
    flexi_tensor_error_t (*op_ve_add)(fnblas_vector_t*, const fnblas_vector_t*, const fnblas_vector_t*);
    flexi_tensor_error_t (*op_ve_sub)(fnblas_vector_t*, const fnblas_vector_t*, const fnblas_vector_t*);
    flexi_tensor_error_t (*op_ve_mul)(fnblas_vector_t*, const fnblas_vector_t*, const fnblas_vector_t*);
    flexi_tensor_error_t (*op_ve_div)(fnblas_vector_t*, const fnblas_vector_t*, const fnblas_vector_t*);
    flexi_tensor_error_t (*op_ve_max)(fnblas_vector_t*, const fnblas_vector_t*, const fnblas_vector_t*);
    flexi_tensor_error_t (*op_ve_min)(fnblas_vector_t*, const fnblas_vector_t*, const fnblas_vector_t*);

    // operator vector-vector
    flexi_tensor_error_t (*op_vv_dot)(fnblas_scalar_t*, const fnblas_vector_t*, const fnblas_vector_t*);

    // operator vector-reduction
    flexi_tensor_error_t (*op_vr_sum)(fnblas_scalar_t*, const fnblas_vector_t*);
    flexi_tensor_error_t (*op_vr_mean)(fnblas_scalar_t*, const fnblas_vector_t*);
    flexi_tensor_error_t (*op_vr_max)(fnblas_scalar_t*, const fnblas_vector_t*);
    flexi_tensor_error_t (*op_vr_min)(fnblas_scalar_t*, const fnblas_vector_t*);

    // operator vector-individual
    flexi_tensor_error_t (*op_vi_exp)(fnblas_vector_t*, const fnblas_vector_t*);

    // operator matrix-individual
    flexi_tensor_error_t (*op_mi_transpose)(fnblas_matrix_t*, const fnblas_matrix_t*);

    // operator matrix-elementwise
    flexi_tensor_error_t (*op_me_add)(fnblas_matrix_t*, const fnblas_matrix_t*, const fnblas_matrix_t*);
    flexi_tensor_error_t (*op_me_sub)(fnblas_matrix_t*, const fnblas_matrix_t*, const fnblas_matrix_t*);
    flexi_tensor_error_t (*op_me_mul)(fnblas_matrix_t*, const fnblas_matrix_t*, const fnblas_matrix_t*);
    flexi_tensor_error_t (*op_me_div)(fnblas_matrix_t*, const fnblas_matrix_t*, const fnblas_matrix_t*);

    // operator matrix-matrix
    flexi_tensor_error_t (*op_mm_matmul)(fnblas_matrix_t*, const fnblas_matrix_t*, const fnblas_matrix_t*);

    // operator matrix-matrix-vector
    flexi_tensor_error_t (*op_mmv_axpy)(fnblas_matrix_t*, const fnblas_matrix_t*, const fnblas_matrix_t*, const fnblas_vector_t*);

    // operator matrix-matrix-matrix
    flexi_tensor_error_t (*op_mmm_axpy)(fnblas_matrix_t*, const fnblas_matrix_t*, const fnblas_matrix_t*, const fnblas_matrix_t*);
} flexi_tensor_backend_t;

extern const flexi_tensor_backend_t flexi_tensor_fnblas_backend;

flexi_tensor_error_t flexi_tensor_use_backend(const flexi_tensor_backend_t* backend);

#ifdef __cplusplus
}
#endif

#endif // FLEXI_TENSOR_BACKEND_H
