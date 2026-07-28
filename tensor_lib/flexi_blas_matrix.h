#ifndef FLEXI_BLAS_MATRIX_H
#define FLEXI_BLAS_MATRIX_H

#include <stddef.h>
#include <stdint.h>
#include "flexi_blas_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initialize every matrix object with FNBLAS_MATRIX_INITIALIZER before use.
fnblas_error_t  fnblas_matrix_create(fnblas_matrix_t* matrix, size_t n_rows, size_t n_cols, fnblas_dtype_t dtype);
fnblas_error_t  fnblas_matrix_create_view(fnblas_matrix_t* view, byte_t* buffer, size_t n_rows, size_t n_cols, fnblas_dtype_t dtype);
fnblas_error_t  fnblas_matrix_create_reshape_view(fnblas_matrix_t* view, fnblas_matrix_t* source, size_t n_rows, size_t n_cols);
fnblas_error_t  fnblas_matrix_create_row_vector_view(fnblas_vector_t* view, fnblas_matrix_t* matrix, size_t row);
fnblas_error_t  fnblas_matrix_destroy(fnblas_matrix_t* matrix);
fnblas_error_t  fnblas_matrix_at(const fnblas_matrix_t* matrix, size_t row, size_t col, byte_t* out_element);
fnblas_error_t  fnblas_matrix_set(fnblas_matrix_t* matrix, size_t row, size_t col, const byte_t* element);
fnblas_error_t  fnblas_matrix_get_packed_buffer(const fnblas_matrix_t* matrix, byte_t* out_buffer);
fnblas_error_t  fnblas_matrix_initialize_from_packed_buffer(fnblas_matrix_t* matrix, const byte_t* packed_buffer);
fnblas_error_t  fnblas_matrix_initialize_from_unpacked_buffer(fnblas_matrix_t* matrix, const byte_t* unpacked_buffer);
size_t          fnblas_matrix_n_rows(const fnblas_matrix_t* matrix);
size_t          fnblas_matrix_n_cols(const fnblas_matrix_t* matrix);
fnblas_dtype_t  fnblas_matrix_dtype(const fnblas_matrix_t* matrix);
size_t          fnblas_matrix_packed_buffer_size(const fnblas_matrix_t* matrix);
size_t          fnblas_matrix_unpacked_buffer_size(const fnblas_matrix_t* matrix);

// Operators: Matrix Individual (MI)
fnblas_error_t  fnblas_op_mi_transpose(fnblas_matrix_t* result, const fnblas_matrix_t* input);   //  matrix transpose

// Operators: Matrix Elementwise (ME)
fnblas_error_t  fnblas_op_me_add(fnblas_matrix_t* result, const fnblas_matrix_t* lhs, const fnblas_matrix_t* rhs);  // elementwise matrix addition
fnblas_error_t  fnblas_op_me_sub(fnblas_matrix_t* result, const fnblas_matrix_t* lhs, const fnblas_matrix_t* rhs);  // elementwise matrix subtraction
fnblas_error_t  fnblas_op_me_mul(fnblas_matrix_t* result, const fnblas_matrix_t* lhs, const fnblas_matrix_t* rhs);  // elementwise matrix multiplication
fnblas_error_t  fnblas_op_me_div(fnblas_matrix_t* result, const fnblas_matrix_t* lhs, const fnblas_matrix_t* rhs);  // elementwise matrix division

// Operators: Matrix-Matrix (MM)
fnblas_error_t  fnblas_op_mm_matmul(fnblas_matrix_t* result, const fnblas_matrix_t* lhs, const fnblas_matrix_t* rhs_t);  // matrix multiplication (rhs_t is the transpose of the right-hand side matrix)

// Operators: Matrix-Matrix-Vector (MMV)
fnblas_error_t  fnblas_op_mmv_axpy(fnblas_matrix_t* result, const fnblas_matrix_t* a, const fnblas_matrix_t* x_t, const fnblas_vector_t* y);  // matrix-vector multiplication with addition (y is added to each row of the result)

// Operators: Matrix-Matrix-Matrix (MMM)
fnblas_error_t  fnblas_op_mmm_axpy(fnblas_matrix_t* result, const fnblas_matrix_t* a, const fnblas_matrix_t* x_t, const fnblas_matrix_t* y);  // matrix-vector multiplication with addition (y is added to the result)

#ifdef __cplusplus
}
#endif

#endif // FLEXI_BLAS_MATRIX_H
