#ifndef FLEXI_BLAS_VECTOR_H
#define FLEXI_BLAS_VECTOR_H

#include <stddef.h>
#include <stdint.h>
#include "flexi_blas_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initialize every vector object with FNBLAS_VECTOR_INITIALIZER before use.
fnblas_error_t  fnblas_vector_create(fnblas_vector_t* vector, size_t n_elements, fnblas_dtype_t dtype);
fnblas_error_t  fnblas_vector_create_view(fnblas_vector_t* view, byte_t* buffer, size_t n_elements, fnblas_dtype_t dtype);
fnblas_error_t  fnblas_vector_destroy(fnblas_vector_t* vector);
fnblas_error_t  fnblas_vector_at(const fnblas_vector_t* vector, size_t index, byte_t* out_element);
fnblas_error_t  fnblas_vector_set(fnblas_vector_t* vector, size_t index, const byte_t* element);
fnblas_error_t  fnblas_vector_get_packed_buffer(const fnblas_vector_t* vector, byte_t* out_buffer);
fnblas_error_t  fnblas_vector_initialize_from_packed_buffer(fnblas_vector_t* vector, const byte_t* packed_buffer);
fnblas_error_t  fnblas_vector_initialize_from_unpacked_buffer(fnblas_vector_t* vector, const byte_t* unpacked_buffer);
size_t          fnblas_vector_n_elements(const fnblas_vector_t* vector);
fnblas_dtype_t  fnblas_vector_dtype(const fnblas_vector_t* vector);
size_t          fnblas_vector_packed_size(const fnblas_vector_t* vector);
size_t          fnblas_vector_unpacked_size(const fnblas_vector_t* vector);

// Operators: Scalar-Vector (SV)
fnblas_error_t  fnblas_op_sv_add(fnblas_vector_t* result, const fnblas_vector_t* lhs, const fnblas_scalar_t* rhs);  // vector addition with scalar
fnblas_error_t  fnblas_op_sv_sub(fnblas_vector_t* result, const fnblas_vector_t* lhs, const fnblas_scalar_t* rhs);  // vector subtraction with scalar
fnblas_error_t  fnblas_op_sv_mul(fnblas_vector_t* result, const fnblas_vector_t* lhs, const fnblas_scalar_t* rhs);  // vector multiplication with scalar
fnblas_error_t  fnblas_op_sv_div(fnblas_vector_t* result, const fnblas_vector_t* lhs, const fnblas_scalar_t* rhs);  // vector division with scalar
fnblas_error_t  fnblas_op_sv_max(fnblas_vector_t* result, const fnblas_vector_t* lhs, const fnblas_scalar_t* rhs);  // elementwise max between vector and scalar
fnblas_error_t  fnblas_op_sv_min(fnblas_vector_t* result, const fnblas_vector_t* lhs, const fnblas_scalar_t* rhs);  // elementwise min between vector and scalar

// Operators: Vector Elementwise (VE)
fnblas_error_t  fnblas_op_ve_add(fnblas_vector_t* result, const fnblas_vector_t* lhs, const fnblas_vector_t* rhs);  // elementwise vector addition
fnblas_error_t  fnblas_op_ve_sub(fnblas_vector_t* result, const fnblas_vector_t* lhs, const fnblas_vector_t* rhs);  // elementwise vector subtraction
fnblas_error_t  fnblas_op_ve_mul(fnblas_vector_t* result, const fnblas_vector_t* lhs, const fnblas_vector_t* rhs);  // elementwise vector multiplication
fnblas_error_t  fnblas_op_ve_div(fnblas_vector_t* result, const fnblas_vector_t* lhs, const fnblas_vector_t* rhs);  // elementwise vector division
fnblas_error_t  fnblas_op_ve_max(fnblas_vector_t* result, const fnblas_vector_t* lhs, const fnblas_vector_t* rhs);  // elementwise max between two vectors
fnblas_error_t  fnblas_op_ve_min(fnblas_vector_t* result, const fnblas_vector_t* lhs, const fnblas_vector_t* rhs);  // elementwise min between two vectors

// Operators: Vector-Vector (VV)
fnblas_error_t  fnblas_op_vv_dot(fnblas_scalar_t* result, const fnblas_vector_t* lhs, const fnblas_vector_t* rhs);  // vector dot product

// Operators: Vector-Recuction (VR)
fnblas_error_t  fnblas_op_vr_sum(fnblas_scalar_t* result, const fnblas_vector_t* input);  // sum of all elements in the vector
fnblas_error_t  fnblas_op_vr_mean(fnblas_scalar_t* result, const fnblas_vector_t* input);  // mean of all elements in the vector
fnblas_error_t  fnblas_op_vr_max(fnblas_scalar_t* result, const fnblas_vector_t* input);  // max of all elements in the vector
fnblas_error_t  fnblas_op_vr_min(fnblas_scalar_t* result, const fnblas_vector_t* input);  // min of all elements in the vector

// Operators: Vector-Individual (VI)
fnblas_error_t  fnblas_op_vi_exp(fnblas_vector_t* result, const fnblas_vector_t* input);  // elementwise exponential of the vector

// Operators: Vector-Cast (VC)
fnblas_error_t  fnblas_op_vc_cast(fnblas_vector_t* result, const fnblas_vector_t* input, fnblas_dtype_t dtype);

// Operators: Vector-Quantization (VQ)
fnblas_error_t  fnblas_op_vq_quant_per_tensor(fnblas_vector_t* result, const fnblas_vector_t* input, const fnblas_scalar_t* scale, const fnblas_scalar_t* zero_point, fnblas_dtype_t qdtype);
fnblas_error_t  fnblas_op_vq_dequant_per_tensor(fnblas_vector_t* result, const fnblas_vector_t* input, const fnblas_scalar_t* scale, const fnblas_scalar_t* zero_point, fnblas_dtype_t dtype);

#ifdef __cplusplus
}
#endif

#endif // FLEXI_BLAS_VECTOR_H
