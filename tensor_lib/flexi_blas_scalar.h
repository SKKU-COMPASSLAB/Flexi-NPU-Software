#ifndef FLEXI_BLAS_SCALAR_H
#define FLEXI_BLAS_SCALAR_H

#include <stddef.h>
#include <stdint.h>
#include "flexi_blas_common.h"

#ifdef __cplusplus
extern "C" {
#endif

fnblas_error_t  fnblas_scalar_create_from_dtype(fnblas_scalar_t* scalar, fnblas_dtype_t dtype);
fnblas_error_t  fnblas_scalar_create_float(fnblas_scalar_t* scalar, float val, fnblas_dtype_t dtype);
fnblas_error_t  fnblas_scalar_create_int(fnblas_scalar_t* scalar, int32_t val, fnblas_dtype_t dtype);
float           fnblas_scalar_as_unpacked_float(const fnblas_scalar_t* scalar);
int32_t         fnblas_scalar_as_unpacked_int32(const fnblas_scalar_t* scalar);
fnblas_scalar_t        fnblas_scalar_cast_to(const fnblas_scalar_t* scalar, fnblas_dtype_t new_dtype);
size_t          fnblas_scalar_size(const fnblas_scalar_t* scalar);
fnblas_dtype_t  fnblas_scalar_dtype(const fnblas_scalar_t* scalar);

fnblas_error_t  fnblas_op_ss_add(fnblas_scalar_t* result, const fnblas_scalar_t* lhs, const fnblas_scalar_t* rhs);  // scalar addition
fnblas_error_t  fnblas_op_ss_sub(fnblas_scalar_t* result, const fnblas_scalar_t* lhs, const fnblas_scalar_t* rhs);  // scalar subtraction
fnblas_error_t  fnblas_op_ss_mul(fnblas_scalar_t* result, const fnblas_scalar_t* lhs, const fnblas_scalar_t* rhs);  // scalar multiplication
fnblas_error_t  fnblas_op_ss_div(fnblas_scalar_t* result, const fnblas_scalar_t* lhs, const fnblas_scalar_t* rhs);  // scalar division

#ifdef __cplusplus
}
#endif

#endif // FLEXI_BLAS_SCALAR_H
