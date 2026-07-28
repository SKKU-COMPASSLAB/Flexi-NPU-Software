#include "flexi_blas_internal.h"

#include <errno.h>
#include <math.h>
#include <string.h>

fnblas_error_t fnblas_scalar_create_from_dtype(
    fnblas_scalar_t* scalar, fnblas_dtype_t dtype)
{
    if (scalar == NULL)
        return FNBLAS_ERR_UNKNOWN;
    if (!_fnblas_dtype_is_valid(dtype))
        return FNBLAS_ERR_MISMATCH;
    memset(scalar, 0, sizeof(*scalar));
    scalar->_dtype = dtype;
    return FNBLAS_SUCCESS;
}

fnblas_error_t fnblas_scalar_create_float(
    fnblas_scalar_t* scalar, float value, fnblas_dtype_t dtype)
{
    if (scalar == NULL)
        return FNBLAS_ERR_UNKNOWN;
    if (!_fnblas_dtype_is_float(dtype))
        return FNBLAS_ERR_MISMATCH;
    scalar->_value.f32 = value;
    scalar->_dtype = dtype;
    return FNBLAS_SUCCESS;
}

fnblas_error_t fnblas_scalar_create_int(
    fnblas_scalar_t* scalar, int32_t value, fnblas_dtype_t dtype)
{
    if (scalar == NULL)
        return FNBLAS_ERR_UNKNOWN;
    if (!_fnblas_dtype_is_integer(dtype))
        return FNBLAS_ERR_MISMATCH;
    if (dtype == INT8)
        scalar->_value.u8 = (uint8_t)value;
    else
        scalar->_value.i32 = value;
    scalar->_dtype = dtype;
    return FNBLAS_SUCCESS;
}

float fnblas_scalar_as_unpacked_float(const fnblas_scalar_t* scalar)
{
    if (scalar == NULL || !_fnblas_dtype_is_float(scalar->_dtype)) {
        errno = EINVAL;
        return 0.0f;
    }
    return scalar->_value.f32;
}

int32_t fnblas_scalar_as_unpacked_int32(const fnblas_scalar_t* scalar)
{
    if (scalar == NULL || !_fnblas_dtype_is_integer(scalar->_dtype)) {
        errno = EINVAL;
        return 0;
    }
    return scalar->_dtype == INT8
        ? (int32_t)scalar->_value.u8
        : scalar->_value.i32;
}

fnblas_scalar_t fnblas_scalar_cast_to(const fnblas_scalar_t* scalar, fnblas_dtype_t new_dtype)
{
    fnblas_scalar_t result;
    double value = 0.0;
    memset(&result, 0, sizeof(result));
    result._dtype = new_dtype;
    if (scalar == NULL ||
        !_fnblas_dtype_is_valid(scalar->_dtype) ||
        !_fnblas_dtype_is_valid(new_dtype)) {
        errno = EINVAL;
        return result;
    }
    value = _fnblas_dtype_is_float(scalar->_dtype)
        ? (double)scalar->_value.f32
        : (double)fnblas_scalar_as_unpacked_int32(scalar);
    if (_fnblas_dtype_is_float(new_dtype)) {
        result._value.f32 = (float)value;
    } else if (!isfinite(value) ||
               value < (double)INT32_MIN ||
               value > (double)INT32_MAX) {
        errno = ERANGE;
    } else if (new_dtype == INT8) {
        if (value < 0.0 || value > (double)UINT8_MAX)
            errno = ERANGE;
        else
            result._value.u8 = (uint8_t)value;
    } else {
        result._value.i32 = (int32_t)value;
    }
    return result;
}

size_t fnblas_scalar_size(const fnblas_scalar_t* scalar)
{
    if (scalar == NULL) {
        errno = EINVAL;
        return 0;
    }
    return fnblas_dtype_size_of(scalar->_dtype);
}

fnblas_dtype_t fnblas_scalar_dtype(const fnblas_scalar_t* scalar)
{
    if (scalar == NULL) {
        errno = EINVAL;
        return INT8;
    }
    return scalar->_dtype;
}

static fnblas_error_t fnblas_scalar_operation(
    fnblas_scalar_t* result,
    const fnblas_scalar_t* lhs,
    const fnblas_scalar_t* rhs,
    fnblas_arithmetic_op_t operation)
{
    fnblas_error_t error;
    if (result == NULL || lhs == NULL || rhs == NULL)
        return FNBLAS_ERR_UNKNOWN;
    if (lhs->_dtype != rhs->_dtype)
        return FNBLAS_ERR_MISMATCH;
    if (operation == DIVIDE && !_fnblas_dtype_is_float(lhs->_dtype) &&
        ((lhs->_dtype == INT8 && rhs->_value.u8 == 0) ||
         (lhs->_dtype != INT8 && rhs->_value.i32 == 0)))
        return FNBLAS_ERR_UNKNOWN;
    error = fnblas_scalar_create_from_dtype(result, lhs->_dtype);
    if (error != FNBLAS_SUCCESS)
        return error;
    if (_fnblas_dtype_is_float(lhs->_dtype)) {
        result->_value.f32 = _fnblas_float_arithmetic(
            lhs->_value.f32, rhs->_value.f32, operation
        );
    } else if (lhs->_dtype == INT8) {
        result->_value.u8 = _fnblas_uint8_arithmetic(
            lhs->_value.u8, rhs->_value.u8, operation
        );
    } else {
        result->_value.i32 = _fnblas_int32_arithmetic(
            lhs->_value.i32, rhs->_value.i32, operation
        );
    }
    return FNBLAS_SUCCESS;
}

fnblas_error_t fnblas_op_ss_add(
    fnblas_scalar_t* result, const fnblas_scalar_t* lhs, const fnblas_scalar_t* rhs)
{
    return fnblas_scalar_operation(result, lhs, rhs, ADD);
}

fnblas_error_t fnblas_op_ss_sub(
    fnblas_scalar_t* result, const fnblas_scalar_t* lhs, const fnblas_scalar_t* rhs)
{
    return fnblas_scalar_operation(result, lhs, rhs, SUBTRACT);
}

fnblas_error_t fnblas_op_ss_mul(
    fnblas_scalar_t* result, const fnblas_scalar_t* lhs, const fnblas_scalar_t* rhs)
{
    return fnblas_scalar_operation(result, lhs, rhs, MULTIPLY);
}

fnblas_error_t fnblas_op_ss_div(
    fnblas_scalar_t* result, const fnblas_scalar_t* lhs, const fnblas_scalar_t* rhs)
{
    return fnblas_scalar_operation(result, lhs, rhs, DIVIDE);
}
