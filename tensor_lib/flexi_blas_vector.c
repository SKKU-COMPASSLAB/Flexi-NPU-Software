#include "flexi_blas_internal.h"

#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

fnblas_error_t fnblas_vector_create(
    fnblas_vector_t* vector, size_t n_elements, fnblas_dtype_t dtype)
{
    fnblas_error_t error;
    byte_t* new_buffer = NULL;
    if (vector == NULL)
        return FNBLAS_ERR_UNKNOWN;
    if (!_fnblas_dtype_is_valid(dtype))
        return FNBLAS_ERR_MISMATCH;
    if (_fnblas_buffer_is_allocated(vector->_status) || _fnblas_buffer_is_view(vector->_status))
        return FNBLAS_ERR_UNKNOWN;
    error = _fnblas_allocate_buffer(&new_buffer, n_elements, dtype);
    if (error != FNBLAS_SUCCESS)
        return error;
    vector->_buffer = new_buffer;
    vector->_n_elements = n_elements;
    vector->_dtype = dtype;
    vector->_status = 0;
    if (new_buffer != NULL)
        vector->_status |= FNBLAS_STATUS_BUFFER_ALLOCATED;
    return FNBLAS_SUCCESS;
}

fnblas_error_t fnblas_vector_create_view(fnblas_vector_t* view, byte_t* buffer, size_t n_elements, fnblas_dtype_t dtype)
{
    if (view == NULL)
        return FNBLAS_ERR_UNKNOWN;
    if (!_fnblas_dtype_is_valid(dtype))
        return FNBLAS_ERR_MISMATCH;
    if (_fnblas_buffer_is_allocated(view->_status) || _fnblas_buffer_is_view(view->_status))
        return FNBLAS_ERR_UNKNOWN;
    if (n_elements != 0 && buffer == NULL)
        return FNBLAS_ERR_MISMATCH;
    view->_buffer = buffer;
    view->_n_elements = n_elements;
    view->_dtype = dtype;
    view->_status = FNBLAS_STATUS_VIEW;
    if (buffer != NULL)
        view->_status |= FNBLAS_STATUS_BUFFER_ALLOCATED;
    return FNBLAS_SUCCESS;
}

fnblas_error_t fnblas_vector_destroy(fnblas_vector_t* vector)
{
    if (vector == NULL)
        return FNBLAS_ERR_UNKNOWN;
    if (_fnblas_buffer_is_allocated(vector->_status) && !_fnblas_buffer_is_view(vector->_status))
        _fnblas_deallocate_buffer(vector->_buffer);
    vector->_buffer = NULL;
    vector->_n_elements = 0;
    vector->_dtype = FP32;
    vector->_status = 0;
    return FNBLAS_SUCCESS;
}

fnblas_error_t fnblas_vector_at(
    const fnblas_vector_t* vector, size_t index, byte_t* out_element)
{
    size_t element_size;
    if (vector == NULL || out_element == NULL ||
        index >= vector->_n_elements) {
        return FNBLAS_ERR_UNKNOWN;
    }
    element_size = _fnblas_dtype_unpacked_size_of(vector->_dtype);
    memcpy(
        out_element,
        vector->_buffer + index * element_size,
        element_size
    );
    return FNBLAS_SUCCESS;
}

fnblas_error_t fnblas_vector_set(
    fnblas_vector_t* vector, size_t index, const byte_t* element)
{
    size_t element_size;
    if (vector == NULL || element == NULL ||
        index >= vector->_n_elements) {
        return FNBLAS_ERR_UNKNOWN;
    }
    element_size = _fnblas_dtype_unpacked_size_of(vector->_dtype);
    memcpy(
        vector->_buffer + index * element_size,
        element,
        element_size
    );
    return FNBLAS_SUCCESS;
}

fnblas_error_t fnblas_vector_get_packed_buffer(
    const fnblas_vector_t* vector, byte_t* out_buffer)
{
    if (vector == NULL ||
        (vector->_n_elements != 0 && out_buffer == NULL)) {
        return FNBLAS_ERR_UNKNOWN;
    }
    _fnblas_pack_buffer(
        vector->_buffer,
        vector->_n_elements,
        vector->_dtype,
        out_buffer
    );
    return FNBLAS_SUCCESS;
}

fnblas_error_t fnblas_vector_initialize_from_packed_buffer(
    fnblas_vector_t* vector, const byte_t* packed_buffer)
{
    if (vector == NULL ||
        (vector->_n_elements != 0 && packed_buffer == NULL)) {
        return FNBLAS_ERR_UNKNOWN;
    }
    _fnblas_unpack_buffer(
        vector->_buffer,
        vector->_n_elements,
        vector->_dtype,
        packed_buffer
    );
    return FNBLAS_SUCCESS;
}

fnblas_error_t fnblas_vector_initialize_from_unpacked_buffer(
    fnblas_vector_t* vector, const byte_t* unpacked_buffer)
{
    size_t size;
    if (vector == NULL ||
        (vector->_n_elements != 0 && unpacked_buffer == NULL)) {
        return FNBLAS_ERR_UNKNOWN;
    }
    size = fnblas_vector_unpacked_size(vector);
    if (size != 0)
        memcpy(vector->_buffer, unpacked_buffer, size);
    return FNBLAS_SUCCESS;
}

size_t fnblas_vector_n_elements(const fnblas_vector_t* vector)
{
    if (vector == NULL) {
        errno = EINVAL;
        return 0;
    }
    return vector->_n_elements;
}

fnblas_dtype_t fnblas_vector_dtype(const fnblas_vector_t* vector)
{
    if (vector == NULL) {
        errno = EINVAL;
        return INT8;
    }
    return vector->_dtype;
}

size_t fnblas_vector_packed_size(const fnblas_vector_t* vector)
{
    if (vector == NULL) {
        errno = EINVAL;
        return 0;
    }
    return _fnblas_packed_buffer_size(
        vector->_n_elements, vector->_dtype
    );
}

size_t fnblas_vector_unpacked_size(const fnblas_vector_t* vector)
{
    size_t element_size;
    if (vector == NULL) {
        errno = EINVAL;
        return 0;
    }
    element_size = _fnblas_dtype_unpacked_size_of(vector->_dtype);
    if (vector->_n_elements > SIZE_MAX / element_size) {
        errno = EOVERFLOW;
        return 0;
    }
    return vector->_n_elements * element_size;
}

static fnblas_error_t fnblas_validate_sv(
    fnblas_vector_t* result, const fnblas_vector_t* lhs, const fnblas_scalar_t* rhs)
{
    if (result == NULL || lhs == NULL || rhs == NULL || result == lhs)
        return FNBLAS_ERR_UNKNOWN;
    if (lhs->_dtype != rhs->_dtype)
        return FNBLAS_ERR_MISMATCH;
    if (_fnblas_buffer_is_allocated(result->_status)) {
        if (result->_dtype != lhs->_dtype ||
            result->_n_elements != lhs->_n_elements) {
            return FNBLAS_ERR_MISMATCH;
        }
        return FNBLAS_SUCCESS;
    }
    return fnblas_vector_create(
        result, lhs->_n_elements, lhs->_dtype
    );
}

static fnblas_error_t fnblas_validate_ve(
    fnblas_vector_t* result, const fnblas_vector_t* lhs, const fnblas_vector_t* rhs)
{
    if (result == NULL || lhs == NULL || rhs == NULL ||
        result == lhs || result == rhs)
        return FNBLAS_ERR_UNKNOWN;
    if (lhs->_dtype != rhs->_dtype ||
        lhs->_n_elements != rhs->_n_elements)
        return FNBLAS_ERR_MISMATCH;
    if (_fnblas_buffer_is_allocated(result->_status)) {
        if (result->_dtype != lhs->_dtype ||
            result->_n_elements != lhs->_n_elements) {
            return FNBLAS_ERR_MISMATCH;
        }
        return FNBLAS_SUCCESS;
    }
    return fnblas_vector_create(
        result, lhs->_n_elements, lhs->_dtype
    );
}

static fnblas_error_t fnblas_validate_vv(
    const fnblas_vector_t* lhs, const fnblas_vector_t* rhs)
{
    if (lhs == NULL || rhs == NULL)
        return FNBLAS_ERR_UNKNOWN;
    if (lhs->_dtype != rhs->_dtype ||
        lhs->_n_elements != rhs->_n_elements)
        return FNBLAS_ERR_MISMATCH;
    return FNBLAS_SUCCESS;
}

fnblas_error_t fnblas_op_sv_add(
    fnblas_vector_t* result, const fnblas_vector_t* lhs, const fnblas_scalar_t* rhs)
{
    fnblas_error_t error;
    size_t index;
    error = fnblas_validate_sv(result, lhs, rhs);
    if (error != FNBLAS_SUCCESS)
        return error;
    if (_fnblas_dtype_is_float(lhs->_dtype)) {
        float* restrict dst = (float*)result->_buffer;
        const float* restrict src = (const float*)lhs->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = src[index] + rhs->_value.f32;
    } else if (lhs->_dtype == INT8) {
        int8_t* restrict dst = (int8_t*)result->_buffer;
        const int8_t* restrict src = (const int8_t*)lhs->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = _fnblas_int8_arithmetic(src[index], rhs->_value.i8, ADD);
    } else {
        int32_t* restrict dst = (int32_t*)result->_buffer;
        const int32_t* restrict src = (const int32_t*)lhs->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = _fnblas_int32_arithmetic(src[index], rhs->_value.i32, ADD);
    }
    return FNBLAS_SUCCESS;
}

fnblas_error_t fnblas_op_sv_sub(
    fnblas_vector_t* result, const fnblas_vector_t* lhs, const fnblas_scalar_t* rhs)
{
    fnblas_error_t error;
    size_t index;
    error = fnblas_validate_sv(result, lhs, rhs);
    if (error != FNBLAS_SUCCESS)
        return error;
    if (_fnblas_dtype_is_float(lhs->_dtype)) {
        float* restrict dst = (float*)result->_buffer;
        const float* restrict src = (const float*)lhs->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = src[index] - rhs->_value.f32;
    } else if (lhs->_dtype == INT8) {
        int8_t* restrict dst = (int8_t*)result->_buffer;
        const int8_t* restrict src = (const int8_t*)lhs->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = _fnblas_int8_arithmetic(src[index], rhs->_value.i8, SUBTRACT);
    } else {
        int32_t* restrict dst = (int32_t*)result->_buffer;
        const int32_t* restrict src = (const int32_t*)lhs->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = _fnblas_int32_arithmetic(src[index], rhs->_value.i32, SUBTRACT);
    }
    return FNBLAS_SUCCESS;
}

fnblas_error_t fnblas_op_sv_mul(
    fnblas_vector_t* result, const fnblas_vector_t* lhs, const fnblas_scalar_t* rhs)
{
    fnblas_error_t error;
    size_t index;
    error = fnblas_validate_sv(result, lhs, rhs);
    if (error != FNBLAS_SUCCESS)
        return error;
    if (_fnblas_dtype_is_float(lhs->_dtype)) {
        float* restrict dst = (float*)result->_buffer;
        const float* restrict src = (const float*)lhs->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = src[index] * rhs->_value.f32;
    } else if (lhs->_dtype == INT8) {
        int8_t* restrict dst = (int8_t*)result->_buffer;
        const int8_t* restrict src = (const int8_t*)lhs->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = _fnblas_int8_arithmetic(src[index], rhs->_value.i8, MULTIPLY);
    } else {
        int32_t* restrict dst = (int32_t*)result->_buffer;
        const int32_t* restrict src = (const int32_t*)lhs->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = _fnblas_int32_arithmetic(src[index], rhs->_value.i32, MULTIPLY);
    }
    return FNBLAS_SUCCESS;
}

fnblas_error_t fnblas_op_sv_div(
    fnblas_vector_t* result, const fnblas_vector_t* lhs, const fnblas_scalar_t* rhs)
{
    fnblas_error_t error;
    size_t index;
    error = fnblas_validate_sv(result, lhs, rhs);
    if (error != FNBLAS_SUCCESS)
        return error;
    if (_fnblas_dtype_is_float(lhs->_dtype)) {
        float* restrict dst = (float*)result->_buffer;
        const float* restrict src = (const float*)lhs->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = src[index] / rhs->_value.f32;
    } else if (lhs->_dtype == INT8) {
        int8_t* restrict dst = (int8_t*)result->_buffer;
        const int8_t* restrict src = (const int8_t*)lhs->_buffer;
        if (rhs->_value.i8 == 0)
            return FNBLAS_ERR_UNKNOWN;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = _fnblas_int8_arithmetic(src[index], rhs->_value.i8, DIVIDE);
    } else {
        int32_t* restrict dst = (int32_t*)result->_buffer;
        const int32_t* restrict src = (const int32_t*)lhs->_buffer;
        if (rhs->_value.i32 == 0)
            return FNBLAS_ERR_UNKNOWN;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = _fnblas_int32_arithmetic(src[index], rhs->_value.i32, DIVIDE);
    }
    return FNBLAS_SUCCESS;
}

fnblas_error_t fnblas_op_sv_max(fnblas_vector_t* result, const fnblas_vector_t* lhs, const fnblas_scalar_t* rhs)
{
    fnblas_error_t error;
    size_t index;
    error = fnblas_validate_sv(result, lhs, rhs);
    if (error != FNBLAS_SUCCESS)
        return error;
    if (_fnblas_dtype_is_float(lhs->_dtype)) {
        float* restrict dst = (float*)result->_buffer;
        const float* restrict src = (const float*)lhs->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = src[index] > rhs->_value.f32 ? src[index] : rhs->_value.f32;
    } else if (lhs->_dtype == INT8) {
        int8_t* restrict dst = (int8_t*)result->_buffer;
        const int8_t* restrict src = (const int8_t*)lhs->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = src[index] > rhs->_value.i8 ? src[index] : rhs->_value.i8;
    } else {
        int32_t* restrict dst = (int32_t*)result->_buffer;
        const int32_t* restrict src = (const int32_t*)lhs->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = src[index] > rhs->_value.i32 ? src[index] : rhs->_value.i32;
    }
    return FNBLAS_SUCCESS;
}

fnblas_error_t fnblas_op_sv_min(fnblas_vector_t* result, const fnblas_vector_t* lhs, const fnblas_scalar_t* rhs)
{
    fnblas_error_t error;
    size_t index;
    error = fnblas_validate_sv(result, lhs, rhs);
    if (error != FNBLAS_SUCCESS)
        return error;
    if (_fnblas_dtype_is_float(lhs->_dtype)) {
        float* restrict dst = (float*)result->_buffer;
        const float* restrict src = (const float*)lhs->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = src[index] < rhs->_value.f32 ? src[index] : rhs->_value.f32;
    } else if (lhs->_dtype == INT8) {
        int8_t* restrict dst = (int8_t*)result->_buffer;
        const int8_t* restrict src = (const int8_t*)lhs->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = src[index] < rhs->_value.i8 ? src[index] : rhs->_value.i8;
    } else {
        int32_t* restrict dst = (int32_t*)result->_buffer;
        const int32_t* restrict src = (const int32_t*)lhs->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = src[index] < rhs->_value.i32 ? src[index] : rhs->_value.i32;
    }
    return FNBLAS_SUCCESS;
}

fnblas_error_t fnblas_op_ve_add(
    fnblas_vector_t* result, const fnblas_vector_t* lhs, const fnblas_vector_t* rhs)
{
    fnblas_error_t error;
    size_t index;
    error = fnblas_validate_ve(result, lhs, rhs);
    if (error != FNBLAS_SUCCESS)
        return error;
    if (_fnblas_dtype_is_float(lhs->_dtype)) {
        float* restrict dst = (float*)result->_buffer;
        const float* restrict left = (const float*)lhs->_buffer;
        const float* restrict right = (const float*)rhs->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = left[index] + right[index];
    } else if (lhs->_dtype == INT8) {
        int8_t* restrict dst = (int8_t*)result->_buffer;
        const int8_t* restrict left = (const int8_t*)lhs->_buffer;
        const int8_t* restrict right = (const int8_t*)rhs->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = _fnblas_int8_arithmetic(left[index], right[index], ADD);
    } else {
        int32_t* restrict dst = (int32_t*)result->_buffer;
        const int32_t* restrict left = (const int32_t*)lhs->_buffer;
        const int32_t* restrict right = (const int32_t*)rhs->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = _fnblas_int32_arithmetic(left[index], right[index], ADD);
    }
    return FNBLAS_SUCCESS;
}

fnblas_error_t fnblas_op_ve_sub(
    fnblas_vector_t* result, const fnblas_vector_t* lhs, const fnblas_vector_t* rhs)
{
    fnblas_error_t error;
    size_t index;
    error = fnblas_validate_ve(result, lhs, rhs);
    if (error != FNBLAS_SUCCESS)
        return error;
    if (_fnblas_dtype_is_float(lhs->_dtype)) {
        float* restrict dst = (float*)result->_buffer;
        const float* restrict left = (const float*)lhs->_buffer;
        const float* restrict right = (const float*)rhs->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = left[index] - right[index];
    } else if (lhs->_dtype == INT8) {
        int8_t* restrict dst = (int8_t*)result->_buffer;
        const int8_t* restrict left = (const int8_t*)lhs->_buffer;
        const int8_t* restrict right = (const int8_t*)rhs->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = _fnblas_int8_arithmetic(left[index], right[index], SUBTRACT);
    } else {
        int32_t* restrict dst = (int32_t*)result->_buffer;
        const int32_t* restrict left = (const int32_t*)lhs->_buffer;
        const int32_t* restrict right = (const int32_t*)rhs->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = _fnblas_int32_arithmetic(left[index], right[index], SUBTRACT);
    }
    return FNBLAS_SUCCESS;
}

fnblas_error_t fnblas_op_ve_mul(
    fnblas_vector_t* result, const fnblas_vector_t* lhs, const fnblas_vector_t* rhs)
{
    fnblas_error_t error;
    size_t index;
    error = fnblas_validate_ve(result, lhs, rhs);
    if (error != FNBLAS_SUCCESS)
        return error;
    if (_fnblas_dtype_is_float(lhs->_dtype)) {
        float* restrict dst = (float*)result->_buffer;
        const float* restrict left = (const float*)lhs->_buffer;
        const float* restrict right = (const float*)rhs->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = left[index] * right[index];
    } else if (lhs->_dtype == INT8) {
        int8_t* restrict dst = (int8_t*)result->_buffer;
        const int8_t* restrict left = (const int8_t*)lhs->_buffer;
        const int8_t* restrict right = (const int8_t*)rhs->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = _fnblas_int8_arithmetic(left[index], right[index], MULTIPLY);
    } else {
        int32_t* restrict dst = (int32_t*)result->_buffer;
        const int32_t* restrict left = (const int32_t*)lhs->_buffer;
        const int32_t* restrict right = (const int32_t*)rhs->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = _fnblas_int32_arithmetic(left[index], right[index], MULTIPLY);
    }
    return FNBLAS_SUCCESS;
}

fnblas_error_t fnblas_op_ve_div(
    fnblas_vector_t* result, const fnblas_vector_t* lhs, const fnblas_vector_t* rhs)
{
    fnblas_error_t error;
    size_t index;
    error = fnblas_validate_ve(result, lhs, rhs);
    if (error != FNBLAS_SUCCESS)
        return error;
    if (_fnblas_dtype_is_float(lhs->_dtype)) {
        float* restrict dst = (float*)result->_buffer;
        const float* restrict left = (const float*)lhs->_buffer;
        const float* restrict right = (const float*)rhs->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = left[index] / right[index];
    } else if (lhs->_dtype == INT8) {
        int8_t* restrict dst = (int8_t*)result->_buffer;
        const int8_t* restrict left = (const int8_t*)lhs->_buffer;
        const int8_t* restrict right = (const int8_t*)rhs->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index) {
            if (right[index] == 0)
                return FNBLAS_ERR_UNKNOWN;
        }
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = _fnblas_int8_arithmetic(left[index], right[index], DIVIDE);
    } else {
        int32_t* restrict dst = (int32_t*)result->_buffer;
        const int32_t* restrict left = (const int32_t*)lhs->_buffer;
        const int32_t* restrict right = (const int32_t*)rhs->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index) {
            if (right[index] == 0)
                return FNBLAS_ERR_UNKNOWN;
        }
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = _fnblas_int32_arithmetic(left[index], right[index], DIVIDE);
    }
    return FNBLAS_SUCCESS;
}

fnblas_error_t fnblas_op_ve_max(fnblas_vector_t* result, const fnblas_vector_t* lhs, const fnblas_vector_t* rhs)
{
    fnblas_error_t error;
    size_t index;
    error = fnblas_validate_ve(result, lhs, rhs);
    if (error != FNBLAS_SUCCESS)
        return error;
    if (_fnblas_dtype_is_float(lhs->_dtype)) {
        float* restrict dst = (float*)result->_buffer;
        const float* restrict left = (const float*)lhs->_buffer;
        const float* restrict right = (const float*)rhs->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = left[index] > right[index] ? left[index] : right[index];
    } else if (lhs->_dtype == INT8) {
        int8_t* restrict dst = (int8_t*)result->_buffer;
        const int8_t* restrict left = (const int8_t*)lhs->_buffer;
        const int8_t* restrict right = (const int8_t*)rhs->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = left[index] > right[index] ? left[index] : right[index];
    } else {
        int32_t* restrict dst = (int32_t*)result->_buffer;
        const int32_t* restrict left = (const int32_t*)lhs->_buffer;
        const int32_t* restrict right = (const int32_t*)rhs->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = left[index] > right[index] ? left[index] : right[index];
    }
    return FNBLAS_SUCCESS;
}

fnblas_error_t fnblas_op_ve_min(fnblas_vector_t* result, const fnblas_vector_t* lhs, const fnblas_vector_t* rhs)
{
    fnblas_error_t error;
    size_t index;
    error = fnblas_validate_ve(result, lhs, rhs);
    if (error != FNBLAS_SUCCESS)
        return error;
    if (_fnblas_dtype_is_float(lhs->_dtype)) {
        float* restrict dst = (float*)result->_buffer;
        const float* restrict left = (const float*)lhs->_buffer;
        const float* restrict right = (const float*)rhs->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = left[index] < right[index] ? left[index] : right[index];
    } else if (lhs->_dtype == INT8) {
        int8_t* restrict dst = (int8_t*)result->_buffer;
        const int8_t* restrict left = (const int8_t*)lhs->_buffer;
        const int8_t* restrict right = (const int8_t*)rhs->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = left[index] < right[index] ? left[index] : right[index];
    } else {
        int32_t* restrict dst = (int32_t*)result->_buffer;
        const int32_t* restrict left = (const int32_t*)lhs->_buffer;
        const int32_t* restrict right = (const int32_t*)rhs->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = left[index] < right[index] ? left[index] : right[index];
    }
    return FNBLAS_SUCCESS;
}

fnblas_error_t fnblas_op_vv_dot(
    fnblas_scalar_t* result, const fnblas_vector_t* lhs, const fnblas_vector_t* rhs)
{
    fnblas_error_t error;
    size_t index;
    if (result == NULL)
        return FNBLAS_ERR_UNKNOWN;
    error = fnblas_validate_vv(lhs, rhs);
    if (error != FNBLAS_SUCCESS)
        return error;
    error = fnblas_scalar_create_from_dtype(result, lhs->_dtype);
    if (error != FNBLAS_SUCCESS)
        return error;
    if (_fnblas_dtype_is_float(lhs->_dtype)) {
        const float* left = (const float*)lhs->_buffer;
        const float* right = (const float*)rhs->_buffer;
        float sum = 0.0f;
        for (index = 0; index < lhs->_n_elements; ++index)
            sum += left[index] * right[index];
        result->_value.f32 = sum;
    } else if (lhs->_dtype == INT8) {
        const int8_t* left = (const int8_t*)lhs->_buffer;
        const int8_t* right = (const int8_t*)rhs->_buffer;
        int8_t sum = 0;
        for (index = 0; index < lhs->_n_elements; ++index) {
            sum = _fnblas_int8_arithmetic(sum, _fnblas_int8_arithmetic(left[index], right[index], MULTIPLY), ADD);
        }
        result->_value.i8 = sum;
    } else {
        const int32_t* left = (const int32_t*)lhs->_buffer;
        const int32_t* right = (const int32_t*)rhs->_buffer;
        int32_t sum = 0;
        for (index = 0; index < lhs->_n_elements; ++index) {
            sum = _fnblas_int32_arithmetic(
                sum,
                _fnblas_int32_arithmetic(
                    left[index], right[index], MULTIPLY
                ),
                ADD
            );
        }
        result->_value.i32 = sum;
    }
    return FNBLAS_SUCCESS;
}

fnblas_error_t fnblas_op_vr_sum(fnblas_scalar_t* result, const fnblas_vector_t* input)
{
    fnblas_error_t error;
    size_t index;
    if (result == NULL || input == NULL || !_fnblas_dtype_is_valid(input->_dtype)) return FNBLAS_ERR_UNKNOWN;
    if (input->_n_elements != 0 && input->_buffer == NULL) return FNBLAS_ERR_UNKNOWN;
    error = fnblas_scalar_create_from_dtype(result, input->_dtype);
    if (error != FNBLAS_SUCCESS) return error;
    if (_fnblas_dtype_is_float(input->_dtype)) {
        const float* source = (const float*)input->_buffer;
        float sum = 0.0f;
        for (index = 0; index < input->_n_elements; ++index) sum += source[index];
        result->_value.f32 = sum;
    } else if (input->_dtype == INT8) {
        const int8_t* source = (const int8_t*)input->_buffer;
        int8_t sum = 0;
        for (index = 0; index < input->_n_elements; ++index) sum = _fnblas_int8_arithmetic(sum, source[index], ADD);
        result->_value.i8 = sum;
    } else {
        const int32_t* source = (const int32_t*)input->_buffer;
        int32_t sum = 0;
        for (index = 0; index < input->_n_elements; ++index) sum = _fnblas_int32_arithmetic(sum, source[index], ADD);
        result->_value.i32 = sum;
    }
    return FNBLAS_SUCCESS;
}

fnblas_error_t fnblas_op_vr_mean(fnblas_scalar_t* result, const fnblas_vector_t* input)
{
    fnblas_error_t error;
    size_t index;
    if (result == NULL || input == NULL || !_fnblas_dtype_is_valid(input->_dtype)) return FNBLAS_ERR_UNKNOWN;
    if (input->_n_elements == 0) return FNBLAS_ERR_MISMATCH;
    if (input->_buffer == NULL) return FNBLAS_ERR_UNKNOWN;
    error = fnblas_scalar_create_from_dtype(result, input->_dtype);
    if (error != FNBLAS_SUCCESS) return error;
    if (_fnblas_dtype_is_float(input->_dtype)) {
        const float* source = (const float*)input->_buffer;
        float sum = 0.0f;
        for (index = 0; index < input->_n_elements; ++index) sum += source[index];
        result->_value.f32 = sum / (float)input->_n_elements;
    } else if (input->_dtype == INT8) {
        const int8_t* source = (const int8_t*)input->_buffer;
        int8_t sum = 0;
        if (input->_n_elements > INT8_MAX) return FNBLAS_ERR_MISMATCH;
        for (index = 0; index < input->_n_elements; ++index) sum = _fnblas_int8_arithmetic(sum, source[index], ADD);
        result->_value.i8 = (int8_t)(sum / (int8_t)input->_n_elements);
    } else {
        const int32_t* source = (const int32_t*)input->_buffer;
        int32_t sum = 0;
        if (input->_n_elements > INT32_MAX) return FNBLAS_ERR_MISMATCH;
        for (index = 0; index < input->_n_elements; ++index) sum = _fnblas_int32_arithmetic(sum, source[index], ADD);
        result->_value.i32 = sum / (int32_t)input->_n_elements;
    }
    return FNBLAS_SUCCESS;
}

fnblas_error_t fnblas_op_vr_max(fnblas_scalar_t* result, const fnblas_vector_t* input)
{
    fnblas_error_t error;
    size_t index;
    if (result == NULL || input == NULL || !_fnblas_dtype_is_valid(input->_dtype)) return FNBLAS_ERR_UNKNOWN;
    if (input->_n_elements == 0) return FNBLAS_ERR_MISMATCH;
    if (input->_buffer == NULL) return FNBLAS_ERR_UNKNOWN;
    error = fnblas_scalar_create_from_dtype(result, input->_dtype);
    if (error != FNBLAS_SUCCESS) return error;
    if (_fnblas_dtype_is_float(input->_dtype)) {
        const float* source = (const float*)input->_buffer;
        float maximum = source[0];
        for (index = 1; index < input->_n_elements; ++index) maximum = source[index] > maximum ? source[index] : maximum;
        result->_value.f32 = maximum;
    } else if (input->_dtype == INT8) {
        const int8_t* source = (const int8_t*)input->_buffer;
        int8_t maximum = source[0];
        for (index = 1; index < input->_n_elements; ++index) maximum = source[index] > maximum ? source[index] : maximum;
        result->_value.i8 = maximum;
    } else {
        const int32_t* source = (const int32_t*)input->_buffer;
        int32_t maximum = source[0];
        for (index = 1; index < input->_n_elements; ++index) maximum = source[index] > maximum ? source[index] : maximum;
        result->_value.i32 = maximum;
    }
    return FNBLAS_SUCCESS;
}

fnblas_error_t fnblas_op_vr_min(fnblas_scalar_t* result, const fnblas_vector_t* input)
{
    fnblas_error_t error;
    size_t index;
    if (result == NULL || input == NULL || !_fnblas_dtype_is_valid(input->_dtype)) return FNBLAS_ERR_UNKNOWN;
    if (input->_n_elements == 0) return FNBLAS_ERR_MISMATCH;
    if (input->_buffer == NULL) return FNBLAS_ERR_UNKNOWN;
    error = fnblas_scalar_create_from_dtype(result, input->_dtype);
    if (error != FNBLAS_SUCCESS) return error;
    if (_fnblas_dtype_is_float(input->_dtype)) {
        const float* source = (const float*)input->_buffer;
        float minimum = source[0];
        for (index = 1; index < input->_n_elements; ++index) minimum = source[index] < minimum ? source[index] : minimum;
        result->_value.f32 = minimum;
    } else if (input->_dtype == INT8) {
        const int8_t* source = (const int8_t*)input->_buffer;
        int8_t minimum = source[0];
        for (index = 1; index < input->_n_elements; ++index) minimum = source[index] < minimum ? source[index] : minimum;
        result->_value.i8 = minimum;
    } else {
        const int32_t* source = (const int32_t*)input->_buffer;
        int32_t minimum = source[0];
        for (index = 1; index < input->_n_elements; ++index) minimum = source[index] < minimum ? source[index] : minimum;
        result->_value.i32 = minimum;
    }
    return FNBLAS_SUCCESS;
}

fnblas_error_t fnblas_op_vi_exp(fnblas_vector_t* result, const fnblas_vector_t* input)
{
    fnblas_error_t error;
    size_t index;
    float* restrict destination;
    const float* restrict source;
    if (result == NULL || input == NULL || result == input) return FNBLAS_ERR_UNKNOWN;
    if (!_fnblas_dtype_is_valid(input->_dtype) || !_fnblas_dtype_is_float(input->_dtype)) return FNBLAS_ERR_MISMATCH;
    if (input->_n_elements != 0 && input->_buffer == NULL) return FNBLAS_ERR_UNKNOWN;
    if (_fnblas_buffer_is_allocated(result->_status)) {
        if (result->_dtype != input->_dtype || result->_n_elements != input->_n_elements) return FNBLAS_ERR_MISMATCH;
    } else {
        error = fnblas_vector_create(result, input->_n_elements, input->_dtype);
        if (error != FNBLAS_SUCCESS) return error;
    }
    destination = (float*)result->_buffer;
    source = (const float*)input->_buffer;
    for (index = 0; index < input->_n_elements; ++index) destination[index] = expf(source[index]);
    return FNBLAS_SUCCESS;
}

static fnblas_error_t fnblas_prepare_mixed_vector(fnblas_vector_t* result, const fnblas_vector_t* input, fnblas_dtype_t dtype)
{
    if (result == NULL || input == NULL || result == input || !_fnblas_dtype_is_valid(input->_dtype) || !_fnblas_dtype_is_valid(dtype)) return FNBLAS_ERR_UNKNOWN;
    if (input->_n_elements != 0 && input->_buffer == NULL) return FNBLAS_ERR_UNKNOWN;
    if (_fnblas_buffer_is_allocated(result->_status)) {
        if (result->_dtype != dtype || result->_n_elements != input->_n_elements || result->_buffer == input->_buffer) return FNBLAS_ERR_MISMATCH;
        return FNBLAS_SUCCESS;
    }
    return fnblas_vector_create(result, input->_n_elements, dtype);
}

static double fnblas_vector_numeric_value(const fnblas_vector_t* input, size_t index)
{
    if (_fnblas_dtype_is_float(input->_dtype)) return (double)((const float*)input->_buffer)[index];
    if (input->_dtype == INT8) return (double)((const int8_t*)input->_buffer)[index];
    return (double)((const int32_t*)input->_buffer)[index];
}

static int32_t fnblas_vector_clamp_integer(double value, fnblas_dtype_t dtype)
{
    double minimum;
    double maximum;
    if (isnan(value)) {
        errno = EDOM;
        return 0;
    }
    if (dtype == INT4) {
        minimum = -8.0;
        maximum = 7.0;
    } else if (dtype == INT8) {
        minimum = (double)INT8_MIN;
        maximum = (double)INT8_MAX;
    } else {
        minimum = (double)INT32_MIN;
        maximum = (double)INT32_MAX;
    }
    if (value < minimum) {
        errno = ERANGE;
        return (int32_t)minimum;
    }
    if (value > maximum) {
        errno = ERANGE;
        return (int32_t)maximum;
    }
    return (int32_t)value;
}

static void fnblas_vector_set_numeric_value(fnblas_vector_t* result, size_t index, double value)
{
    if (_fnblas_dtype_is_float(result->_dtype)) {
        ((float*)result->_buffer)[index] = (float)value;
    } else {
        const int32_t integer = fnblas_vector_clamp_integer(round(value), result->_dtype);
        if (result->_dtype == INT8)
            ((int8_t*)result->_buffer)[index] = (int8_t)integer;
        else
            ((int32_t*)result->_buffer)[index] = integer;
    }
}

fnblas_error_t fnblas_op_vc_cast(fnblas_vector_t* result, const fnblas_vector_t* input, fnblas_dtype_t dtype)
{
    fnblas_error_t error;
    size_t index;
    if (input == NULL || !_fnblas_dtype_is_valid(dtype)) return FNBLAS_ERR_UNKNOWN;
    if (_fnblas_dtype_is_integer(dtype)) {
        for (index = 0; index < input->_n_elements; ++index) {
            if (isnan(fnblas_vector_numeric_value(input, index))) return FNBLAS_ERR_MISMATCH;
        }
    }
    error = fnblas_prepare_mixed_vector(result, input, dtype);
    if (error != FNBLAS_SUCCESS) return error;
    for (index = 0; index < input->_n_elements; ++index) fnblas_vector_set_numeric_value(result, index, fnblas_vector_numeric_value(input, index));
    return FNBLAS_SUCCESS;
}

fnblas_error_t fnblas_op_vq_quant_per_tensor(fnblas_vector_t* result, const fnblas_vector_t* input, const fnblas_scalar_t* scale, const fnblas_scalar_t* zero_point, fnblas_dtype_t qdtype)
{
    fnblas_error_t error;
    const int is_fp4 = qdtype == FP4;
    const int is_integer = qdtype == INT8 || qdtype == INT4;
    double scale_value;
    double zero_point_value;
    size_t index;
    if (scale == NULL || zero_point == NULL || !_fnblas_dtype_is_float(scale->_dtype)) return FNBLAS_ERR_MISMATCH;
    if ((is_fp4 && !(_fnblas_dtype_is_float(input->_dtype) && (input->_dtype == FP32 || input->_dtype == FP16 || input->_dtype == BF16) && _fnblas_dtype_is_float(zero_point->_dtype))) ||
        (is_integer && !((input->_dtype == FP32 || input->_dtype == BF16) && _fnblas_dtype_is_integer(zero_point->_dtype))) ||
        (!is_fp4 && !is_integer)) return FNBLAS_ERR_MISMATCH;
    scale_value = (double)fnblas_scalar_as_unpacked_float(scale);
    zero_point_value = _fnblas_dtype_is_float(zero_point->_dtype)
        ? (double)fnblas_scalar_as_unpacked_float(zero_point)
        : (double)fnblas_scalar_as_unpacked_int32(zero_point);
    if (!isfinite(scale_value) || scale_value <= 0.0 || !isfinite(zero_point_value)) return FNBLAS_ERR_MISMATCH;
    for (index = 0; index < input->_n_elements; ++index) {
        if (isnan(fnblas_vector_numeric_value(input, index))) return FNBLAS_ERR_MISMATCH;
    }
    error = fnblas_prepare_mixed_vector(result, input, qdtype);
    if (error != FNBLAS_SUCCESS) return error;
    for (index = 0; index < input->_n_elements; ++index) {
        const double input_value = fnblas_vector_numeric_value(input, index);
        const double transformed = input_value / scale_value + zero_point_value;
        if (is_fp4) {
            const byte_t bits = _fnblas_fp32_to_fp4_bits((float)transformed);
            ((float*)result->_buffer)[index] = _fnblas_fp4_bits_to_fp32(bits);
        } else {
            fnblas_vector_set_numeric_value(result, index, transformed);
        }
    }
    return FNBLAS_SUCCESS;
}

fnblas_error_t fnblas_op_vq_dequant_per_tensor(fnblas_vector_t* result, const fnblas_vector_t* input, const fnblas_scalar_t* scale, const fnblas_scalar_t* zero_point, fnblas_dtype_t dtype)
{
    fnblas_error_t error;
    const int is_fp4 = input != NULL && input->_dtype == FP4;
    const int is_integer = input != NULL && (input->_dtype == INT8 || input->_dtype == INT4);
    double scale_value;
    double zero_point_value;
    size_t index;
    if (input == NULL || scale == NULL || zero_point == NULL || !_fnblas_dtype_is_float(scale->_dtype) || !(dtype == FP32 || dtype == FP16 || dtype == BF16)) return FNBLAS_ERR_MISMATCH;
    if ((is_fp4 && !_fnblas_dtype_is_float(zero_point->_dtype)) || (is_integer && !_fnblas_dtype_is_integer(zero_point->_dtype)) || (!is_fp4 && !is_integer)) return FNBLAS_ERR_MISMATCH;
    scale_value = (double)fnblas_scalar_as_unpacked_float(scale);
    zero_point_value = _fnblas_dtype_is_float(zero_point->_dtype)
        ? (double)fnblas_scalar_as_unpacked_float(zero_point)
        : (double)fnblas_scalar_as_unpacked_int32(zero_point);
    if (!isfinite(scale_value) || scale_value <= 0.0 || !isfinite(zero_point_value)) return FNBLAS_ERR_MISMATCH;
    for (index = 0; index < input->_n_elements; ++index) {
        if (isnan(fnblas_vector_numeric_value(input, index))) return FNBLAS_ERR_MISMATCH;
    }
    error = fnblas_prepare_mixed_vector(result, input, dtype);
    if (error != FNBLAS_SUCCESS) return error;
    for (index = 0; index < input->_n_elements; ++index) {
        const double dequantized = (fnblas_vector_numeric_value(input, index) - zero_point_value) * scale_value;
        ((float*)result->_buffer)[index] = (float)dequantized;
    }
    return FNBLAS_SUCCESS;
}
