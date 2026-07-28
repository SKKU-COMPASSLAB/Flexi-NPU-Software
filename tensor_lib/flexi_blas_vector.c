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
        uint8_t* restrict dst = result->_buffer;
        const uint8_t* restrict src = lhs->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = (uint8_t)(src[index] + rhs->_value.u8);
    } else {
        uint32_t scalar;
        uint32_t* restrict dst = (uint32_t*)result->_buffer;
        const uint32_t* restrict src = (const uint32_t*)lhs->_buffer;
        memcpy(&scalar, &rhs->_value.i32, sizeof(scalar));
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = src[index] + scalar;
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
        uint8_t* restrict dst = result->_buffer;
        const uint8_t* restrict src = lhs->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = (uint8_t)(src[index] - rhs->_value.u8);
    } else {
        uint32_t scalar;
        uint32_t* restrict dst = (uint32_t*)result->_buffer;
        const uint32_t* restrict src = (const uint32_t*)lhs->_buffer;
        memcpy(&scalar, &rhs->_value.i32, sizeof(scalar));
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = src[index] - scalar;
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
        uint8_t* restrict dst = result->_buffer;
        const uint8_t* restrict src = lhs->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = (uint8_t)(src[index] * rhs->_value.u8);
    } else {
        uint32_t scalar;
        uint32_t* restrict dst = (uint32_t*)result->_buffer;
        const uint32_t* restrict src = (const uint32_t*)lhs->_buffer;
        memcpy(&scalar, &rhs->_value.i32, sizeof(scalar));
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = src[index] * scalar;
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
        uint8_t* restrict dst = result->_buffer;
        const uint8_t* restrict src = lhs->_buffer;
        if (rhs->_value.u8 == 0)
            return FNBLAS_ERR_UNKNOWN;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = (uint8_t)(src[index] / rhs->_value.u8);
    } else {
        int32_t* restrict dst = (int32_t*)result->_buffer;
        const int32_t* restrict src = (const int32_t*)lhs->_buffer;
        if (rhs->_value.i32 == 0)
            return FNBLAS_ERR_UNKNOWN;
        for (index = 0; index < lhs->_n_elements; ++index) {
            dst[index] =
                src[index] == INT32_MIN && rhs->_value.i32 == -1
                ? INT32_MIN
                : src[index] / rhs->_value.i32;
        }
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
        uint8_t* restrict dst = result->_buffer;
        const uint8_t* restrict src = lhs->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = src[index] > rhs->_value.u8 ? src[index] : rhs->_value.u8;
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
        uint8_t* restrict dst = result->_buffer;
        const uint8_t* restrict src = lhs->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = src[index] < rhs->_value.u8 ? src[index] : rhs->_value.u8;
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
        uint8_t* restrict dst = result->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = (uint8_t)(
                lhs->_buffer[index] + rhs->_buffer[index]
            );
    } else {
        uint32_t* restrict dst = (uint32_t*)result->_buffer;
        const uint32_t* restrict left = (const uint32_t*)lhs->_buffer;
        const uint32_t* restrict right = (const uint32_t*)rhs->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = left[index] + right[index];
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
        uint8_t* restrict dst = result->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = (uint8_t)(
                lhs->_buffer[index] - rhs->_buffer[index]
            );
    } else {
        uint32_t* restrict dst = (uint32_t*)result->_buffer;
        const uint32_t* restrict left = (const uint32_t*)lhs->_buffer;
        const uint32_t* restrict right = (const uint32_t*)rhs->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = left[index] - right[index];
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
        uint8_t* restrict dst = result->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = (uint8_t)(
                lhs->_buffer[index] * rhs->_buffer[index]
            );
    } else {
        uint32_t* restrict dst = (uint32_t*)result->_buffer;
        const uint32_t* restrict left = (const uint32_t*)lhs->_buffer;
        const uint32_t* restrict right = (const uint32_t*)rhs->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = left[index] * right[index];
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
        uint8_t* restrict dst = result->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index) {
            if (rhs->_buffer[index] == 0)
                return FNBLAS_ERR_UNKNOWN;
        }
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = (uint8_t)(
                lhs->_buffer[index] / rhs->_buffer[index]
            );
    } else {
        int32_t* restrict dst = (int32_t*)result->_buffer;
        const int32_t* restrict left = (const int32_t*)lhs->_buffer;
        const int32_t* restrict right = (const int32_t*)rhs->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index) {
            if (right[index] == 0)
                return FNBLAS_ERR_UNKNOWN;
        }
        for (index = 0; index < lhs->_n_elements; ++index) {
            dst[index] =
                left[index] == INT32_MIN && right[index] == -1
                ? INT32_MIN
                : left[index] / right[index];
        }
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
        uint8_t* restrict dst = result->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = lhs->_buffer[index] > rhs->_buffer[index] ? lhs->_buffer[index] : rhs->_buffer[index];
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
        uint8_t* restrict dst = result->_buffer;
        for (index = 0; index < lhs->_n_elements; ++index)
            dst[index] = lhs->_buffer[index] < rhs->_buffer[index] ? lhs->_buffer[index] : rhs->_buffer[index];
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
        uint8_t sum = 0;
        for (index = 0; index < lhs->_n_elements; ++index) {
            sum = _fnblas_uint8_arithmetic(
                sum,
                _fnblas_uint8_arithmetic(
                    lhs->_buffer[index],
                    rhs->_buffer[index],
                    MULTIPLY
                ),
                ADD
            );
        }
        result->_value.u8 = sum;
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
        uint8_t sum = 0;
        for (index = 0; index < input->_n_elements; ++index) sum = _fnblas_uint8_arithmetic(sum, input->_buffer[index], ADD);
        result->_value.u8 = sum;
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
        uint8_t sum = 0;
        if (input->_n_elements > UINT8_MAX) return FNBLAS_ERR_MISMATCH;
        for (index = 0; index < input->_n_elements; ++index) sum = _fnblas_uint8_arithmetic(sum, input->_buffer[index], ADD);
        result->_value.u8 = (uint8_t)(sum / (uint8_t)input->_n_elements);
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
        uint8_t maximum = input->_buffer[0];
        for (index = 1; index < input->_n_elements; ++index) maximum = input->_buffer[index] > maximum ? input->_buffer[index] : maximum;
        result->_value.u8 = maximum;
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
        uint8_t minimum = input->_buffer[0];
        for (index = 1; index < input->_n_elements; ++index) minimum = input->_buffer[index] < minimum ? input->_buffer[index] : minimum;
        result->_value.u8 = minimum;
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
