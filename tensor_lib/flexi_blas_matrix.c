#include "flexi_blas_internal.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

static fnblas_error_t fnblas_matrix_element_count(
    size_t n_rows, size_t n_cols, size_t* count)
{
    if (count == NULL || (n_rows != 0 && n_cols > SIZE_MAX / n_rows)) {
        errno = EOVERFLOW;
        return FNBLAS_ERR_UNKNOWN;
    }
    *count = n_rows * n_cols;
    return FNBLAS_SUCCESS;
}

fnblas_error_t fnblas_matrix_create(
    fnblas_matrix_t* matrix,
    size_t n_rows,
    size_t n_cols,
    fnblas_dtype_t dtype)
{
    fnblas_error_t error;
    size_t count;
    byte_t* new_buffer = NULL;
    if (matrix == NULL)
        return FNBLAS_ERR_UNKNOWN;
    if (!_fnblas_dtype_is_valid(dtype))
        return FNBLAS_ERR_MISMATCH;
    if (_fnblas_buffer_is_allocated(matrix->_status) || _fnblas_buffer_is_view(matrix->_status))
        return FNBLAS_ERR_UNKNOWN;
    error = fnblas_matrix_element_count(n_rows, n_cols, &count);
    if (error != FNBLAS_SUCCESS)
        return error;
    error = _fnblas_allocate_buffer(&new_buffer, count, dtype);
    if (error != FNBLAS_SUCCESS)
        return error;
    matrix->_buffer = new_buffer;
    matrix->_n_rows = n_rows;
    matrix->_n_cols = n_cols;
    matrix->_dtype = dtype;
    matrix->_status = 0;
    if (new_buffer != NULL)
        matrix->_status |= FNBLAS_STATUS_BUFFER_ALLOCATED;
    return FNBLAS_SUCCESS;
}

fnblas_error_t fnblas_matrix_create_view(fnblas_matrix_t* view, byte_t* buffer, size_t n_rows, size_t n_cols, fnblas_dtype_t dtype)
{
    fnblas_error_t error;
    size_t count;
    if (view == NULL)
        return FNBLAS_ERR_UNKNOWN;
    if (!_fnblas_dtype_is_valid(dtype))
        return FNBLAS_ERR_MISMATCH;
    if (_fnblas_buffer_is_allocated(view->_status) || _fnblas_buffer_is_view(view->_status))
        return FNBLAS_ERR_UNKNOWN;
    error = fnblas_matrix_element_count(n_rows, n_cols, &count);
    if (error != FNBLAS_SUCCESS)
        return error;
    if (count != 0 && buffer == NULL)
        return FNBLAS_ERR_MISMATCH;
    view->_buffer = buffer;
    view->_n_rows = n_rows;
    view->_n_cols = n_cols;
    view->_dtype = dtype;
    view->_status = FNBLAS_STATUS_VIEW;
    if (buffer != NULL)
        view->_status |= FNBLAS_STATUS_BUFFER_ALLOCATED;
    return FNBLAS_SUCCESS;
}

fnblas_error_t fnblas_matrix_create_reshape_view(fnblas_matrix_t* view, fnblas_matrix_t* source, size_t n_rows, size_t n_cols)
{
    fnblas_error_t error;
    size_t source_count;
    size_t view_count;
    if (source == NULL)
        return FNBLAS_ERR_UNKNOWN;
    error = fnblas_matrix_element_count(source->_n_rows, source->_n_cols, &source_count);
    if (error != FNBLAS_SUCCESS)
        return error;
    error = fnblas_matrix_element_count(n_rows, n_cols, &view_count);
    if (error != FNBLAS_SUCCESS)
        return error;
    if (source_count != view_count)
        return FNBLAS_ERR_MISMATCH;
    return fnblas_matrix_create_view(view, source->_buffer, n_rows, n_cols, source->_dtype);
}

fnblas_error_t fnblas_matrix_create_row_vector_view(fnblas_vector_t* view, fnblas_matrix_t* matrix, size_t row)
{
    size_t element_size;
    size_t offset;
    if (matrix == NULL || row >= matrix->_n_rows)
        return FNBLAS_ERR_MISMATCH;
    element_size = _fnblas_dtype_unpacked_size_of(matrix->_dtype);
    if (element_size == 0)
        return FNBLAS_ERR_MISMATCH;
    offset = row * matrix->_n_cols * element_size;
    return fnblas_vector_create_view(view, matrix->_buffer == NULL ? NULL : matrix->_buffer + offset, matrix->_n_cols, matrix->_dtype);
}

fnblas_error_t fnblas_matrix_destroy(fnblas_matrix_t* matrix)
{
    if (matrix == NULL)
        return FNBLAS_ERR_UNKNOWN;
    if (_fnblas_buffer_is_allocated(matrix->_status) && !_fnblas_buffer_is_view(matrix->_status))
        _fnblas_deallocate_buffer(matrix->_buffer);
    matrix->_buffer = NULL;
    matrix->_n_rows = 0;
    matrix->_n_cols = 0;
    matrix->_dtype = FP32;
    matrix->_status = 0;
    return FNBLAS_SUCCESS;
}

fnblas_error_t fnblas_matrix_at(
    const fnblas_matrix_t* matrix,
    size_t row,
    size_t col,
    byte_t* out_element)
{
    size_t element_size;
    size_t index;
    if (matrix == NULL || out_element == NULL ||
        row >= matrix->_n_rows || col >= matrix->_n_cols) {
        return FNBLAS_ERR_UNKNOWN;
    }
    element_size = _fnblas_dtype_unpacked_size_of(matrix->_dtype);
    index = row * matrix->_n_cols + col;
    memcpy(
        out_element,
        matrix->_buffer + index * element_size,
        element_size
    );
    return FNBLAS_SUCCESS;
}

fnblas_error_t fnblas_matrix_set(
    fnblas_matrix_t* matrix,
    size_t row,
    size_t col,
    const byte_t* element)
{
    size_t element_size;
    size_t index;
    if (matrix == NULL || element == NULL ||
        row >= matrix->_n_rows || col >= matrix->_n_cols) {
        return FNBLAS_ERR_UNKNOWN;
    }
    element_size = _fnblas_dtype_unpacked_size_of(matrix->_dtype);
    index = row * matrix->_n_cols + col;
    memcpy(
        matrix->_buffer + index * element_size,
        element,
        element_size
    );
    return FNBLAS_SUCCESS;
}

fnblas_error_t fnblas_matrix_get_packed_buffer(
    const fnblas_matrix_t* matrix, byte_t* out_buffer)
{
    fnblas_error_t error;
    size_t count;
    if (matrix == NULL)
        return FNBLAS_ERR_UNKNOWN;
    error = fnblas_matrix_element_count(
        matrix->_n_rows, matrix->_n_cols, &count
    );
    if (error != FNBLAS_SUCCESS)
        return error;
    if (count != 0 && out_buffer == NULL)
        return FNBLAS_ERR_UNKNOWN;
    _fnblas_pack_buffer(
        matrix->_buffer, count, matrix->_dtype, out_buffer
    );
    return FNBLAS_SUCCESS;
}

fnblas_error_t fnblas_matrix_initialize_from_packed_buffer(
    fnblas_matrix_t* matrix, const byte_t* packed_buffer)
{
    fnblas_error_t error;
    size_t count;
    if (matrix == NULL)
        return FNBLAS_ERR_UNKNOWN;
    error = fnblas_matrix_element_count(
        matrix->_n_rows, matrix->_n_cols, &count
    );
    if (error != FNBLAS_SUCCESS)
        return error;
    if (count != 0 && packed_buffer == NULL)
        return FNBLAS_ERR_UNKNOWN;
    _fnblas_unpack_buffer(
        matrix->_buffer, count, matrix->_dtype, packed_buffer
    );
    return FNBLAS_SUCCESS;
}

fnblas_error_t fnblas_matrix_initialize_from_unpacked_buffer(
    fnblas_matrix_t* matrix, const byte_t* unpacked_buffer)
{
    fnblas_error_t error;
    size_t count;
    size_t element_size;
    if (matrix == NULL)
        return FNBLAS_ERR_UNKNOWN;
    error = fnblas_matrix_element_count(
        matrix->_n_rows, matrix->_n_cols, &count
    );
    if (error != FNBLAS_SUCCESS)
        return error;
    if (count != 0 && unpacked_buffer == NULL)
        return FNBLAS_ERR_UNKNOWN;
    element_size = _fnblas_dtype_unpacked_size_of(matrix->_dtype);
    if (count != 0)
        memcpy(matrix->_buffer, unpacked_buffer, count * element_size);
    return FNBLAS_SUCCESS;
}

size_t fnblas_matrix_n_rows(const fnblas_matrix_t* matrix)
{
    if (matrix == NULL) {
        errno = EINVAL;
        return 0;
    }
    return matrix->_n_rows;
}

size_t fnblas_matrix_n_cols(const fnblas_matrix_t* matrix)
{
    if (matrix == NULL) {
        errno = EINVAL;
        return 0;
    }
    return matrix->_n_cols;
}

fnblas_dtype_t fnblas_matrix_dtype(const fnblas_matrix_t* matrix)
{
    if (matrix == NULL) {
        errno = EINVAL;
        return INT8;
    }
    return matrix->_dtype;
}

fnblas_error_t fnblas_op_mi_transpose(
    fnblas_matrix_t* result, const fnblas_matrix_t* input)
{
    fnblas_error_t error;
    size_t row;
    size_t col;
    size_t element_size;
    if (result == NULL || input == NULL || result == input)
        return FNBLAS_ERR_UNKNOWN;
    if (_fnblas_buffer_is_allocated(result->_status)) {
        if (result->_dtype != input->_dtype ||
            result->_n_rows != input->_n_cols ||
            result->_n_cols != input->_n_rows) {
            return FNBLAS_ERR_MISMATCH;
        }
    } else {
        error = fnblas_matrix_create(
            result, input->_n_cols, input->_n_rows, input->_dtype
        );
        if (error != FNBLAS_SUCCESS)
            return error;
    }
    element_size = _fnblas_dtype_unpacked_size_of(input->_dtype);
    for (row = 0; row < input->_n_rows; ++row) {
        for (col = 0; col < input->_n_cols; ++col) {
            const size_t source = row * input->_n_cols + col;
            const size_t destination = col * input->_n_rows + row;
            memcpy(
                result->_buffer + destination * element_size,
                input->_buffer + source * element_size,
                element_size
            );
        }
    }
    return FNBLAS_SUCCESS;
}

static fnblas_error_t fnblas_prepare_matrix_elementwise(
    fnblas_matrix_t* result, const fnblas_matrix_t* lhs, const fnblas_matrix_t* rhs)
{
    if (result == NULL || lhs == NULL || rhs == NULL ||
        result == lhs || result == rhs)
        return FNBLAS_ERR_UNKNOWN;
    if (lhs->_dtype != rhs->_dtype ||
        lhs->_n_rows != rhs->_n_rows ||
        lhs->_n_cols != rhs->_n_cols)
        return FNBLAS_ERR_MISMATCH;
    if (_fnblas_buffer_is_allocated(result->_status)) {
        if (result->_n_rows != lhs->_n_rows ||
            result->_n_cols != lhs->_n_cols) {
            return FNBLAS_ERR_MISMATCH;
        }
        return FNBLAS_SUCCESS;
    }
    return fnblas_matrix_create(
        result, lhs->_n_rows, lhs->_n_cols, lhs->_dtype
    );
}

static fnblas_error_t fnblas_matrix_elementwise(
    fnblas_matrix_t* result,
    const fnblas_matrix_t* lhs,
    const fnblas_matrix_t* rhs,
    fnblas_arithmetic_op_t operation)
{
    fnblas_error_t error;
    size_t count;
    size_t index;
    error = fnblas_prepare_matrix_elementwise(result, lhs, rhs);
    if (error != FNBLAS_SUCCESS)
        return error;
    count = lhs->_n_rows * lhs->_n_cols;
    if (operation == DIVIDE && !_fnblas_dtype_is_float(lhs->_dtype)) {
        if (lhs->_dtype == INT8) {
            for (index = 0; index < count; ++index) {
                if (rhs->_buffer[index] == 0)
                    return FNBLAS_ERR_UNKNOWN;
            }
        } else {
            const int32_t* right = (const int32_t*)rhs->_buffer;
            for (index = 0; index < count; ++index) {
                if (right[index] == 0)
                    return FNBLAS_ERR_UNKNOWN;
            }
        }
    }
    if (_fnblas_dtype_is_float(lhs->_dtype)) {
        float* dst = (float*)result->_buffer;
        const float* left = (const float*)lhs->_buffer;
        const float* right = (const float*)rhs->_buffer;
        for (index = 0; index < count; ++index)
            dst[index] = _fnblas_float_arithmetic(
                left[index], right[index], operation
            );
    } else if (lhs->_dtype == INT8) {
        for (index = 0; index < count; ++index)
            result->_buffer[index] = _fnblas_uint8_arithmetic(
                lhs->_buffer[index], rhs->_buffer[index], operation
            );
    } else {
        int32_t* dst = (int32_t*)result->_buffer;
        const int32_t* left = (const int32_t*)lhs->_buffer;
        const int32_t* right = (const int32_t*)rhs->_buffer;
        for (index = 0; index < count; ++index)
            dst[index] = _fnblas_int32_arithmetic(
                left[index], right[index], operation
            );
    }
    return FNBLAS_SUCCESS;
}

fnblas_error_t fnblas_op_me_add(
    fnblas_matrix_t* result, const fnblas_matrix_t* lhs, const fnblas_matrix_t* rhs)
{
    return fnblas_matrix_elementwise(result, lhs, rhs, ADD);
}

fnblas_error_t fnblas_op_me_sub(
    fnblas_matrix_t* result, const fnblas_matrix_t* lhs, const fnblas_matrix_t* rhs)
{
    return fnblas_matrix_elementwise(result, lhs, rhs, SUBTRACT);
}

fnblas_error_t fnblas_op_me_mul(
    fnblas_matrix_t* result, const fnblas_matrix_t* lhs, const fnblas_matrix_t* rhs)
{
    return fnblas_matrix_elementwise(result, lhs, rhs, MULTIPLY);
}

fnblas_error_t fnblas_op_me_div(
    fnblas_matrix_t* result, const fnblas_matrix_t* lhs, const fnblas_matrix_t* rhs)
{
    return fnblas_matrix_elementwise(result, lhs, rhs, DIVIDE);
}

fnblas_error_t fnblas_op_mm_matmul(
    fnblas_matrix_t* result, const fnblas_matrix_t* lhs, const fnblas_matrix_t* rhs_t)
{
    fnblas_error_t error;
    size_t row;
    size_t col;
    size_t reduction;
    if (result == NULL || lhs == NULL || rhs_t == NULL ||
        result == lhs || result == rhs_t)
        return FNBLAS_ERR_UNKNOWN;
    if (lhs->_dtype != rhs_t->_dtype ||
        lhs->_n_cols != rhs_t->_n_cols)
        return FNBLAS_ERR_MISMATCH;
    if (_fnblas_buffer_is_allocated(result->_status)) {
        if (result->_n_rows != lhs->_n_rows ||
            result->_n_cols != rhs_t->_n_rows) {
            return FNBLAS_ERR_MISMATCH;
        }
    } else {
        error = fnblas_matrix_create(
            result, lhs->_n_rows, rhs_t->_n_rows, lhs->_dtype
        );
        if (error != FNBLAS_SUCCESS)
            return error;
    }

    if (_fnblas_dtype_is_float(lhs->_dtype)) {
        float* dst = (float*)result->_buffer;
        const float* left = (const float*)lhs->_buffer;
        const float* right = (const float*)rhs_t->_buffer;
        for (row = 0; row < lhs->_n_rows; ++row) {
            for (col = 0; col < rhs_t->_n_rows; ++col) {
                float sum = 0.0f;
                for (reduction = 0;
                     reduction < lhs->_n_cols;
                     ++reduction) {
                    sum +=
                        left[row * lhs->_n_cols + reduction] *
                        right[col * rhs_t->_n_cols + reduction];
                }
                dst[row * rhs_t->_n_rows + col] = sum;
            }
        }
    } else if (lhs->_dtype == INT8) {
        for (row = 0; row < lhs->_n_rows; ++row) {
            for (col = 0; col < rhs_t->_n_rows; ++col) {
                uint8_t sum = 0;
                for (reduction = 0;
                     reduction < lhs->_n_cols;
                     ++reduction) {
                    const uint8_t product = _fnblas_uint8_arithmetic(
                        lhs->_buffer[
                            row * lhs->_n_cols + reduction
                        ],
                        rhs_t->_buffer[
                            col * rhs_t->_n_cols + reduction
                        ],
                        MULTIPLY
                    );
                    sum = _fnblas_uint8_arithmetic(sum, product, ADD);
                }
                result->_buffer[
                    row * rhs_t->_n_rows + col
                ] = sum;
            }
        }
    } else {
        int32_t* dst = (int32_t*)result->_buffer;
        const int32_t* left = (const int32_t*)lhs->_buffer;
        const int32_t* right = (const int32_t*)rhs_t->_buffer;
        for (row = 0; row < lhs->_n_rows; ++row) {
            for (col = 0; col < rhs_t->_n_rows; ++col) {
                int32_t sum = 0;
                for (reduction = 0;
                     reduction < lhs->_n_cols;
                     ++reduction) {
                    const int32_t product = _fnblas_int32_arithmetic(
                        left[row * lhs->_n_cols + reduction],
                        right[col * rhs_t->_n_cols + reduction],
                        MULTIPLY
                    );
                    sum = _fnblas_int32_arithmetic(sum, product, ADD);
                }
                dst[row * rhs_t->_n_rows + col] = sum;
            }
        }
    }
    return FNBLAS_SUCCESS;
}

fnblas_error_t fnblas_op_mmv_axpy(fnblas_matrix_t* result, const fnblas_matrix_t* a, const fnblas_matrix_t* x_t, const fnblas_vector_t* y)
{
    fnblas_error_t error;
    size_t row;
    size_t col;
    size_t reduction;
    if (result == NULL || a == NULL || x_t == NULL || y == NULL) return FNBLAS_ERR_UNKNOWN;
    if (a->_dtype != x_t->_dtype || a->_dtype != y->_dtype || a->_n_cols != x_t->_n_cols || y->_n_elements != x_t->_n_rows) return FNBLAS_ERR_MISMATCH;
    if (result == a || result == x_t || result->_buffer == y->_buffer) return FNBLAS_ERR_UNKNOWN;
    if (_fnblas_buffer_is_allocated(result->_status)) {
        if (result->_dtype != a->_dtype || result->_n_rows != a->_n_rows || result->_n_cols != x_t->_n_rows) return FNBLAS_ERR_MISMATCH;
    } else {
        error = fnblas_matrix_create(result, a->_n_rows, x_t->_n_rows, a->_dtype);
        if (error != FNBLAS_SUCCESS) return error;
    }
    if (_fnblas_dtype_is_float(a->_dtype)) {
        float* dst = (float*)result->_buffer;
        const float* left = (const float*)a->_buffer;
        const float* right = (const float*)x_t->_buffer;
        const float* addend = (const float*)y->_buffer;
        for (row = 0; row < a->_n_rows; ++row) {
            for (col = 0; col < x_t->_n_rows; ++col) {
                float sum = 0.0f;
                for (reduction = 0; reduction < a->_n_cols; ++reduction) sum += left[row * a->_n_cols + reduction] * right[col * x_t->_n_cols + reduction];
                dst[row * x_t->_n_rows + col] = sum + addend[col];
            }
        }
    } else if (a->_dtype == INT8) {
        for (row = 0; row < a->_n_rows; ++row) {
            for (col = 0; col < x_t->_n_rows; ++col) {
                uint8_t sum = 0;
                for (reduction = 0; reduction < a->_n_cols; ++reduction) {
                    const uint8_t product = _fnblas_uint8_arithmetic(a->_buffer[row * a->_n_cols + reduction], x_t->_buffer[col * x_t->_n_cols + reduction], MULTIPLY);
                    sum = _fnblas_uint8_arithmetic(sum, product, ADD);
                }
                result->_buffer[row * x_t->_n_rows + col] = _fnblas_uint8_arithmetic(sum, y->_buffer[col], ADD);
            }
        }
    } else {
        int32_t* dst = (int32_t*)result->_buffer;
        const int32_t* left = (const int32_t*)a->_buffer;
        const int32_t* right = (const int32_t*)x_t->_buffer;
        const int32_t* addend = (const int32_t*)y->_buffer;
        for (row = 0; row < a->_n_rows; ++row) {
            for (col = 0; col < x_t->_n_rows; ++col) {
                int32_t sum = 0;
                for (reduction = 0; reduction < a->_n_cols; ++reduction) {
                    const int32_t product = _fnblas_int32_arithmetic(left[row * a->_n_cols + reduction], right[col * x_t->_n_cols + reduction], MULTIPLY);
                    sum = _fnblas_int32_arithmetic(sum, product, ADD);
                }
                dst[row * x_t->_n_rows + col] = _fnblas_int32_arithmetic(sum, addend[col], ADD);
            }
        }
    }
    return FNBLAS_SUCCESS;
}

fnblas_error_t fnblas_op_mmm_axpy(fnblas_matrix_t* result, const fnblas_matrix_t* a, const fnblas_matrix_t* x_t, const fnblas_matrix_t* y)
{
    fnblas_error_t error;
    size_t row;
    size_t col;
    size_t reduction;
    if (result == NULL || a == NULL || x_t == NULL || y == NULL) return FNBLAS_ERR_UNKNOWN;
    if (a->_dtype != x_t->_dtype || a->_dtype != y->_dtype || a->_n_cols != x_t->_n_cols || y->_n_rows != a->_n_rows || y->_n_cols != x_t->_n_rows) return FNBLAS_ERR_MISMATCH;
    if (result == a || result == x_t || result == y) return FNBLAS_ERR_UNKNOWN;
    if (_fnblas_buffer_is_allocated(result->_status)) {
        if (result->_dtype != a->_dtype || result->_n_rows != a->_n_rows || result->_n_cols != x_t->_n_rows) return FNBLAS_ERR_MISMATCH;
    } else {
        error = fnblas_matrix_create(result, a->_n_rows, x_t->_n_rows, a->_dtype);
        if (error != FNBLAS_SUCCESS) return error;
    }
    if (_fnblas_dtype_is_float(a->_dtype)) {
        float* dst = (float*)result->_buffer;
        const float* left = (const float*)a->_buffer;
        const float* right = (const float*)x_t->_buffer;
        const float* addend = (const float*)y->_buffer;
        for (row = 0; row < a->_n_rows; ++row) {
            for (col = 0; col < x_t->_n_rows; ++col) {
                float sum = 0.0f;
                for (reduction = 0; reduction < a->_n_cols; ++reduction) sum += left[row * a->_n_cols + reduction] * right[col * x_t->_n_cols + reduction];
                dst[row * x_t->_n_rows + col] = sum + addend[row * y->_n_cols + col];
            }
        }
    } else if (a->_dtype == INT8) {
        for (row = 0; row < a->_n_rows; ++row) {
            for (col = 0; col < x_t->_n_rows; ++col) {
                uint8_t sum = 0;
                for (reduction = 0; reduction < a->_n_cols; ++reduction) {
                    const uint8_t product = _fnblas_uint8_arithmetic(a->_buffer[row * a->_n_cols + reduction], x_t->_buffer[col * x_t->_n_cols + reduction], MULTIPLY);
                    sum = _fnblas_uint8_arithmetic(sum, product, ADD);
                }
                result->_buffer[row * x_t->_n_rows + col] = _fnblas_uint8_arithmetic(sum, y->_buffer[row * y->_n_cols + col], ADD);
            }
        }
    } else {
        int32_t* dst = (int32_t*)result->_buffer;
        const int32_t* left = (const int32_t*)a->_buffer;
        const int32_t* right = (const int32_t*)x_t->_buffer;
        const int32_t* addend = (const int32_t*)y->_buffer;
        for (row = 0; row < a->_n_rows; ++row) {
            for (col = 0; col < x_t->_n_rows; ++col) {
                int32_t sum = 0;
                for (reduction = 0; reduction < a->_n_cols; ++reduction) {
                    const int32_t product = _fnblas_int32_arithmetic(left[row * a->_n_cols + reduction], right[col * x_t->_n_cols + reduction], MULTIPLY);
                    sum = _fnblas_int32_arithmetic(sum, product, ADD);
                }
                dst[row * x_t->_n_rows + col] = _fnblas_int32_arithmetic(sum, addend[row * y->_n_cols + col], ADD);
            }
        }
    }
    return FNBLAS_SUCCESS;
}

size_t fnblas_matrix_packed_buffer_size(const fnblas_matrix_t* matrix)
{
    return matrix->_n_rows * matrix->_n_cols * fnblas_dtype_size_of(matrix->_dtype) / fnblas_dtype_pack_size_of(matrix->_dtype);
}

size_t fnblas_matrix_unpacked_buffer_size(const fnblas_matrix_t* matrix)
{
    return matrix->_n_rows * matrix->_n_cols * fnblas_dtype_size_of(matrix->_dtype);
}
