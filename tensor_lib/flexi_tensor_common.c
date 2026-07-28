#include "flexi_tensor_common.h"
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

static flexi_tensor_error_t flexi_tensor_allocate_size_array(size_t** array, size_t count) {
    fnblas_error_t error;
    byte_t* buffer = NULL;
    if (array == NULL || count > SIZE_MAX / sizeof(**array)) {
        errno = count > SIZE_MAX / sizeof(**array) ? EOVERFLOW : EINVAL;
        return count > SIZE_MAX / sizeof(**array) ? FLEXI_TENSOR_ERR_MALLOC_FAIL : FLEXI_TENSOR_ERR_UNKNOWN;
    }
    error = _fnblas_allocate_buffer(&buffer, count * sizeof(**array), INT8);
    if (error != FNBLAS_SUCCESS) return flexi_tensor_from_fnblas_error(error);
    *array = (size_t*)buffer;
    return FLEXI_TENSOR_SUCCESS;
}

static flexi_tensor_error_t flexi_tensor_element_count(const flexi_tensor_t* tensor, size_t* count) {
    size_t dim;
    size_t result = 1;
    if (tensor == NULL || count == NULL || tensor->_n_dims == 0 || tensor->_n_dims > 4 || tensor->_shape == NULL) {
        errno = EINVAL;
        return FLEXI_TENSOR_ERR_UNKNOWN;
    }
    for (dim = 0; dim < tensor->_n_dims; ++dim) {
        if (tensor->_shape[dim] != 0 && result > SIZE_MAX / tensor->_shape[dim]) {
            errno = EOVERFLOW;
            return FLEXI_TENSOR_ERR_MALLOC_FAIL;
        }
        result *= tensor->_shape[dim];
    }
    *count = result;
    return FLEXI_TENSOR_SUCCESS;
}

flexi_tensor_error_t flexi_tensor_create(flexi_tensor_t* tensor, flexi_tuple_t shape, fnblas_dtype_t dtype) {
    flexi_tensor_error_t tensor_error;
    fnblas_error_t blas_error;
    size_t count;
    size_t dim;
    size_t stride = 1;
    size_t* new_shape;
    size_t* new_strides;
    byte_t* new_buffer = NULL;
    flexi_tensor_t temporary = FLEXI_TENSOR_INITIALIZER;
    if (tensor == NULL || shape._n_dims == 0 || shape._n_dims > 4) return FLEXI_TENSOR_ERR_UNKNOWN;
    if (!_fnblas_dtype_is_valid(dtype)) return FLEXI_TENSOR_ERR_MISMATCH;
    if (_fnblas_buffer_is_allocated(tensor->_status) || _fnblas_buffer_is_view(tensor->_status) || tensor->_shape != NULL || tensor->_strides != NULL) return FLEXI_TENSOR_ERR_UNKNOWN;
    tensor_error = flexi_tensor_allocate_size_array(&new_shape, shape._n_dims);
    if (tensor_error != FLEXI_TENSOR_SUCCESS) return tensor_error;
    tensor_error = flexi_tensor_allocate_size_array(&new_strides, shape._n_dims);
    if (tensor_error != FLEXI_TENSOR_SUCCESS) {
        _fnblas_deallocate_buffer((byte_t*)new_shape);
        return tensor_error;
    }
    memcpy(new_shape, shape._values, shape._n_dims * sizeof(*new_shape));
    for (dim = shape._n_dims; dim != 0; --dim) {
        const size_t index = dim - 1;
        new_strides[index] = stride;
        if (new_shape[index] != 0 && stride > SIZE_MAX / new_shape[index]) {
            _fnblas_deallocate_buffer((byte_t*)new_strides);
            _fnblas_deallocate_buffer((byte_t*)new_shape);
            errno = EOVERFLOW;
            return FLEXI_TENSOR_ERR_MALLOC_FAIL;
        }
        stride *= new_shape[index];
    }
    temporary._n_dims = shape._n_dims;
    temporary._shape = new_shape;
    temporary._strides = new_strides;
    temporary._dtype = dtype;
    tensor_error = flexi_tensor_element_count(&temporary, &count);
    if (tensor_error != FLEXI_TENSOR_SUCCESS) {
        _fnblas_deallocate_buffer((byte_t*)new_strides);
        _fnblas_deallocate_buffer((byte_t*)new_shape);
        return tensor_error;
    }
    blas_error = _fnblas_allocate_buffer(&new_buffer, count, dtype);
    if (blas_error != FNBLAS_SUCCESS) {
        _fnblas_deallocate_buffer((byte_t*)new_strides);
        _fnblas_deallocate_buffer((byte_t*)new_shape);
        return flexi_tensor_from_fnblas_error(blas_error);
    }
    tensor->_buffer = new_buffer;
    tensor->_n_dims = shape._n_dims;
    tensor->_shape = new_shape;
    tensor->_strides = new_strides;
    tensor->_dtype = dtype;
    tensor->_status = 0;
    if (new_buffer != NULL) tensor->_status |= FNBLAS_STATUS_BUFFER_ALLOCATED;
    return FLEXI_TENSOR_SUCCESS;
}

flexi_tensor_error_t flexi_tensor_create_view(flexi_tensor_t* view, byte_t* buffer, flexi_tuple_t shape, fnblas_dtype_t dtype) {
    flexi_tensor_error_t error;
    size_t dim;
    size_t stride = 1;
    size_t* new_shape;
    size_t* new_strides;
    if (view == NULL || shape._n_dims == 0 || shape._n_dims > 4) return FLEXI_TENSOR_ERR_UNKNOWN;
    if (!_fnblas_dtype_is_valid(dtype)) return FLEXI_TENSOR_ERR_MISMATCH;
    if (_fnblas_buffer_is_allocated(view->_status) || _fnblas_buffer_is_view(view->_status) || view->_shape != NULL || view->_strides != NULL) return FLEXI_TENSOR_ERR_UNKNOWN;
    error = flexi_tensor_allocate_size_array(&new_shape, shape._n_dims);
    if (error != FLEXI_TENSOR_SUCCESS) return error;
    error = flexi_tensor_allocate_size_array(&new_strides, shape._n_dims);
    if (error != FLEXI_TENSOR_SUCCESS) {
        _fnblas_deallocate_buffer((byte_t*)new_shape);
        return error;
    }
    memcpy(new_shape, shape._values, shape._n_dims * sizeof(*new_shape));
    for (dim = shape._n_dims; dim != 0; --dim) {
        const size_t current = dim - 1;
        new_strides[current] = stride;
        if (new_shape[current] != 0 && stride > SIZE_MAX / new_shape[current]) {
            _fnblas_deallocate_buffer((byte_t*)new_strides);
            _fnblas_deallocate_buffer((byte_t*)new_shape);
            errno = EOVERFLOW;
            return FLEXI_TENSOR_ERR_MISMATCH;
        }
        stride *= new_shape[current];
    }
    if (stride != 0 && buffer == NULL) {
        _fnblas_deallocate_buffer((byte_t*)new_strides);
        _fnblas_deallocate_buffer((byte_t*)new_shape);
        return FLEXI_TENSOR_ERR_MISMATCH;
    }
    view->_buffer = buffer;
    view->_n_dims = shape._n_dims;
    view->_shape = new_shape;
    view->_strides = new_strides;
    view->_dtype = dtype;
    view->_status = FNBLAS_STATUS_VIEW;
    if (buffer != NULL) view->_status |= FNBLAS_STATUS_BUFFER_ALLOCATED;
    return FLEXI_TENSOR_SUCCESS;
}

flexi_tensor_error_t flexi_tensor_create_reshape_view(flexi_tensor_t* view, flexi_tensor_t* source, flexi_tuple_t shape) {
    flexi_tensor_error_t error;
    size_t source_count;
    size_t view_count = 1;
    size_t dim;
    if (source == NULL || shape._n_dims == 0 || shape._n_dims > 4) return FLEXI_TENSOR_ERR_UNKNOWN;
    error = flexi_tensor_element_count(source, &source_count);
    if (error != FLEXI_TENSOR_SUCCESS) return error;
    for (dim = 0; dim < shape._n_dims; ++dim) {
        if (shape._values[dim] != 0 && view_count > SIZE_MAX / shape._values[dim]) {
            errno = EOVERFLOW;
            return FLEXI_TENSOR_ERR_MISMATCH;
        }
        view_count *= shape._values[dim];
    }
    if (source_count != view_count) return FLEXI_TENSOR_ERR_MISMATCH;
    return flexi_tensor_create_view(view, source->_buffer, shape, source->_dtype);
}

static flexi_tensor_error_t flexi_tensor_slice_offset(const flexi_tensor_t* tensor, flexi_tuple_t index, size_t n_inner_dims, size_t* offset) {
    const size_t n_outer_dims = tensor == NULL || tensor->_n_dims < n_inner_dims ? 0 : tensor->_n_dims - n_inner_dims;
    size_t dim;
    size_t result = 0;
    if (tensor == NULL || offset == NULL || tensor->_n_dims < n_inner_dims || tensor->_shape == NULL || tensor->_strides == NULL || index._n_dims != n_outer_dims) return FLEXI_TENSOR_ERR_MISMATCH;
    for (dim = 0; dim < n_outer_dims; ++dim) {
        if (index._values[dim] >= tensor->_shape[dim]) return FLEXI_TENSOR_ERR_MISMATCH;
        result += index._values[dim] * tensor->_strides[dim];
    }
    *offset = result;
    return FLEXI_TENSOR_SUCCESS;
}

flexi_tensor_error_t flexi_tensor_create_matrix_view(fnblas_matrix_t* view, flexi_tensor_t* tensor, flexi_tuple_t index) {
    flexi_tensor_error_t error;
    fnblas_error_t blas_error;
    size_t offset;
    size_t element_size;
    if (view == NULL || tensor == NULL || tensor->_n_dims < 2) return FLEXI_TENSOR_ERR_MISMATCH;
    error = flexi_tensor_slice_offset(tensor, index, 2, &offset);
    if (error != FLEXI_TENSOR_SUCCESS) return error;
    element_size = _fnblas_dtype_unpacked_size_of(tensor->_dtype);
    if (element_size == 0) return FLEXI_TENSOR_ERR_MISMATCH;
    blas_error = fnblas_matrix_create_view(view, tensor->_buffer == NULL ? NULL : tensor->_buffer + offset * element_size, tensor->_shape[tensor->_n_dims - 2], tensor->_shape[tensor->_n_dims - 1], tensor->_dtype);
    return flexi_tensor_from_fnblas_error(blas_error);
}

flexi_tensor_error_t flexi_tensor_create_vector_view(fnblas_vector_t* view, flexi_tensor_t* tensor, flexi_tuple_t index) {
    flexi_tensor_error_t error;
    fnblas_error_t blas_error;
    size_t offset;
    size_t element_size;
    if (view == NULL || tensor == NULL || tensor->_n_dims < 1) return FLEXI_TENSOR_ERR_MISMATCH;
    error = flexi_tensor_slice_offset(tensor, index, 1, &offset);
    if (error != FLEXI_TENSOR_SUCCESS) return error;
    element_size = _fnblas_dtype_unpacked_size_of(tensor->_dtype);
    if (element_size == 0) return FLEXI_TENSOR_ERR_MISMATCH;
    blas_error = fnblas_vector_create_view(view, tensor->_buffer == NULL ? NULL : tensor->_buffer + offset * element_size, tensor->_shape[tensor->_n_dims - 1], tensor->_dtype);
    return flexi_tensor_from_fnblas_error(blas_error);
}

flexi_tensor_error_t flexi_tensor_destroy(flexi_tensor_t* tensor) {
    if (tensor == NULL) return FLEXI_TENSOR_ERR_UNKNOWN;
    if (_fnblas_buffer_is_allocated(tensor->_status) && !_fnblas_buffer_is_view(tensor->_status)) _fnblas_deallocate_buffer(tensor->_buffer);
    _fnblas_deallocate_buffer((byte_t*)tensor->_shape);
    _fnblas_deallocate_buffer((byte_t*)tensor->_strides);
    tensor->_buffer = NULL;
    tensor->_n_dims = 0;
    tensor->_shape = NULL;
    tensor->_strides = NULL;
    tensor->_dtype = FP32;
    tensor->_status = 0;
    return FLEXI_TENSOR_SUCCESS;
}

static flexi_tensor_error_t flexi_tensor_offset(const flexi_tensor_t* tensor, flexi_tuple_t index, size_t* offset) {
    size_t dim;
    size_t result = 0;
    if (tensor == NULL || offset == NULL || tensor->_n_dims == 0 || tensor->_n_dims > 4 || tensor->_shape == NULL || tensor->_strides == NULL || index._n_dims != tensor->_n_dims) return FLEXI_TENSOR_ERR_UNKNOWN;
    for (dim = 0; dim < tensor->_n_dims; ++dim) {
        if (index._values[dim] >= tensor->_shape[dim]) return FLEXI_TENSOR_ERR_MISMATCH;
        result += index._values[dim] * tensor->_strides[dim];
    }
    *offset = result;
    return FLEXI_TENSOR_SUCCESS;
}

flexi_tensor_error_t flexi_tensor_at(const flexi_tensor_t* tensor, flexi_tuple_t index, byte_t* out_element) {
    flexi_tensor_error_t error;
    size_t offset;
    size_t element_size;
    if (out_element == NULL) return FLEXI_TENSOR_ERR_UNKNOWN;
    error = flexi_tensor_offset(tensor, index, &offset);
    if (error != FLEXI_TENSOR_SUCCESS) return error;
    element_size = _fnblas_dtype_unpacked_size_of(tensor->_dtype);
    if (element_size == 0) return FLEXI_TENSOR_ERR_MISMATCH;
    memcpy(out_element, tensor->_buffer + offset * element_size, element_size);
    return FLEXI_TENSOR_SUCCESS;
}

flexi_tensor_error_t flexi_tensor_set(flexi_tensor_t* tensor, flexi_tuple_t index, const byte_t* element) {
    flexi_tensor_error_t error;
    size_t offset;
    size_t element_size;
    if (element == NULL) return FLEXI_TENSOR_ERR_UNKNOWN;
    error = flexi_tensor_offset(tensor, index, &offset);
    if (error != FLEXI_TENSOR_SUCCESS) return error;
    element_size = _fnblas_dtype_unpacked_size_of(tensor->_dtype);
    if (element_size == 0) return FLEXI_TENSOR_ERR_MISMATCH;
    memcpy(tensor->_buffer + offset * element_size, element, element_size);
    return FLEXI_TENSOR_SUCCESS;
}

flexi_tensor_error_t flexi_tensor_get_packed_buffer(const flexi_tensor_t* tensor, byte_t* out_buffer) {
    flexi_tensor_error_t error;
    size_t count;
    error = flexi_tensor_element_count(tensor, &count);
    if (error != FLEXI_TENSOR_SUCCESS) return error;
    if (count != 0 && out_buffer == NULL) return FLEXI_TENSOR_ERR_UNKNOWN;
    _fnblas_pack_buffer(tensor->_buffer, count, tensor->_dtype, out_buffer);
    return FLEXI_TENSOR_SUCCESS;
}

flexi_tensor_error_t flexi_tensor_initialize_from_packed_buffer(flexi_tensor_t* tensor, const byte_t* packed_buffer) {
    flexi_tensor_error_t error;
    size_t count;
    error = flexi_tensor_element_count(tensor, &count);
    if (error != FLEXI_TENSOR_SUCCESS) return error;
    if (count != 0 && packed_buffer == NULL) return FLEXI_TENSOR_ERR_UNKNOWN;
    _fnblas_unpack_buffer(tensor->_buffer, count, tensor->_dtype, packed_buffer);
    return FLEXI_TENSOR_SUCCESS;
}

flexi_tensor_error_t flexi_tensor_initialize_from_unpacked_buffer(flexi_tensor_t* tensor, const byte_t* unpacked_buffer) {
    flexi_tensor_error_t error;
    size_t count;
    size_t element_size;
    error = flexi_tensor_element_count(tensor, &count);
    if (error != FLEXI_TENSOR_SUCCESS) return error;
    if (count != 0 && unpacked_buffer == NULL) return FLEXI_TENSOR_ERR_UNKNOWN;
    element_size = _fnblas_dtype_unpacked_size_of(tensor->_dtype);
    if (element_size == 0) return FLEXI_TENSOR_ERR_MISMATCH;
    if (count != 0) memcpy(tensor->_buffer, unpacked_buffer, count * element_size);
    return FLEXI_TENSOR_SUCCESS;
}

size_t flexi_tensor_n_dims(const flexi_tensor_t* tensor) {
    if (tensor == NULL) {
        errno = EINVAL;
        return 0;
    }
    return tensor->_n_dims;
}

size_t flexi_tensor_dim_size(const flexi_tensor_t* tensor, size_t dim) {
    if (tensor == NULL || tensor->_shape == NULL || dim >= tensor->_n_dims) {
        errno = EINVAL;
        return 0;
    }
    return tensor->_shape[dim];
}

fnblas_dtype_t flexi_tensor_dtype(const flexi_tensor_t* tensor) {
    if (tensor == NULL) {
        errno = EINVAL;
        return INT8;
    }
    return tensor->_dtype;
}

size_t flexi_tensor_packed_buffer_size(const flexi_tensor_t* tensor) {
    flexi_tensor_error_t error;
    size_t count;
    error = flexi_tensor_element_count(tensor, &count);
    if (error != FLEXI_TENSOR_SUCCESS) return 0;
    return _fnblas_packed_buffer_size(count, tensor->_dtype);
}

size_t flexi_tensor_unpacked_buffer_size(const flexi_tensor_t* tensor) {
    flexi_tensor_error_t error;
    size_t count;
    size_t element_size;
    error = flexi_tensor_element_count(tensor, &count);
    if (error != FLEXI_TENSOR_SUCCESS) return 0;
    element_size = _fnblas_dtype_unpacked_size_of(tensor->_dtype);
    if (element_size == 0 || count > SIZE_MAX / element_size) {
        errno = EOVERFLOW;
        return 0;
    }
    return count * element_size;
}
