#ifndef FLEXI_TENSOR_COMMON_H
#define FLEXI_TENSOR_COMMON_H

#include <stddef.h>
#include <stdint.h>

#include "flexi_blas.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    FLEXI_TENSOR_SUCCESS = 0,
    FLEXI_TENSOR_ERR_MALLOC_FAIL,
    FLEXI_TENSOR_ERR_MISMATCH,
    FLEXI_TENSOR_ERR_UNSUPPORTED,
    FLEXI_TENSOR_ERR_BACKEND,
    FLEXI_TENSOR_ERR_UNKNOWN
} flexi_tensor_error_t;

typedef struct {
    size_t _n_dims;
    size_t _values[4];
} flexi_tuple_t;

typedef struct {
    byte_t* _buffer;    // pointer to the underlying data buffer (row-major order)
    size_t  _n_dims;    // number of dimensions
    size_t* _shape;     // size of each dimension
    size_t* _strides;   // memory layout strides for each dimension
    fnblas_dtype_t _dtype;     // data type of the tensor elements
    uint8_t _status;    // status flags for the tensor (same with the fnblas_vector_t and fnblas_matrix_t)
} flexi_tensor_t;

#define FLEXI_TENSOR_INITIALIZER { \
    ._buffer = NULL,               \
    ._n_dims = 0,                  \
    ._shape = NULL,                \
    ._strides = NULL,              \
    ._dtype = FP32,                \
    ._status = 0                   \
}

#ifdef __cplusplus
#define FLEXI_TUPLE_VALUE(n_dims, a, b, c, d) flexi_tuple_t{(n_dims), {(a), (b), (c), (d)}}
#else
#define FLEXI_TUPLE_VALUE(n_dims, a, b, c, d) ((flexi_tuple_t){(n_dims), {(a), (b), (c), (d)}})
#endif
#define FLEXI_TUPLE_EMPTY FLEXI_TUPLE_VALUE(0, 0, 0, 0, 0)
#define FLEXI_TUPLE_1(a) FLEXI_TUPLE_VALUE(1, a, 0, 0, 0)
#define FLEXI_TUPLE_2(a, b) FLEXI_TUPLE_VALUE(2, a, b, 0, 0)
#define FLEXI_TUPLE_3(a, b, c) FLEXI_TUPLE_VALUE(3, a, b, c, 0)
#define FLEXI_TUPLE_4(a, b, c, d) FLEXI_TUPLE_VALUE(4, a, b, c, d)
#define FLEXI_TUPLE_NARGS_IMPL(_1, _2, _3, _4, N, ...) N
#define FLEXI_TUPLE_NARGS(...) FLEXI_TUPLE_NARGS_IMPL(__VA_ARGS__, 4, 3, 2, 1)
#define FLEXI_TUPLE_CONCAT_IMPL(lhs, rhs) lhs##rhs
#define FLEXI_TUPLE_CONCAT(lhs, rhs) FLEXI_TUPLE_CONCAT_IMPL(lhs, rhs)
#define FLEXI_TUPLE_DISPATCH(count) FLEXI_TUPLE_CONCAT(FLEXI_TUPLE_, count)
#define FLEXI_TUPLE(...) FLEXI_TUPLE_DISPATCH(FLEXI_TUPLE_NARGS(__VA_ARGS__))(__VA_ARGS__)

flexi_tensor_error_t flexi_tensor_create(flexi_tensor_t* tensor, flexi_tuple_t shape, fnblas_dtype_t dtype);
flexi_tensor_error_t flexi_tensor_create_view(flexi_tensor_t* view, byte_t* buffer, flexi_tuple_t shape, fnblas_dtype_t dtype);
flexi_tensor_error_t flexi_tensor_create_reshape_view(flexi_tensor_t* view, flexi_tensor_t* source, flexi_tuple_t shape);
flexi_tensor_error_t flexi_tensor_create_matrix_view(fnblas_matrix_t* view, flexi_tensor_t* tensor, flexi_tuple_t index);
flexi_tensor_error_t flexi_tensor_create_vector_view(fnblas_vector_t* view, flexi_tensor_t* tensor, flexi_tuple_t index);
flexi_tensor_error_t flexi_tensor_destroy(flexi_tensor_t* tensor);

flexi_tensor_error_t flexi_tensor_at(const flexi_tensor_t* tensor, flexi_tuple_t index, byte_t* out_element);
flexi_tensor_error_t flexi_tensor_set(flexi_tensor_t* tensor, flexi_tuple_t index, const byte_t* element);
flexi_tensor_error_t flexi_tensor_get_packed_buffer(const flexi_tensor_t* tensor, byte_t* out_buffer);
flexi_tensor_error_t flexi_tensor_initialize_from_packed_buffer(flexi_tensor_t* tensor, const byte_t* packed_buffer);
flexi_tensor_error_t flexi_tensor_initialize_from_unpacked_buffer(flexi_tensor_t* tensor, const byte_t* unpacked_buffer);
size_t flexi_tensor_n_dims(const flexi_tensor_t* tensor);
size_t flexi_tensor_dim_size(const flexi_tensor_t* tensor, size_t dim);
fnblas_dtype_t flexi_tensor_dtype(const flexi_tensor_t* tensor);
size_t flexi_tensor_packed_buffer_size(const flexi_tensor_t* tensor);
size_t flexi_tensor_unpacked_buffer_size(const flexi_tensor_t* tensor);

#ifdef __cplusplus
}
#endif

#endif // FLEXI_TENSOR_COMMON_H
