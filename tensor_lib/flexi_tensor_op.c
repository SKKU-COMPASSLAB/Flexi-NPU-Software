#include "flexi_tensor_op.h"
#include "flexi_tensor_backend_internal.h"
#include "flexi_blas_internal.h"

#include <stddef.h>
#include <string.h>

static flexi_tensor_error_t flexi_tensor_op_from_fnblas_error(fnblas_error_t error) {
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

static int flexi_tensor_op_is_initialized(const flexi_tensor_t* tensor) {
    return tensor != NULL && (tensor->_shape != NULL || tensor->_strides != NULL || _fnblas_buffer_is_allocated(tensor->_status) || _fnblas_buffer_is_view(tensor->_status));
}

flexi_tensor_error_t flexi_tensor_op_linear(flexi_tensor_t* out, const flexi_tensor_t* x, const flexi_tensor_t* w, const flexi_tensor_t* b) {
    _flexi_tensor_backend_kernel_op_mm_matmul matmul_kernel = FLEXI_TENSOR_BACKEND_KERNEL_GET(_flexi_tensor_backend_kernel_op_mm_matmul);
    _flexi_tensor_backend_kernel_op_ve_add add_kernel = FLEXI_TENSOR_BACKEND_KERNEL_GET(_flexi_tensor_backend_kernel_op_ve_add);
    fnblas_matrix_t out_matrix = FNBLAS_MATRIX_INITIALIZER;
    fnblas_matrix_t x_matrix = FNBLAS_MATRIX_INITIALIZER;
    fnblas_matrix_t w_matrix = FNBLAS_MATRIX_INITIALIZER;
    fnblas_vector_t bias_vector = FNBLAS_VECTOR_INITIALIZER;
    flexi_tensor_error_t error;
    fnblas_error_t blas_error;
    size_t rows = 1;
    size_t dim;
    size_t row;
    size_t in_features;
    size_t out_features;
    int out_created = 0;
    flexi_tuple_t out_shape = FLEXI_TUPLE_EMPTY;
    if (out == NULL || x == NULL || w == NULL || out == x || out == w || out == b) return FLEXI_TENSOR_ERR_UNKNOWN;
    if (x->_n_dims == 0 || x->_n_dims > 4 || x->_shape == NULL || x->_strides == NULL || w->_n_dims != 2 || w->_shape == NULL || w->_strides == NULL) return FLEXI_TENSOR_ERR_MISMATCH;
    if (x->_dtype != w->_dtype || (b != NULL && x->_dtype != b->_dtype)) return FLEXI_TENSOR_ERR_MISMATCH;
    in_features = x->_shape[x->_n_dims - 1];
    out_features = w->_shape[0];
    if (in_features == 0 || out_features == 0 || w->_shape[1] != in_features) return FLEXI_TENSOR_ERR_MISMATCH;
    if (b != NULL && (b->_n_dims != 1 || b->_shape == NULL || b->_strides == NULL || b->_shape[0] != out_features)) return FLEXI_TENSOR_ERR_MISMATCH;
    for (dim = 0; dim + 1 < x->_n_dims; ++dim) {
        if (x->_shape[dim] == 0 || rows > SIZE_MAX / x->_shape[dim]) return FLEXI_TENSOR_ERR_MISMATCH;
        rows *= x->_shape[dim];
    }
    if (matmul_kernel == NULL || (b != NULL && add_kernel == NULL)) return FLEXI_TENSOR_ERR_UNSUPPORTED;
    if (flexi_tensor_op_is_initialized(out)) {
        if (out->_n_dims != x->_n_dims || out->_dtype != x->_dtype || out->_shape == NULL || out->_strides == NULL) return FLEXI_TENSOR_ERR_MISMATCH;
        for (dim = 0; dim + 1 < x->_n_dims; ++dim) {
            if (out->_shape[dim] != x->_shape[dim]) return FLEXI_TENSOR_ERR_MISMATCH;
        }
        if (out->_shape[x->_n_dims - 1] != out_features) return FLEXI_TENSOR_ERR_MISMATCH;
    } else {
        out_created = 1;
        out_shape._n_dims = x->_n_dims;
        for (dim = 0; dim + 1 < x->_n_dims; ++dim) out_shape._values[dim] = x->_shape[dim];
        out_shape._values[x->_n_dims - 1] = out_features;
        error = flexi_tensor_create(out, out_shape, x->_dtype);
        if (error != FLEXI_TENSOR_SUCCESS) return error;
    }
    if (out->_buffer == x->_buffer || out->_buffer == w->_buffer || (b != NULL && out->_buffer == b->_buffer)) {
        error = FLEXI_TENSOR_ERR_MISMATCH;
        goto _flexi_tensor_op_linear_cleanup;
    }
    blas_error = fnblas_matrix_create_view(&x_matrix, (byte_t*)x->_buffer, rows, in_features, x->_dtype);
    if (blas_error != FNBLAS_SUCCESS) {
        error = flexi_tensor_op_from_fnblas_error(blas_error);
        goto _flexi_tensor_op_linear_cleanup;
    }
    blas_error = fnblas_matrix_create_view(&w_matrix, (byte_t*)w->_buffer, out_features, in_features, w->_dtype);
    if (blas_error != FNBLAS_SUCCESS) {
        error = flexi_tensor_op_from_fnblas_error(blas_error);
        goto _flexi_tensor_op_linear_cleanup;
    }
    blas_error = fnblas_matrix_create_view(&out_matrix, out->_buffer, rows, out_features, out->_dtype);
    if (blas_error != FNBLAS_SUCCESS) {
        error = flexi_tensor_op_from_fnblas_error(blas_error);
        goto _flexi_tensor_op_linear_cleanup;
    }
    error = matmul_kernel(&out_matrix, &x_matrix, &w_matrix);
    if (error != FLEXI_TENSOR_SUCCESS) goto _flexi_tensor_op_linear_cleanup;
    if (b != NULL) {
        blas_error = fnblas_vector_create_view(&bias_vector, (byte_t*)b->_buffer, out_features, b->_dtype);
        if (blas_error != FNBLAS_SUCCESS) {
            error = flexi_tensor_op_from_fnblas_error(blas_error);
            goto _flexi_tensor_op_linear_cleanup;
        }
        for (row = 0; row < rows; ++row) {
            fnblas_vector_t result_row = FNBLAS_VECTOR_INITIALIZER;
            fnblas_vector_t input_row = FNBLAS_VECTOR_INITIALIZER;
            blas_error = fnblas_matrix_create_row_vector_view(&result_row, &out_matrix, row);
            if (blas_error == FNBLAS_SUCCESS) blas_error = fnblas_matrix_create_row_vector_view(&input_row, &out_matrix, row);
            if (blas_error != FNBLAS_SUCCESS) {
                fnblas_vector_destroy(&result_row);
                fnblas_vector_destroy(&input_row);
                error = flexi_tensor_op_from_fnblas_error(blas_error);
                goto _flexi_tensor_op_linear_cleanup;
            }
            error = add_kernel(&result_row, &input_row, &bias_vector);
            fnblas_vector_destroy(&result_row);
            fnblas_vector_destroy(&input_row);
            if (error != FLEXI_TENSOR_SUCCESS) goto _flexi_tensor_op_linear_cleanup;
        }
    }
    error = FLEXI_TENSOR_SUCCESS;

_flexi_tensor_op_linear_cleanup:
    fnblas_vector_destroy(&bias_vector);
    fnblas_matrix_destroy(&out_matrix);
    fnblas_matrix_destroy(&w_matrix);
    fnblas_matrix_destroy(&x_matrix);
    if (error != FLEXI_TENSOR_SUCCESS && out_created) flexi_tensor_destroy(out);
    return error;
}

flexi_tensor_error_t flexi_tensor_op_relu(flexi_tensor_t* out, const flexi_tensor_t* x) {
    _flexi_tensor_backend_kernel_op_sv_max max_kernel = FLEXI_TENSOR_BACKEND_KERNEL_GET(_flexi_tensor_backend_kernel_op_sv_max);
    fnblas_vector_t out_vector = FNBLAS_VECTOR_INITIALIZER;
    fnblas_vector_t x_vector = FNBLAS_VECTOR_INITIALIZER;
    fnblas_scalar_t zero;
    flexi_tuple_t out_shape = FLEXI_TUPLE_EMPTY;
    flexi_tensor_error_t error;
    fnblas_error_t blas_error;
    size_t n_elements = 1;
    size_t dim;
    int out_created = 0;
    if (out == NULL || x == NULL || out == x) return FLEXI_TENSOR_ERR_UNKNOWN;
    if (x->_n_dims == 0 || x->_n_dims > 4 || x->_shape == NULL || x->_strides == NULL || !_fnblas_dtype_is_valid(x->_dtype)) return FLEXI_TENSOR_ERR_MISMATCH;
    for (dim = 0; dim < x->_n_dims; ++dim) {
        if (x->_shape[dim] != 0 && n_elements > SIZE_MAX / x->_shape[dim]) return FLEXI_TENSOR_ERR_MISMATCH;
        n_elements *= x->_shape[dim];
    }
    if (max_kernel == NULL) return FLEXI_TENSOR_ERR_UNSUPPORTED;
    if (flexi_tensor_op_is_initialized(out)) {
        if (out->_n_dims != x->_n_dims || out->_dtype != x->_dtype || out->_shape == NULL || out->_strides == NULL) return FLEXI_TENSOR_ERR_MISMATCH;
        for (dim = 0; dim < x->_n_dims; ++dim) {
            if (out->_shape[dim] != x->_shape[dim]) return FLEXI_TENSOR_ERR_MISMATCH;
        }
    } else {
        out_created = 1;
        out_shape._n_dims = x->_n_dims;
        for (dim = 0; dim < x->_n_dims; ++dim) out_shape._values[dim] = x->_shape[dim];
        error = flexi_tensor_create(out, out_shape, x->_dtype);
        if (error != FLEXI_TENSOR_SUCCESS) return error;
    }
    if (out->_buffer == x->_buffer && n_elements != 0) {
        error = FLEXI_TENSOR_ERR_MISMATCH;
        goto _flexi_tensor_op_relu_cleanup;
    }
    if (n_elements == 0) return FLEXI_TENSOR_SUCCESS;
    blas_error = fnblas_scalar_create_from_dtype(&zero, x->_dtype);
    if (blas_error != FNBLAS_SUCCESS) {
        error = flexi_tensor_op_from_fnblas_error(blas_error);
        goto _flexi_tensor_op_relu_cleanup;
    }
    blas_error = fnblas_vector_create_view(&x_vector, (byte_t*)x->_buffer, n_elements, x->_dtype);
    if (blas_error != FNBLAS_SUCCESS) {
        error = flexi_tensor_op_from_fnblas_error(blas_error);
        goto _flexi_tensor_op_relu_cleanup;
    }
    blas_error = fnblas_vector_create_view(&out_vector, out->_buffer, n_elements, out->_dtype);
    if (blas_error != FNBLAS_SUCCESS) {
        error = flexi_tensor_op_from_fnblas_error(blas_error);
        goto _flexi_tensor_op_relu_cleanup;
    }
    error = max_kernel(&out_vector, &x_vector, &zero);

_flexi_tensor_op_relu_cleanup:
    fnblas_vector_destroy(&out_vector);
    fnblas_vector_destroy(&x_vector);
    if (error != FLEXI_TENSOR_SUCCESS && out_created) flexi_tensor_destroy(out);
    return error;
}

flexi_tensor_error_t flexi_tensor_op_conv2d(flexi_tensor_t* out, const flexi_tensor_t* x, const flexi_tensor_t* w, const flexi_tensor_t* b, const size_t stride, const size_t padding, const size_t dilation) {
    _flexi_tensor_backend_kernel_op_mm_matmul matmul_kernel = FLEXI_TENSOR_BACKEND_KERNEL_GET(_flexi_tensor_backend_kernel_op_mm_matmul);
    _flexi_tensor_backend_kernel_op_me_add add_partial_kernel = FLEXI_TENSOR_BACKEND_KERNEL_GET(_flexi_tensor_backend_kernel_op_me_add);
    _flexi_tensor_backend_kernel_op_ve_add add_bias_kernel = FLEXI_TENSOR_BACKEND_KERNEL_GET(_flexi_tensor_backend_kernel_op_ve_add);
    flexi_tensor_t padded_x = FLEXI_TENSOR_INITIALIZER;
    fnblas_matrix_t gathered_x = FNBLAS_MATRIX_INITIALIZER;
    fnblas_matrix_t x_view = FNBLAS_MATRIX_INITIALIZER;
    fnblas_matrix_t w_view = FNBLAS_MATRIX_INITIALIZER;
    fnblas_matrix_t partial_sum = FNBLAS_MATRIX_INITIALIZER;
    fnblas_matrix_t out_result_view = FNBLAS_MATRIX_INITIALIZER;
    fnblas_matrix_t out_input_view = FNBLAS_MATRIX_INITIALIZER;
    fnblas_matrix_t out_flat_view = FNBLAS_MATRIX_INITIALIZER;
    fnblas_vector_t bias_view = FNBLAS_VECTOR_INITIALIZER;
    fnblas_vector_t out_row_result = FNBLAS_VECTOR_INITIALIZER;
    fnblas_vector_t out_row_input = FNBLAS_VECTOR_INITIALIZER;
    flexi_tensor_error_t error;
    fnblas_error_t blas_error;
    flexi_tuple_t padded_shape;
    flexi_tuple_t out_shape;
    size_t n_batches;
    size_t input_height;
    size_t input_width;
    size_t channels;
    size_t filter_height;
    size_t filter_width;
    size_t output_channels;
    size_t padded_height;
    size_t padded_width;
    size_t effective_filter_height;
    size_t effective_filter_width;
    size_t output_height;
    size_t output_width;
    size_t element_size;
    size_t channel_bytes;
    size_t output_size;
    size_t batch;
    size_t input_row;
    size_t input_col;
    size_t output_row;
    size_t output_col;
    size_t filter_row;
    size_t filter_col;
    size_t flat_row;
    int out_created = 0;

    /* Validate NHWC input, (FH, FW, K, C) weights, and optional K-element bias. */
    if (out == NULL || x == NULL || w == NULL || out == x || out == w || out == b) return FLEXI_TENSOR_ERR_UNKNOWN;
    if (stride == 0 || dilation == 0) return FLEXI_TENSOR_ERR_MISMATCH;
    if (x->_n_dims != 4 || w->_n_dims != 4 || x->_shape == NULL || x->_strides == NULL || w->_shape == NULL || w->_strides == NULL) return FLEXI_TENSOR_ERR_MISMATCH;
    if (x->_dtype != w->_dtype || (b != NULL && x->_dtype != b->_dtype)) return FLEXI_TENSOR_ERR_MISMATCH;

    n_batches = x->_shape[0];
    input_height = x->_shape[1];
    input_width = x->_shape[2];
    channels = x->_shape[3];
    filter_height = w->_shape[0];
    filter_width = w->_shape[1];
    output_channels = w->_shape[2];
    if (channels == 0 || filter_height == 0 || filter_width == 0 || output_channels == 0 || w->_shape[3] != channels) return FLEXI_TENSOR_ERR_MISMATCH;
    if (b != NULL && (b->_n_dims != 1 || b->_shape == NULL || b->_strides == NULL || b->_shape[0] != output_channels)) return FLEXI_TENSOR_ERR_MISMATCH;

    if (padding > (SIZE_MAX - input_height) / 2 || padding > (SIZE_MAX - input_width) / 2) return FLEXI_TENSOR_ERR_MISMATCH;
    padded_height = input_height + 2 * padding;
    padded_width = input_width + 2 * padding;
    if (filter_height - 1 > (SIZE_MAX - 1) / dilation || filter_width - 1 > (SIZE_MAX - 1) / dilation) return FLEXI_TENSOR_ERR_MISMATCH;
    effective_filter_height = dilation * (filter_height - 1) + 1;
    effective_filter_width = dilation * (filter_width - 1) + 1;
    if (padded_height < effective_filter_height || padded_width < effective_filter_width) return FLEXI_TENSOR_ERR_MISMATCH;
    output_height = (padded_height - effective_filter_height) / stride + 1;
    output_width = (padded_width - effective_filter_width) / stride + 1;

    /* Create or validate the NHWK output tensor. */
    if (matmul_kernel == NULL || add_partial_kernel == NULL || (b != NULL && add_bias_kernel == NULL)) return FLEXI_TENSOR_ERR_UNSUPPORTED;
    padded_shape = FLEXI_TUPLE(n_batches, padded_height, padded_width, channels);
    out_shape = FLEXI_TUPLE(n_batches, output_height, output_width, output_channels);

    if (flexi_tensor_op_is_initialized(out)) {
        if (out->_n_dims != 4 || out->_dtype != x->_dtype || out->_shape == NULL || out->_strides == NULL) return FLEXI_TENSOR_ERR_MISMATCH;
        if (out->_shape[0] != n_batches || out->_shape[1] != output_height || out->_shape[2] != output_width || out->_shape[3] != output_channels) return FLEXI_TENSOR_ERR_MISMATCH;
    } else {
        out_created = 1;
        error = flexi_tensor_create(out, out_shape, x->_dtype);
        if (error != FLEXI_TENSOR_SUCCESS) return error;
    }
    if ((out->_buffer == x->_buffer || out->_buffer == w->_buffer || (b != NULL && out->_buffer == b->_buffer)) && n_batches != 0) {
        error = FLEXI_TENSOR_ERR_MISMATCH;
        goto _flexi_tensor_op_conv2d_cleanup;
    }

    /* Materialize the zero-padded NHWC input and copy x into its center. */
    error = flexi_tensor_create(&padded_x, padded_shape, x->_dtype);
    if (error != FLEXI_TENSOR_SUCCESS) goto _flexi_tensor_op_conv2d_cleanup;
    element_size = _fnblas_dtype_unpacked_size_of(x->_dtype);
    if (channels > SIZE_MAX / element_size) {
        error = FLEXI_TENSOR_ERR_MISMATCH;
        goto _flexi_tensor_op_conv2d_cleanup;
    }
    channel_bytes = channels * element_size;

    for (batch = 0; batch < n_batches; ++batch) {
        for (input_row = 0; input_row < input_height; ++input_row) {
            for (input_col = 0; input_col < input_width; ++input_col) {
                const size_t source_offset = batch * x->_strides[0] + input_row * x->_strides[1] + input_col * x->_strides[2];
                const size_t destination_offset = batch * padded_x._strides[0] + (input_row + padding) * padded_x._strides[1] + (input_col + padding) * padded_x._strides[2];
                memcpy(padded_x._buffer + destination_offset * element_size, x->_buffer + source_offset * element_size, channel_bytes);
            }
        }
    }

    output_size = flexi_tensor_unpacked_buffer_size(out);
    if (output_size != 0) memset(out->_buffer, 0, output_size);

    /* Reuse one (OW, K) partial-sum matrix for every filter position. */
    blas_error = fnblas_matrix_create(&partial_sum, output_width, output_channels, x->_dtype);
    if (blas_error != FNBLAS_SUCCESS) {
        error = flexi_tensor_op_from_fnblas_error(blas_error);
        goto _flexi_tensor_op_conv2d_cleanup;
    }
    if (stride != 1) {
        blas_error = fnblas_matrix_create(&gathered_x, output_width, channels, x->_dtype);
        if (blas_error != FNBLAS_SUCCESS) {
            error = flexi_tensor_op_from_fnblas_error(blas_error);
            goto _flexi_tensor_op_conv2d_cleanup;
        }
    }

    /*
     * Each (n, oh) output slice is an (OW, K) matrix. For every (fh, fw),
     * multiply the corresponding input (OW, C) matrix by the weight (K, C)
     * matrix and accumulate the resulting partial sum.
     */
    for (batch = 0; batch < n_batches; ++batch) {
        for (output_row = 0; output_row < output_height; ++output_row) {
            error = flexi_tensor_create_matrix_view(&out_result_view, out, FLEXI_TUPLE(batch, output_row));
            if (error == FLEXI_TENSOR_SUCCESS) error = flexi_tensor_create_matrix_view(&out_input_view, out, FLEXI_TUPLE(batch, output_row));
            if (error != FLEXI_TENSOR_SUCCESS) goto _flexi_tensor_op_conv2d_cleanup;

            for (filter_row = 0; filter_row < filter_height; ++filter_row) {
                input_row = output_row * stride + filter_row * dilation;
                for (filter_col = 0; filter_col < filter_width; ++filter_col) {
                    const size_t first_input_col = filter_col * dilation;
                    const size_t first_input_offset = batch * padded_x._strides[0] + input_row * padded_x._strides[1] + first_input_col * padded_x._strides[2];
                    error = flexi_tensor_create_matrix_view(&w_view, (flexi_tensor_t*)w, FLEXI_TUPLE(filter_row, filter_col));
                    if (error != FLEXI_TENSOR_SUCCESS) goto _flexi_tensor_op_conv2d_cleanup;

                    if (stride == 1) {
                        blas_error = fnblas_matrix_create_view(&x_view, padded_x._buffer + first_input_offset * element_size, output_width, channels, x->_dtype);
                        if (blas_error != FNBLAS_SUCCESS) {
                            error = flexi_tensor_op_from_fnblas_error(blas_error);
                            goto _flexi_tensor_op_conv2d_cleanup;
                        }
                        error = matmul_kernel(&partial_sum, &x_view, &w_view);
                    } else {
                        for (output_col = 0; output_col < output_width; ++output_col) {
                            const size_t gathered_input_col = output_col * stride + first_input_col;
                            const size_t gathered_input_offset = batch * padded_x._strides[0] + input_row * padded_x._strides[1] + gathered_input_col * padded_x._strides[2];
                            memcpy(gathered_x._buffer + output_col * channel_bytes, padded_x._buffer + gathered_input_offset * element_size, channel_bytes);
                        }
                        error = matmul_kernel(&partial_sum, &gathered_x, &w_view);
                    }
                    if (error == FLEXI_TENSOR_SUCCESS) error = add_partial_kernel(&out_result_view, &out_input_view, &partial_sum);
                    fnblas_matrix_destroy(&x_view);
                    fnblas_matrix_destroy(&w_view);
                    if (error != FLEXI_TENSOR_SUCCESS) goto _flexi_tensor_op_conv2d_cleanup;
                }
            }
            fnblas_matrix_destroy(&out_input_view);
            fnblas_matrix_destroy(&out_result_view);
        }
    }

    /* Broadcast the optional K-element bias over every output position. */
    if (b != NULL && n_batches != 0) {
        if (n_batches > SIZE_MAX / output_height || n_batches * output_height > SIZE_MAX / output_width) {
            error = FLEXI_TENSOR_ERR_MISMATCH;
            goto _flexi_tensor_op_conv2d_cleanup;
        }
        flat_row = n_batches * output_height * output_width;
        blas_error = fnblas_matrix_create_view(&out_flat_view, out->_buffer, flat_row, output_channels, out->_dtype);
        if (blas_error == FNBLAS_SUCCESS) blas_error = fnblas_vector_create_view(&bias_view, (byte_t*)b->_buffer, output_channels, b->_dtype);
        if (blas_error != FNBLAS_SUCCESS) {
            error = flexi_tensor_op_from_fnblas_error(blas_error);
            goto _flexi_tensor_op_conv2d_cleanup;
        }
        for (output_row = 0; output_row < flat_row; ++output_row) {
            blas_error = fnblas_matrix_create_row_vector_view(&out_row_result, &out_flat_view, output_row);
            if (blas_error == FNBLAS_SUCCESS) blas_error = fnblas_matrix_create_row_vector_view(&out_row_input, &out_flat_view, output_row);
            if (blas_error != FNBLAS_SUCCESS) {
                error = flexi_tensor_op_from_fnblas_error(blas_error);
                goto _flexi_tensor_op_conv2d_cleanup;
            }
            error = add_bias_kernel(&out_row_result, &out_row_input, &bias_view);
            fnblas_vector_destroy(&out_row_input);
            fnblas_vector_destroy(&out_row_result);
            if (error != FLEXI_TENSOR_SUCCESS) goto _flexi_tensor_op_conv2d_cleanup;
        }
    }
    error = FLEXI_TENSOR_SUCCESS;

_flexi_tensor_op_conv2d_cleanup:
    fnblas_vector_destroy(&out_row_input);
    fnblas_vector_destroy(&out_row_result);
    fnblas_vector_destroy(&bias_view);
    fnblas_matrix_destroy(&out_flat_view);
    fnblas_matrix_destroy(&out_input_view);
    fnblas_matrix_destroy(&out_result_view);
    fnblas_matrix_destroy(&partial_sum);
    fnblas_matrix_destroy(&w_view);
    fnblas_matrix_destroy(&x_view);
    fnblas_matrix_destroy(&gathered_x);
    flexi_tensor_destroy(&padded_x);
    if (error != FLEXI_TENSOR_SUCCESS && out_created) flexi_tensor_destroy(out);
    return error;
}

typedef enum {
    FLEXI_TENSOR_POOL_REDUCE_MAX,
    FLEXI_TENSOR_POOL_REDUCE_AVG
} flexi_tensor_pool_reduction_t;

static flexi_tensor_error_t flexi_tensor_op_pool2d(flexi_tensor_t* out, const flexi_tensor_t* x, size_t kernel_size, size_t stride, size_t padding, flexi_tensor_pool_reduction_t reduction) {
    _flexi_tensor_backend_kernel_op_ve_add add_kernel = FLEXI_TENSOR_BACKEND_KERNEL_GET(_flexi_tensor_backend_kernel_op_ve_add);
    _flexi_tensor_backend_kernel_op_ve_div div_kernel = FLEXI_TENSOR_BACKEND_KERNEL_GET(_flexi_tensor_backend_kernel_op_ve_div);
    _flexi_tensor_backend_kernel_op_ve_max max_kernel = FLEXI_TENSOR_BACKEND_KERNEL_GET(_flexi_tensor_backend_kernel_op_ve_max);
    fnblas_vector_t zero_vector = FNBLAS_VECTOR_INITIALIZER;
    fnblas_vector_t divisor_vector = FNBLAS_VECTOR_INITIALIZER;
    flexi_tensor_error_t error;
    fnblas_error_t blas_error;
    flexi_tuple_t out_shape;
    size_t n_batches;
    size_t input_height;
    size_t input_width;
    size_t channels;
    size_t padded_height;
    size_t padded_width;
    size_t output_height;
    size_t output_width;
    size_t window_area;
    size_t batch;
    size_t output_row;
    size_t output_col;
    size_t kernel_row;
    size_t kernel_col;
    size_t element_size;
    size_t channel;
    int out_created = 0;

    if (out == NULL || x == NULL || out == x) return FLEXI_TENSOR_ERR_UNKNOWN;
    if (kernel_size == 0 || stride == 0) return FLEXI_TENSOR_ERR_MISMATCH;
    if (reduction != FLEXI_TENSOR_POOL_REDUCE_MAX && reduction != FLEXI_TENSOR_POOL_REDUCE_AVG) return FLEXI_TENSOR_ERR_UNKNOWN;
    if (x->_n_dims != 4 || x->_shape == NULL || x->_strides == NULL || !_fnblas_dtype_is_valid(x->_dtype)) return FLEXI_TENSOR_ERR_MISMATCH;

    n_batches = x->_shape[0];
    input_height = x->_shape[1];
    input_width = x->_shape[2];
    channels = x->_shape[3];
    if (channels == 0) return FLEXI_TENSOR_ERR_MISMATCH;
    if (padding > (SIZE_MAX - input_height) / 2 || padding > (SIZE_MAX - input_width) / 2) return FLEXI_TENSOR_ERR_MISMATCH;
    padded_height = input_height + 2 * padding;
    padded_width = input_width + 2 * padding;
    if (padded_height < kernel_size || padded_width < kernel_size) return FLEXI_TENSOR_ERR_MISMATCH;
    if (kernel_size > SIZE_MAX / kernel_size) return FLEXI_TENSOR_ERR_MISMATCH;
    window_area = kernel_size * kernel_size;
    output_height = (padded_height - kernel_size) / stride + 1;
    output_width = (padded_width - kernel_size) / stride + 1;
    element_size = fnblas_dtype_size_of(x->_dtype);
    if (element_size == 0 || channels > SIZE_MAX / element_size) return FLEXI_TENSOR_ERR_MISMATCH;
    if (reduction == FLEXI_TENSOR_POOL_REDUCE_MAX && max_kernel == NULL) return FLEXI_TENSOR_ERR_UNSUPPORTED;
    if (reduction == FLEXI_TENSOR_POOL_REDUCE_AVG && (add_kernel == NULL || div_kernel == NULL)) return FLEXI_TENSOR_ERR_UNSUPPORTED;
    if (reduction == FLEXI_TENSOR_POOL_REDUCE_AVG && x->_dtype == INT8 && window_area > INT8_MAX) return FLEXI_TENSOR_ERR_MISMATCH;
    if (reduction == FLEXI_TENSOR_POOL_REDUCE_AVG && !_fnblas_dtype_is_float(x->_dtype) && x->_dtype != INT8 && window_area > INT32_MAX) return FLEXI_TENSOR_ERR_MISMATCH;
    out_shape = FLEXI_TUPLE(n_batches, output_height, output_width, channels);

    if (flexi_tensor_op_is_initialized(out)) {
        if (out->_n_dims != 4 || out->_dtype != x->_dtype || out->_shape == NULL || out->_strides == NULL) return FLEXI_TENSOR_ERR_MISMATCH;
        if (out->_shape[0] != n_batches || out->_shape[1] != output_height || out->_shape[2] != output_width || out->_shape[3] != channels) return FLEXI_TENSOR_ERR_MISMATCH;
    } else {
        out_created = 1;
        error = flexi_tensor_create(out, out_shape, x->_dtype);
        if (error != FLEXI_TENSOR_SUCCESS) return error;
    }
    if (out->_buffer == x->_buffer && n_batches != 0) {
        if (out_created) flexi_tensor_destroy(out);
        return FLEXI_TENSOR_ERR_MISMATCH;
    }

    error = FLEXI_TENSOR_SUCCESS;
    if (reduction == FLEXI_TENSOR_POOL_REDUCE_MAX) {
        blas_error = fnblas_vector_create(&zero_vector, channels, x->_dtype);
        if (blas_error != FNBLAS_SUCCESS) {
            error = flexi_tensor_op_from_fnblas_error(blas_error);
            goto _flexi_tensor_op_pool2d_cleanup;
        }
        memset(zero_vector._buffer, 0, channels * element_size);
    } else {
        blas_error = fnblas_vector_create(&divisor_vector, channels, x->_dtype);
        if (blas_error != FNBLAS_SUCCESS) {
            error = flexi_tensor_op_from_fnblas_error(blas_error);
            goto _flexi_tensor_op_pool2d_cleanup;
        }
        if (_fnblas_dtype_is_float(x->_dtype)) {
            float* divisor = (float*)divisor_vector._buffer;
            for (channel = 0; channel < channels; ++channel) divisor[channel] = (float)window_area;
        } else if (x->_dtype == INT8) {
            memset(divisor_vector._buffer, (int8_t)window_area, channels);
        } else {
            int32_t* divisor = (int32_t*)divisor_vector._buffer;
            for (channel = 0; channel < channels; ++channel) divisor[channel] = (int32_t)window_area;
        }
    }

    for (batch = 0; batch < n_batches; ++batch) {
        for (output_row = 0; output_row < output_height; ++output_row) {
            for (output_col = 0; output_col < output_width; ++output_col) {
                fnblas_vector_t out_vector = FNBLAS_VECTOR_INITIALIZER;
                fnblas_vector_t accumulator_vector = FNBLAS_VECTOR_INITIALIZER;
                int initialized = 0;
                const size_t window_row = output_row * stride;
                const size_t window_col = output_col * stride;
                const int window_has_padding = window_row < padding || window_col < padding || window_row + kernel_size > padding + input_height || window_col + kernel_size > padding + input_width;
                const size_t destination_offset = batch * out->_strides[0] + output_row * out->_strides[1] + output_col * out->_strides[2];

                blas_error = fnblas_vector_create_view(&out_vector, out->_buffer + destination_offset * element_size, channels, out->_dtype);
                if (blas_error != FNBLAS_SUCCESS) {
                    error = flexi_tensor_op_from_fnblas_error(blas_error);
                    goto _flexi_tensor_op_pool2d_cleanup;
                }
                blas_error = fnblas_vector_create_view(&accumulator_vector, out_vector._buffer, channels, out->_dtype);
                if (blas_error != FNBLAS_SUCCESS) {
                    fnblas_vector_destroy(&out_vector);
                    error = flexi_tensor_op_from_fnblas_error(blas_error);
                    goto _flexi_tensor_op_pool2d_cleanup;
                }
                if (reduction == FLEXI_TENSOR_POOL_REDUCE_AVG) {
                    memset(out_vector._buffer, 0, channels * element_size);
                    initialized = 1;
                } else if (window_has_padding) {
                    memcpy(out_vector._buffer, zero_vector._buffer, channels * element_size);
                    initialized = 1;
                }

                for (kernel_row = 0; kernel_row < kernel_size; ++kernel_row) {
                    const size_t padded_row = window_row + kernel_row;
                    const int row_is_padding = padded_row < padding || padded_row - padding >= input_height;
                    if (row_is_padding) continue;
                    for (kernel_col = 0; kernel_col < kernel_size; ++kernel_col) {
                        fnblas_vector_t input_vector = FNBLAS_VECTOR_INITIALIZER;
                        const size_t padded_col = window_col + kernel_col;
                        const int col_is_padding = padded_col < padding || padded_col - padding >= input_width;
                        const size_t source_row = padded_row - padding;
                        size_t source_col;
                        size_t source_offset;

                        if (col_is_padding) continue;
                        source_col = padded_col - padding;
                        source_offset = batch * x->_strides[0] + source_row * x->_strides[1] + source_col * x->_strides[2];
                        blas_error = fnblas_vector_create_view(&input_vector, x->_buffer + source_offset * element_size, channels, x->_dtype);
                        if (blas_error != FNBLAS_SUCCESS) {
                            fnblas_vector_destroy(&accumulator_vector);
                            fnblas_vector_destroy(&out_vector);
                            error = flexi_tensor_op_from_fnblas_error(blas_error);
                            goto _flexi_tensor_op_pool2d_cleanup;
                        }
                        if (!initialized) {
                            memcpy(out_vector._buffer, input_vector._buffer, channels * element_size);
                            initialized = 1;
                        } else if (reduction == FLEXI_TENSOR_POOL_REDUCE_MAX) {
                            error = max_kernel(&out_vector, &accumulator_vector, &input_vector);
                        } else {
                            error = add_kernel(&out_vector, &accumulator_vector, &input_vector);
                        }
                        fnblas_vector_destroy(&input_vector);
                        if (error != FLEXI_TENSOR_SUCCESS) {
                            fnblas_vector_destroy(&accumulator_vector);
                            fnblas_vector_destroy(&out_vector);
                            goto _flexi_tensor_op_pool2d_cleanup;
                        }
                    }
                }
                if (reduction == FLEXI_TENSOR_POOL_REDUCE_AVG) error = div_kernel(&out_vector, &accumulator_vector, &divisor_vector);
                fnblas_vector_destroy(&accumulator_vector);
                fnblas_vector_destroy(&out_vector);
                if (error != FLEXI_TENSOR_SUCCESS) goto _flexi_tensor_op_pool2d_cleanup;
            }
        }
    }
    error = FLEXI_TENSOR_SUCCESS;

_flexi_tensor_op_pool2d_cleanup:
    fnblas_vector_destroy(&divisor_vector);
    fnblas_vector_destroy(&zero_vector);
    if (error != FLEXI_TENSOR_SUCCESS && out_created) flexi_tensor_destroy(out);
    return error;
}

flexi_tensor_error_t flexi_tensor_op_maxpool2d(flexi_tensor_t* out, const flexi_tensor_t* x, const size_t kernel_size, const size_t stride, const size_t padding) {
    return flexi_tensor_op_pool2d(out, x, kernel_size, stride, padding, FLEXI_TENSOR_POOL_REDUCE_MAX);
}

flexi_tensor_error_t flexi_tensor_op_avgpool2d(flexi_tensor_t* out, const flexi_tensor_t* x, const size_t kernel_size, const size_t stride, const size_t padding) {
    return flexi_tensor_op_pool2d(out, x, kernel_size, stride, padding, FLEXI_TENSOR_POOL_REDUCE_AVG);
}

flexi_tensor_error_t flexi_tensor_op_softmax(flexi_tensor_t* out, const flexi_tensor_t* x) {
    _flexi_tensor_backend_kernel_op_sv_sub subtract_kernel = FLEXI_TENSOR_BACKEND_KERNEL_GET(_flexi_tensor_backend_kernel_op_sv_sub);
    _flexi_tensor_backend_kernel_op_ve_div divide_kernel = FLEXI_TENSOR_BACKEND_KERNEL_GET(_flexi_tensor_backend_kernel_op_ve_div);
    _flexi_tensor_backend_kernel_op_vr_sum sum_kernel = FLEXI_TENSOR_BACKEND_KERNEL_GET(_flexi_tensor_backend_kernel_op_vr_sum);
    _flexi_tensor_backend_kernel_op_vr_max max_kernel = FLEXI_TENSOR_BACKEND_KERNEL_GET(_flexi_tensor_backend_kernel_op_vr_max);
    _flexi_tensor_backend_kernel_op_vi_exp exp_kernel = FLEXI_TENSOR_BACKEND_KERNEL_GET(_flexi_tensor_backend_kernel_op_vi_exp);
    fnblas_matrix_t x_matrix = FNBLAS_MATRIX_INITIALIZER;
    fnblas_matrix_t out_matrix = FNBLAS_MATRIX_INITIALIZER;
    fnblas_vector_t shifted_vector = FNBLAS_VECTOR_INITIALIZER;
    fnblas_vector_t divisor_vector = FNBLAS_VECTOR_INITIALIZER;
    flexi_tuple_t out_shape = FLEXI_TUPLE_EMPTY;
    flexi_tensor_error_t error;
    fnblas_error_t blas_error;
    size_t rows = 1;
    size_t columns;
    size_t dimension;
    size_t row;
    size_t column;
    int out_created = 0;

    if (out == NULL || x == NULL || out == x) return FLEXI_TENSOR_ERR_UNKNOWN;
    if (x->_n_dims == 0 || x->_n_dims > 4 || x->_shape == NULL || x->_strides == NULL || !_fnblas_dtype_is_valid(x->_dtype)) return FLEXI_TENSOR_ERR_MISMATCH;
    if (!_fnblas_dtype_is_float(x->_dtype)) return FLEXI_TENSOR_ERR_UNSUPPORTED;
    columns = x->_shape[x->_n_dims - 1];
    if (columns == 0) return FLEXI_TENSOR_ERR_MISMATCH;
    for (dimension = 0; dimension + 1 < x->_n_dims; ++dimension) {
        if (x->_shape[dimension] != 0 && rows > SIZE_MAX / x->_shape[dimension]) return FLEXI_TENSOR_ERR_MISMATCH;
        rows *= x->_shape[dimension];
    }
    if (subtract_kernel == NULL || divide_kernel == NULL || sum_kernel == NULL || max_kernel == NULL || exp_kernel == NULL) return FLEXI_TENSOR_ERR_UNSUPPORTED;

    if (flexi_tensor_op_is_initialized(out)) {
        if (out->_n_dims != x->_n_dims || out->_dtype != x->_dtype || out->_shape == NULL || out->_strides == NULL) return FLEXI_TENSOR_ERR_MISMATCH;
        for (dimension = 0; dimension < x->_n_dims; ++dimension) {
            if (out->_shape[dimension] != x->_shape[dimension]) return FLEXI_TENSOR_ERR_MISMATCH;
        }
    } else {
        out_created = 1;
        out_shape._n_dims = x->_n_dims;
        for (dimension = 0; dimension < x->_n_dims; ++dimension) out_shape._values[dimension] = x->_shape[dimension];
        error = flexi_tensor_create(out, out_shape, x->_dtype);
        if (error != FLEXI_TENSOR_SUCCESS) return error;
    }
    if (out->_buffer == x->_buffer && rows != 0) {
        error = FLEXI_TENSOR_ERR_MISMATCH;
        goto _flexi_tensor_op_softmax_cleanup;
    }
    if (rows == 0) {
        error = FLEXI_TENSOR_SUCCESS;
        goto _flexi_tensor_op_softmax_cleanup;
    }

    blas_error = fnblas_matrix_create_view(&x_matrix, (byte_t*)x->_buffer, rows, columns, x->_dtype);
    if (blas_error != FNBLAS_SUCCESS) {
        error = flexi_tensor_op_from_fnblas_error(blas_error);
        goto _flexi_tensor_op_softmax_cleanup;
    }
    blas_error = fnblas_matrix_create_view(&out_matrix, out->_buffer, rows, columns, out->_dtype);
    if (blas_error != FNBLAS_SUCCESS) {
        error = flexi_tensor_op_from_fnblas_error(blas_error);
        goto _flexi_tensor_op_softmax_cleanup;
    }
    blas_error = fnblas_vector_create(&shifted_vector, columns, x->_dtype);
    if (blas_error != FNBLAS_SUCCESS) {
        error = flexi_tensor_op_from_fnblas_error(blas_error);
        goto _flexi_tensor_op_softmax_cleanup;
    }
    blas_error = fnblas_vector_create(&divisor_vector, columns, x->_dtype);
    if (blas_error != FNBLAS_SUCCESS) {
        error = flexi_tensor_op_from_fnblas_error(blas_error);
        goto _flexi_tensor_op_softmax_cleanup;
    }

    for (row = 0; row < rows; ++row) {
        fnblas_vector_t x_row = FNBLAS_VECTOR_INITIALIZER;
        fnblas_vector_t out_row = FNBLAS_VECTOR_INITIALIZER;
        fnblas_vector_t out_row_input = FNBLAS_VECTOR_INITIALIZER;
        fnblas_scalar_t maximum;
        fnblas_scalar_t sum;

        blas_error = fnblas_matrix_create_row_vector_view(&x_row, &x_matrix, row);
        if (blas_error == FNBLAS_SUCCESS) blas_error = fnblas_matrix_create_row_vector_view(&out_row, &out_matrix, row);
        if (blas_error == FNBLAS_SUCCESS) blas_error = fnblas_matrix_create_row_vector_view(&out_row_input, &out_matrix, row);
        if (blas_error != FNBLAS_SUCCESS) {
            fnblas_vector_destroy(&out_row_input);
            fnblas_vector_destroy(&out_row);
            fnblas_vector_destroy(&x_row);
            error = flexi_tensor_op_from_fnblas_error(blas_error);
            goto _flexi_tensor_op_softmax_cleanup;
        }
        error = max_kernel(&maximum, &x_row);
        if (error == FLEXI_TENSOR_SUCCESS) error = subtract_kernel(&shifted_vector, &x_row, &maximum);
        if (error == FLEXI_TENSOR_SUCCESS) error = exp_kernel(&out_row, &shifted_vector);
        if (error == FLEXI_TENSOR_SUCCESS) error = sum_kernel(&sum, &out_row);
        if (error == FLEXI_TENSOR_SUCCESS) {
            float* divisor = (float*)divisor_vector._buffer;
            for (column = 0; column < columns; ++column) divisor[column] = sum._value.f32;
            error = divide_kernel(&out_row, &out_row_input, &divisor_vector);
        }
        fnblas_vector_destroy(&out_row_input);
        fnblas_vector_destroy(&out_row);
        fnblas_vector_destroy(&x_row);
        if (error != FLEXI_TENSOR_SUCCESS) goto _flexi_tensor_op_softmax_cleanup;
    }
    error = FLEXI_TENSOR_SUCCESS;

_flexi_tensor_op_softmax_cleanup:
    fnblas_vector_destroy(&divisor_vector);
    fnblas_vector_destroy(&shifted_vector);
    fnblas_matrix_destroy(&out_matrix);
    fnblas_matrix_destroy(&x_matrix);
    if (error != FLEXI_TENSOR_SUCCESS && out_created) flexi_tensor_destroy(out);
    return error;
}

static flexi_tensor_error_t flexi_tensor_op_quantization(flexi_tensor_t* out, const flexi_tensor_t* x, const fnblas_scalar_t* scale, const fnblas_scalar_t* zero_point, fnblas_dtype_t out_dtype, int dequantize)
{
    _flexi_tensor_backend_kernel_op_vq kernel = dequantize
        ? FLEXI_TENSOR_BACKEND_KERNEL_GET(_flexi_tensor_backend_kernel_op_vq_dequant_per_tensor)
        : FLEXI_TENSOR_BACKEND_KERNEL_GET(_flexi_tensor_backend_kernel_op_vq_quant_per_tensor);
    fnblas_vector_t out_vector = FNBLAS_VECTOR_INITIALIZER;
    fnblas_vector_t x_vector = FNBLAS_VECTOR_INITIALIZER;
    flexi_tuple_t out_shape = FLEXI_TUPLE_EMPTY;
    flexi_tensor_error_t error;
    fnblas_error_t blas_error;
    size_t n_elements = 1;
    size_t dimension;
    int out_created = 0;
    if (out == NULL || x == NULL || out == x || scale == NULL || zero_point == NULL) return FLEXI_TENSOR_ERR_UNKNOWN;
    if (kernel == NULL) return FLEXI_TENSOR_ERR_UNSUPPORTED;
    if (x->_n_dims == 0 || x->_n_dims > 4 || x->_shape == NULL || x->_strides == NULL || !_fnblas_dtype_is_valid(x->_dtype) || !_fnblas_dtype_is_valid(out_dtype)) return FLEXI_TENSOR_ERR_MISMATCH;
    for (dimension = 0; dimension < x->_n_dims; ++dimension) {
        if (x->_shape[dimension] != 0 && n_elements > SIZE_MAX / x->_shape[dimension]) return FLEXI_TENSOR_ERR_MISMATCH;
        n_elements *= x->_shape[dimension];
    }
    if (n_elements != 0 && x->_buffer == NULL) return FLEXI_TENSOR_ERR_MISMATCH;
    if (flexi_tensor_op_is_initialized(out)) {
        if (out->_n_dims != x->_n_dims || out->_dtype != out_dtype || out->_shape == NULL || out->_strides == NULL) return FLEXI_TENSOR_ERR_MISMATCH;
        for (dimension = 0; dimension < x->_n_dims; ++dimension) {
            if (out->_shape[dimension] != x->_shape[dimension]) return FLEXI_TENSOR_ERR_MISMATCH;
        }
    } else {
        out_created = 1;
        out_shape._n_dims = x->_n_dims;
        for (dimension = 0; dimension < x->_n_dims; ++dimension) out_shape._values[dimension] = x->_shape[dimension];
        error = flexi_tensor_create(out, out_shape, out_dtype);
        if (error != FLEXI_TENSOR_SUCCESS) return error;
    }
    if (out->_buffer == x->_buffer && n_elements != 0) {
        error = FLEXI_TENSOR_ERR_MISMATCH;
        goto _flexi_tensor_op_quantization_cleanup;
    }
    if (n_elements == 0) {
        error = FLEXI_TENSOR_SUCCESS;
        goto _flexi_tensor_op_quantization_cleanup;
    }
    blas_error = fnblas_vector_create_view(&x_vector, (byte_t*)x->_buffer, n_elements, x->_dtype);
    if (blas_error == FNBLAS_SUCCESS) blas_error = fnblas_vector_create_view(&out_vector, out->_buffer, n_elements, out->_dtype);
    if (blas_error != FNBLAS_SUCCESS) {
        error = flexi_tensor_op_from_fnblas_error(blas_error);
        goto _flexi_tensor_op_quantization_cleanup;
    }
    error = kernel(&out_vector, &x_vector, scale, zero_point, out_dtype);

_flexi_tensor_op_quantization_cleanup:
    fnblas_vector_destroy(&out_vector);
    fnblas_vector_destroy(&x_vector);
    if (error != FLEXI_TENSOR_SUCCESS && out_created) flexi_tensor_destroy(out);
    return error;
}

flexi_tensor_error_t flexi_tensor_op_quant_per_tensor(flexi_tensor_t* out, const flexi_tensor_t* x, const fnblas_scalar_t scale, const fnblas_scalar_t zero_point, const fnblas_dtype_t out_dtype)
{
    return flexi_tensor_op_quantization(out, x, &scale, &zero_point, out_dtype, 0);
}

flexi_tensor_error_t flexi_tensor_op_dequant_per_tensor(flexi_tensor_t* out, const flexi_tensor_t* x, const fnblas_scalar_t scale, const fnblas_scalar_t zero_point, const fnblas_dtype_t out_dtype)
{
    return flexi_tensor_op_quantization(out, x, &scale, &zero_point, out_dtype, 1);
}
