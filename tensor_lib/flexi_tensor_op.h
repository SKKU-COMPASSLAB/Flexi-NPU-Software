#ifndef FLEXI_TENSOR_OP_H
#define FLEXI_TENSOR_OP_H

#include <stddef.h>
#include <stdint.h>

#include "flexi_blas.h"
#include "flexi_tensor_common.h"
#include "flexi_tensor_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// Operators
flexi_tensor_error_t flexi_tensor_op_linear(flexi_tensor_t* out, const flexi_tensor_t* x, const flexi_tensor_t* w, const flexi_tensor_t* b);
flexi_tensor_error_t flexi_tensor_op_relu(flexi_tensor_t* out, const flexi_tensor_t* x);
flexi_tensor_error_t flexi_tensor_op_conv2d(flexi_tensor_t* out, const flexi_tensor_t* x, const flexi_tensor_t* w, const flexi_tensor_t* b, const size_t stride, const size_t padding, const size_t dilation);
flexi_tensor_error_t flexi_tensor_op_maxpool2d(flexi_tensor_t* out, const flexi_tensor_t* x, const size_t kernel_size, const size_t stride, const size_t padding);
flexi_tensor_error_t flexi_tensor_op_avgpool2d(flexi_tensor_t* out, const flexi_tensor_t* x, const size_t kernel_size, const size_t stride, const size_t padding);
flexi_tensor_error_t flexi_tensor_op_softmax(flexi_tensor_t* out, const flexi_tensor_t* x);

#ifdef __cplusplus
}
#endif

#endif // FLEXI_TENSOR_OP_H
