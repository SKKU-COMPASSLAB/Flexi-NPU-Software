#include "flexi_tensor.h"
#include "flexi_npu_isa.h"
#include "flexi_npu_backend.h"

#include <errno.h>
#include <stdio.h>

#define TEST_STATIC_MEMORY_SIZE (2 * 1024 * 1024)
#define TEST_STATIC_MEMORY_BUFFERS 128

static byte_t test_static_memory[TEST_STATIC_MEMORY_SIZE] __attribute__((aligned(FNBLAS_MEM_ALIGNMENT)));

static int check_error(flexi_tensor_error_t error, const char* operation) {
    if (error == FLEXI_TENSOR_SUCCESS) return 0;
    printf("%s failed with error %d, errno %d\n", operation, (int)error, errno);
    return 1;
}

int main(int argc, char** argv) {
    flexi_npu_context_t ctx = FLEXI_NPU_BF16_DEFAULT_CONTEXT;
    flexi_tensor_t mat_ifm = FLEXI_TENSOR_INITIALIZER;
    flexi_tensor_t mat_wgt = FLEXI_TENSOR_INITIALIZER;
    flexi_tensor_t mat_ofm_fnblas = FLEXI_TENSOR_INITIALIZER;
    flexi_tensor_t mat_ofm_npu = FLEXI_TENSOR_INITIALIZER;

    const size_t N = 1;
    const size_t C = 32;
    const size_t H = 32;
    const size_t W = 32;

    const size_t K = 32;
    const size_t FH = 3;
    const size_t FW = 3;

    const size_t S = 1; // stride
    const size_t P = 1; // padding
    const size_t D = 1; // dilation

    float* a_buffer;
    float* b_buffer;
    float* fnblas_buffer;
    float* npu_buffer;

    if (fnblas_create_static_memory(test_static_memory, sizeof(test_static_memory), TEST_STATIC_MEMORY_BUFFERS) != FNBLAS_SUCCESS) {
        printf("fnblas_create_static_memory failed with errno %d\n", errno);
        return 1;
    }
    if (check_error(flexi_npu_init(ctx), "flexi_npu_init")) return 1;
    if (check_error(flexi_tensor_create(&mat_ifm, FLEXI_TUPLE(N, H, W, C), BF16), "create mat_ifm")) return 1;
    if (check_error(flexi_tensor_create(&mat_wgt, FLEXI_TUPLE(FH, FW, K, C), BF16), "create mat_wgt")) return 1;

    a_buffer = (float*)mat_ifm._buffer;
    b_buffer = (float*)mat_wgt._buffer;
    for (size_t i = 0; i < N * H * W * C; ++i) a_buffer[i] = (float)(i % 8) - 4.0f;
    for (size_t i = 0; i < FH * FW * K * C; ++i) b_buffer[i] = (float)(i % 8) - 4.0f;

    if (check_error(flexi_tensor_op_conv2d(&mat_ofm_fnblas, &mat_ifm, &mat_wgt, NULL, S, P, D), "fnblas conv2d")) return 1;
    if (check_error(flexi_tensor_use_backend(&flexi_npu_backend), "use NPU backend")) return 1;
    if (check_error(flexi_tensor_op_conv2d(&mat_ofm_npu, &mat_ifm, &mat_wgt, NULL, S, P, D), "NPU conv2d")) return 1;

    fnblas_buffer = (float*)mat_ofm_fnblas._buffer;
    npu_buffer = (float*)mat_ofm_npu._buffer;
    for (size_t i = 0; i < N * H * W * K; ++i) {
        if (fnblas_buffer[i] != npu_buffer[i]) {
            printf("Mismatch at index %zu: fnblas=%f, npu=%f\n", i, fnblas_buffer[i], npu_buffer[i]);
            return 1;
        }
    }

    printf("Test Passed: NPU backend matches fnblas backend results.\n");
    return 0;
}
