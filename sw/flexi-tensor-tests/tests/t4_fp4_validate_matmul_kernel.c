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
    flexi_npu_context_t ctx = FLEXI_NPU_FP4_DEFAULT_CONTEXT;
    flexi_tensor_t mat_a = FLEXI_TENSOR_INITIALIZER;
    flexi_tensor_t mat_b = FLEXI_TENSOR_INITIALIZER;
    flexi_tensor_t mat_c_fnblas = FLEXI_TENSOR_INITIALIZER;
    flexi_tensor_t mat_c_npu = FLEXI_TENSOR_INITIALIZER;
    const size_t M = 128;
    const size_t N = 128;
    const size_t K = 128;
    float* a_buffer;
    float* b_buffer;
    float* fnblas_buffer;
    float* npu_buffer;

    if (fnblas_create_static_memory(test_static_memory, sizeof(test_static_memory), TEST_STATIC_MEMORY_BUFFERS) != FNBLAS_SUCCESS) {
        printf("fnblas_create_static_memory failed with errno %d\n", errno);
        return 1;
    }
    if (check_error(flexi_npu_init(ctx), "flexi_npu_init")) return 1;
    if (check_error(flexi_tensor_create(&mat_a, FLEXI_TUPLE(M, K), FP4), "create mat_a")) return 1;
    if (check_error(flexi_tensor_create(&mat_b, FLEXI_TUPLE(N, K), FP4), "create mat_b")) return 1;
    if (check_error(flexi_tensor_create(&mat_c_fnblas, FLEXI_TUPLE(M, N), BF16), "create mat_c_fnblas")) return 1;
    if (check_error(flexi_tensor_create(&mat_c_npu, FLEXI_TUPLE(M, N), BF16), "create mat_c_npu")) return 1;

    a_buffer = (float*)mat_a._buffer;
    b_buffer = (float*)mat_b._buffer;
    for (size_t row = 0; row < M; ++row) {
        for (size_t col = 0; col < K; ++col) a_buffer[row * K + col] = row == col ? 1.0f : 0.0f;
    }
    for (size_t row = 0; row < N; ++row) {
        for (size_t col = 0; col < K; ++col) b_buffer[row * K + col] = (float)((row + col) % 8) - 4.0f;
    }

    if (check_error(flexi_tensor_op_linear(&mat_c_fnblas, &mat_a, &mat_b, NULL), "fnblas linear")) return 1;
    if (check_error(flexi_tensor_use_backend(&flexi_npu_backend), "use NPU backend")) return 1;
    if (check_error(flexi_tensor_op_linear(&mat_c_npu, &mat_a, &mat_b, NULL), "NPU linear")) return 1;

    fnblas_buffer = (float*)mat_c_fnblas._buffer;
    npu_buffer = (float*)mat_c_npu._buffer;
    for (size_t i = 0; i < M * N; ++i) {
        if (fnblas_buffer[i] != npu_buffer[i]) {
            printf("Mismatch at index %zu: fnblas=%f, npu=%f\n", i, fnblas_buffer[i], npu_buffer[i]);
            return 1;
        }
    }

    printf("Test Passed: NPU backend matches fnblas backend results.\n");
    return 0;
}
