#include "flexi_tensor.h"
#include "flexi_npu_backend.h"

#include <errno.h>
#include <stdio.h>

#define TEST_STATIC_MEMORY_SIZE (2 * 1024 * 1024)
#define TEST_STATIC_MEMORY_BUFFERS 128
#define TEST_VECTOR_LENGTH 65

static byte_t test_static_memory[TEST_STATIC_MEMORY_SIZE] __attribute__((aligned(FNBLAS_MEM_ALIGNMENT)));

static int check_tensor_error(flexi_tensor_error_t error, const char* operation) {
    if (error == FLEXI_TENSOR_SUCCESS) return 0;
    printf("%s failed with error %d, errno %d\n", operation, (int)error, errno);
    return 1;
}

static int check_blas_error(fnblas_error_t error, const char* operation) {
    if (error == FNBLAS_SUCCESS) return 0;
    printf("%s failed with error %d, errno %d\n", operation, (int)error, errno);
    return 1;
}

static int validate_bf16_to_fp4(void) {
    static const float values[] = {
        -16.0f, -12.0f, -8.0f, -7.0f, -6.0f, -5.0f, -4.0f, -3.0f,
        -2.0f, -1.5f, -1.0f, -0.5f, -0.25f, 0.0f, 0.25f, 0.5f,
        1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 12.0f
    };
    flexi_tensor_t input = FLEXI_TENSOR_INITIALIZER;
    flexi_tensor_t reference = FLEXI_TENSOR_INITIALIZER;
    flexi_tensor_t result = FLEXI_TENSOR_INITIALIZER;
    fnblas_scalar_t scale;
    fnblas_scalar_t zero_point;
    float* input_buffer;
    float* reference_buffer;
    float* result_buffer;
    size_t index;

    if (check_tensor_error(flexi_tensor_create(&input, FLEXI_TUPLE(TEST_VECTOR_LENGTH), BF16), "create BF16 input")) return 1;
    input_buffer = (float*)input._buffer;
    for (index = 0; index < TEST_VECTOR_LENGTH; ++index) input_buffer[index] = values[index % (sizeof(values) / sizeof(values[0]))];
    if (check_blas_error(fnblas_scalar_create_float(&scale, 2.0f, BF16), "create BF16 scale")) return 1;
    if (check_blas_error(fnblas_scalar_create_float(&zero_point, 0.0f, BF16), "create BF16 zero point")) return 1;

    if (check_tensor_error(flexi_tensor_use_backend(&flexi_tensor_fnblas_backend), "use fnblas backend")) return 1;
    if (check_tensor_error(flexi_tensor_op_quant_per_tensor(&reference, &input, scale, zero_point, FP4), "fnblas BF16-to-FP4 quantization")) return 1;
    if (check_tensor_error(flexi_tensor_use_backend(&flexi_npu_backend), "use NPU backend")) return 1;
    if (check_tensor_error(flexi_tensor_op_quant_per_tensor(&result, &input, scale, zero_point, FP4), "NPU BF16-to-FP4 quantization")) return 1;

    reference_buffer = (float*)reference._buffer;
    result_buffer = (float*)result._buffer;
    for (index = 0; index < TEST_VECTOR_LENGTH; ++index) {
        if (reference_buffer[index] != result_buffer[index]) {
            printf("BF16-to-FP4 mismatch at index %zu: input=%f, fnblas=%f, npu=%f\n", index, input_buffer[index], reference_buffer[index], result_buffer[index]);
            return 1;
        }
    }

    flexi_tensor_destroy(&result);
    flexi_tensor_destroy(&reference);
    flexi_tensor_destroy(&input);
    return 0;
}

static int validate_fallback(void) {
    flexi_tensor_t input = FLEXI_TENSOR_INITIALIZER;
    flexi_tensor_t reference = FLEXI_TENSOR_INITIALIZER;
    flexi_tensor_t result = FLEXI_TENSOR_INITIALIZER;
    fnblas_scalar_t scale;
    fnblas_scalar_t zero_point;
    float* input_buffer;
    int8_t* reference_buffer;
    int8_t* result_buffer;
    size_t index;

    if (check_tensor_error(flexi_tensor_create(&input, FLEXI_TUPLE(7), FP32), "create FP32 input")) return 1;
    input_buffer = (float*)input._buffer;
    input_buffer[0] = -1.0f;
    input_buffer[1] = -1.4f;
    input_buffer[2] = 0.0f;
    input_buffer[3] = 1.4f;
    input_buffer[4] = 0.2f;
    input_buffer[5] = 0.4f;
    input_buffer[6] = 0.8f;
    if (check_blas_error(fnblas_scalar_create_float(&scale, 1.0f, FP32), "create FP32 scale")) return 1;
    if (check_blas_error(fnblas_scalar_create_float(&zero_point, 0.0f, FP32), "create FP32 zero point")) return 1;

    if (check_tensor_error(flexi_tensor_use_backend(&flexi_tensor_fnblas_backend), "use fnblas fallback reference")) return 1;
    if (check_tensor_error(flexi_tensor_op_quant_per_tensor(&reference, &input, scale, zero_point, FP4), "fnblas FP32-to-FP4 quantization")) return 1;
    if (check_tensor_error(flexi_tensor_use_backend(&flexi_npu_backend), "use NPU fallback backend")) return 1;
    if (check_tensor_error(flexi_tensor_op_quant_per_tensor(&result, &input, scale, zero_point, FP4), "NPU fallback FP32-to-FP4 quantization")) return 1;

    reference_buffer = (int8_t*)reference._buffer;
    result_buffer = (int8_t*)result._buffer;
    for (index = 0; index < 7; ++index) {
        if (reference_buffer[index] != result_buffer[index]) {
            printf("Fallback mismatch at index %zu: fnblas=%d, npu=%d\n", index, (int)reference_buffer[index], (int)result_buffer[index]);
            return 1;
        }
    }

    flexi_tensor_destroy(&result);
    flexi_tensor_destroy(&reference);
    flexi_tensor_destroy(&input);
    return 0;
}

int main(int argc, char** argv) {
    if (fnblas_create_static_memory(test_static_memory, sizeof(test_static_memory), TEST_STATIC_MEMORY_BUFFERS) != FNBLAS_SUCCESS) {
        printf("fnblas_create_static_memory failed with errno %d\n", errno);
        return 1;
    }
    flexi_npu_context_t ctx = FLEXI_NPU_FP4_DEFAULT_CONTEXT;
    if (flexi_npu_init(ctx) != FLEXI_TENSOR_SUCCESS) {
        printf("flexi_npu_init failed with errno %d\n", errno);
        return 1;
    }
    if (validate_bf16_to_fp4()) return 1;
    if (validate_fallback()) return 1;
    printf("Test Passed: NPU quantization backend matches fnblas results.\n");
    return 0;
}
