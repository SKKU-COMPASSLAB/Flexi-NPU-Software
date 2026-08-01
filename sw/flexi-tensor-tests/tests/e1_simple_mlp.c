#include "flexi_tensor.h"
#include "flexi_npu_backend.h"
#include "e1_simple_mlp_dump.h"

#include <errno.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#define TEST_STATIC_MEMORY_SIZE (16 * 1024 * 1024)
#define TEST_STATIC_MEMORY_BUFFERS 512

static byte_t test_static_memory[TEST_STATIC_MEMORY_SIZE] __attribute__((aligned(FNBLAS_MEM_ALIGNMENT)));

void required(flexi_tensor_error_t error, const char* operation) {
    if (error != FLEXI_TENSOR_SUCCESS) {
        printf("%s failed with error %d, errno %d\n", operation, (int)error, errno);
        exit(1);
    }
}

flexi_tensor_error_t create_dump_tensor(flexi_tensor_t* tensor, flexi_tuple_t shape, uintptr_t address) {
    flexi_tensor_error_t error = flexi_tensor_create(tensor, shape, FP32);
    if (error != FLEXI_TENSOR_SUCCESS) return error;
    error = flexi_tensor_initialize_from_unpacked_buffer(tensor, (const byte_t*)address);
    if (error != FLEXI_TENSOR_SUCCESS) flexi_tensor_destroy(tensor);
    return error;
}

int main(int argc, char** argv) {
    if (fnblas_create_static_memory(test_static_memory, sizeof(test_static_memory), TEST_STATIC_MEMORY_BUFFERS) != FNBLAS_SUCCESS) {
        printf("fnblas_create_static_memory failed with errno %d\n", errno);
        return 1;
    }

    flexi_tensor_t x = FLEXI_TENSOR_INITIALIZER;
    flexi_tensor_t w1 = FLEXI_TENSOR_INITIALIZER;
    flexi_tensor_t b1 = FLEXI_TENSOR_INITIALIZER;
    flexi_tensor_t w2 = FLEXI_TENSOR_INITIALIZER;
    flexi_tensor_t b2 = FLEXI_TENSOR_INITIALIZER;

    const size_t input_dim = E1_SIMPLE_MLP_INPUT_DIM;
    const size_t hidden_dim = E1_SIMPLE_MLP_HIDDEN_DIM;
    const size_t output_dim = E1_SIMPLE_MLP_OUTPUT_DIM;

    required(create_dump_tensor(&x, FLEXI_TUPLE(1, input_dim), E1_SIMPLE_MLP_X_ADDRESS), "create input tensor");
    required(create_dump_tensor(&w1, FLEXI_TUPLE(hidden_dim, input_dim), E1_SIMPLE_MLP_W1_ADDRESS), "create linear1 weights");
    required(create_dump_tensor(&b1, FLEXI_TUPLE(hidden_dim), E1_SIMPLE_MLP_B1_ADDRESS), "create linear1 bias");
    required(create_dump_tensor(&w2, FLEXI_TUPLE(output_dim, hidden_dim), E1_SIMPLE_MLP_W2_ADDRESS), "create linear2 weights");
    required(create_dump_tensor(&b2, FLEXI_TUPLE(output_dim), E1_SIMPLE_MLP_B2_ADDRESS), "create linear2 bias");

    // layer 1: x -> linear1 -> relu
    flexi_tensor_t linear1_out = FLEXI_TENSOR_INITIALIZER;
    flexi_tensor_t relu1_out = FLEXI_TENSOR_INITIALIZER;
    required(flexi_tensor_op_linear(&linear1_out, &x, &w1, &b1), "linear1");
    required(flexi_tensor_op_relu(&relu1_out, &linear1_out), "relu1");
    flexi_tensor_destroy(&linear1_out);
    flexi_tensor_destroy(&x);

    // layer 2: linear1_out -> linear2
    flexi_tensor_t linear2_out = FLEXI_TENSOR_INITIALIZER;
    flexi_tensor_t relu2_out = FLEXI_TENSOR_INITIALIZER;
    required(flexi_tensor_op_linear(&linear2_out, &relu1_out, &w2, &b2), "linear2");
    required(flexi_tensor_op_relu(&relu2_out, &linear2_out), "relu2");
    flexi_tensor_destroy(&linear2_out);
    flexi_tensor_destroy(&relu1_out);

    // layer 3: linear2_out -> output
    flexi_tensor_t y = FLEXI_TENSOR_INITIALIZER;
    required(flexi_tensor_op_softmax(&y, &relu2_out), "softmax");
    flexi_tensor_destroy(&relu2_out);

    // clean up
    flexi_tensor_destroy(&w1);
    flexi_tensor_destroy(&b1);
    flexi_tensor_destroy(&w2);
    flexi_tensor_destroy(&b2);

    // print output
    float* y_buffer = (float*)y._buffer;
    const float* golden_buffer = (const float*)E1_SIMPLE_MLP_GOLDEN_ADDRESS;
    float probability_sum = 0.0f;
    printf("Output probabilities:\n");
    for (size_t i = 0; i < output_dim; ++i) {
        int probability_millionths = (int)(y_buffer[i] * 1000000.0f + 0.5f);
        printf("Class %d: %d / 1000000\n", (int)i, probability_millionths);
        if (y_buffer[i] < 0.0f || y_buffer[i] > 1.0f) {
            printf("Invalid probability at class %d\n", (int)i);
            return 1;
        }
        if (y_buffer[i] < golden_buffer[i] - 0.000001f || y_buffer[i] > golden_buffer[i] + 0.000001f) {
            printf("Golden mismatch at class %d\n", (int)i);
            return 1;
        }
        probability_sum += y_buffer[i];
    }
    if (probability_sum < 0.999f || probability_sum > 1.001f) {
        printf("Invalid probability sum\n");
        return 1;
    }
    flexi_tensor_destroy(&y);

    printf("Simple MLP test completed successfully.\n");

    return 0;
}
