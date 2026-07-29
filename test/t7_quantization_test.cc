#include "flexi_tensor.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr float TOLERANCE = 1.0e-6f;

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

fnblas_scalar_t make_float_scalar(float value)
{
    fnblas_scalar_t scalar{};
    require(fnblas_scalar_create_float(&scalar, value, FP32) == FNBLAS_SUCCESS, "create float scalar");
    return scalar;
}

fnblas_scalar_t make_int_scalar(int32_t value)
{
    fnblas_scalar_t scalar{};
    require(fnblas_scalar_create_int(&scalar, value, INT32) == FNBLAS_SUCCESS, "create integer scalar");
    return scalar;
}

fnblas_vector_t make_float_vector(const std::vector<float>& values, fnblas_dtype_t dtype)
{
    fnblas_vector_t vector = FNBLAS_VECTOR_INITIALIZER;
    require(fnblas_vector_create(&vector, values.size(), dtype) == FNBLAS_SUCCESS, "create float vector");
    require(fnblas_vector_initialize_from_unpacked_buffer(&vector, (const byte_t*)values.data()) == FNBLAS_SUCCESS, "initialize float vector");
    return vector;
}

void require_float_vector(const fnblas_vector_t& vector, const std::vector<float>& expected, const std::string& message)
{
    const float* values = (const float*)vector._buffer;
    require(vector._n_elements == expected.size(), message + ": size");
    for (size_t index = 0; index < expected.size(); ++index) {
        if (std::fabs(values[index] - expected[index]) > TOLERANCE)
            throw std::runtime_error(message + ": value");
    }
}

void require_integer_vector(const fnblas_vector_t& vector, const std::vector<int32_t>& expected, const std::string& message)
{
    require(vector._n_elements == expected.size(), message + ": size");
    for (size_t index = 0; index < expected.size(); ++index) {
        const int32_t value = vector._dtype == INT8
            ? (int32_t)((const int8_t*)vector._buffer)[index]
            : ((const int32_t*)vector._buffer)[index];
        if (value != expected[index])
            throw std::runtime_error(message + ": value");
    }
}

void test_mixed_dtype_cast()
{
    fnblas_vector_t input = make_float_vector({1.6f, -1.6f, 200.0f, -200.0f}, FP32);
    fnblas_vector_t integers = FNBLAS_VECTOR_INITIALIZER;
    fnblas_vector_t restored = FNBLAS_VECTOR_INITIALIZER;
    require(fnblas_op_vc_cast(&integers, &input, INT8) == FNBLAS_SUCCESS, "FP32 to INT8 cast");
    require_integer_vector(integers, {2, -2, 127, -128}, "rounded saturating cast");
    require(fnblas_op_vc_cast(&restored, &integers, FP32) == FNBLAS_SUCCESS, "INT8 to FP32 cast");
    require_float_vector(restored, {2.0f, -2.0f, 127.0f, -128.0f}, "restored cast");
    fnblas_vector_destroy(&restored);
    fnblas_vector_destroy(&integers);
    fnblas_vector_destroy(&input);
}

void test_fp4_quantization()
{
    fnblas_scalar_t scale = make_float_scalar(2.0f);
    fnblas_scalar_t zero_point = make_float_scalar(0.0f);
    fnblas_vector_t input = make_float_vector({0.9f, 2.6f, 7.5f, -13.0f}, FP32);
    fnblas_vector_t quantized = FNBLAS_VECTOR_INITIALIZER;
    fnblas_vector_t dequantized = FNBLAS_VECTOR_INITIALIZER;
    require(fnblas_op_vq_quant_per_tensor(&quantized, &input, &scale, &zero_point, FP4) == FNBLAS_SUCCESS, "FP4 quantization");
    require_float_vector(quantized, {0.5f, 1.5f, 4.0f, -6.0f}, "S1E2M1 bins");
    require(fnblas_op_vq_dequant_per_tensor(&dequantized, &quantized, &scale, &zero_point, FP32) == FNBLAS_SUCCESS, "FP4 dequantization");
    require_float_vector(dequantized, {1.0f, 3.0f, 8.0f, -12.0f}, "FP4 dequantized values");
    fnblas_vector_destroy(&dequantized);
    fnblas_vector_destroy(&quantized);
    fnblas_vector_destroy(&input);
}

void test_integer_quantization()
{
    fnblas_scalar_t scale = make_float_scalar(0.5f);
    fnblas_scalar_t zero_point = make_int_scalar(-3);
    fnblas_vector_t input = make_float_vector({-100.0f, 0.0f, 1.2f, 100.0f}, BF16);
    fnblas_vector_t quantized = FNBLAS_VECTOR_INITIALIZER;
    fnblas_vector_t dequantized = FNBLAS_VECTOR_INITIALIZER;
    require(fnblas_op_vq_quant_per_tensor(&quantized, &input, &scale, &zero_point, INT8) == FNBLAS_SUCCESS, "INT8 quantization");
    require_integer_vector(quantized, {-128, -3, -1, 127}, "INT8 quantized values");
    require(fnblas_op_vq_dequant_per_tensor(&dequantized, &quantized, &scale, &zero_point, BF16) == FNBLAS_SUCCESS, "INT8 dequantization");
    require_float_vector(dequantized, {-62.5f, 0.0f, 1.0f, 65.0f}, "INT8 dequantized values");
    fnblas_vector_destroy(&dequantized);
    fnblas_vector_destroy(&quantized);
    fnblas_vector_destroy(&input);

    scale = make_float_scalar(1.0f);
    zero_point = make_int_scalar(0);
    input = make_float_vector({-20.0f, -1.6f, 1.6f, 20.0f}, FP32);
    require(fnblas_op_vq_quant_per_tensor(&quantized, &input, &scale, &zero_point, INT4) == FNBLAS_SUCCESS, "INT4 quantization");
    require_integer_vector(quantized, {-8, -2, 2, 7}, "INT4 quantized values");
    fnblas_vector_destroy(&quantized);
    fnblas_vector_destroy(&input);
}

void test_validation_and_backend_registration()
{
    fnblas_scalar_t scale = make_float_scalar(1.0f);
    fnblas_scalar_t zero_point = make_int_scalar(0);
    fnblas_vector_t fp16 = make_float_vector({1.0f}, FP16);
    fnblas_vector_t result = FNBLAS_VECTOR_INITIALIZER;
    require(fnblas_op_vq_quant_per_tensor(&result, &fp16, &scale, &zero_point, INT8) == FNBLAS_ERR_MISMATCH, "reject FP16 to INT8 quantization");
    require(flexi_tensor_fnblas_backend.op_vc_cast != nullptr, "cast backend registration");
    require(flexi_tensor_fnblas_backend.op_vq_quant_per_tensor != nullptr, "quant backend registration");
    require(flexi_tensor_fnblas_backend.op_vq_dequant_per_tensor != nullptr, "dequant backend registration");
    fnblas_vector_destroy(&result);
    fnblas_vector_destroy(&fp16);
}

bool quant_backend_called = false;

flexi_tensor_error_t tracking_quant_kernel(fnblas_vector_t* result, const fnblas_vector_t* input, const fnblas_scalar_t* scale, const fnblas_scalar_t* zero_point, fnblas_dtype_t out_dtype)
{
    quant_backend_called = true;
    const fnblas_error_t error = fnblas_op_vq_quant_per_tensor(result, input, scale, zero_point, out_dtype);
    return error == FNBLAS_SUCCESS ? FLEXI_TENSOR_SUCCESS : FLEXI_TENSOR_ERR_BACKEND;
}

void test_tensor_quantization()
{
    fnblas_scalar_t scale = make_float_scalar(0.5f);
    fnblas_scalar_t zero_point = make_int_scalar(-3);
    flexi_tensor_t input = FLEXI_TENSOR_INITIALIZER;
    flexi_tensor_t quantized = FLEXI_TENSOR_INITIALIZER;
    flexi_tensor_t dequantized = FLEXI_TENSOR_INITIALIZER;
    const float values[2][2] = {{-100.0f, 0.0f}, {1.2f, 100.0f}};
    require(flexi_tensor_create(&input, FLEXI_TUPLE(2, 2), FP32) == FLEXI_TENSOR_SUCCESS, "create input tensor");
    require(flexi_tensor_initialize_from_unpacked_buffer(&input, (const byte_t*)values) == FLEXI_TENSOR_SUCCESS, "initialize input tensor");

    flexi_tensor_backend_t tracking_backend{};
    tracking_backend.op_vq_quant_per_tensor = tracking_quant_kernel;
    require(flexi_tensor_use_backend(&tracking_backend) == FLEXI_TENSOR_SUCCESS, "select tracking backend");
    require(flexi_tensor_op_quant_per_tensor(&quantized, &input, scale, zero_point, INT8) == FLEXI_TENSOR_SUCCESS, "tensor INT8 quantization");
    require(quant_backend_called, "tensor quantization backend dispatch");
    require(quantized._dtype == INT8 && quantized._n_dims == 2 && quantized._shape[0] == 2 && quantized._shape[1] == 2, "quantized tensor metadata");
    require((int32_t)((const int8_t*)quantized._buffer)[0] == -128, "tensor INT8 lower clamp");
    require((int32_t)((const int8_t*)quantized._buffer)[1] == -3, "tensor INT8 zero point");
    require((int32_t)((const int8_t*)quantized._buffer)[2] == -1, "tensor INT8 rounding");
    require((int32_t)((const int8_t*)quantized._buffer)[3] == 127, "tensor INT8 upper clamp");

    require(flexi_tensor_op_dequant_per_tensor(&dequantized, &quantized, scale, zero_point, FP32) == FLEXI_TENSOR_SUCCESS, "tensor INT8 dequantization");
    require(dequantized._dtype == FP32 && dequantized._n_dims == 2 && dequantized._shape[0] == 2 && dequantized._shape[1] == 2, "dequantized tensor metadata");
    const float* restored = (const float*)dequantized._buffer;
    require(std::fabs(restored[0] - -62.5f) <= TOLERANCE, "tensor dequantized lower value");
    require(std::fabs(restored[1] - 0.0f) <= TOLERANCE, "tensor dequantized zero");
    require(std::fabs(restored[2] - 1.0f) <= TOLERANCE, "tensor dequantized rounded value");
    require(std::fabs(restored[3] - 65.0f) <= TOLERANCE, "tensor dequantized upper value");

    require(flexi_tensor_use_backend(&flexi_tensor_fnblas_backend) == FLEXI_TENSOR_SUCCESS, "restore fnblas backend");
    flexi_tensor_destroy(&dequantized);
    flexi_tensor_destroy(&quantized);
    flexi_tensor_destroy(&input);
}

void test_tensor_fp4_and_output_validation()
{
    fnblas_scalar_t scale = make_float_scalar(2.0f);
    fnblas_scalar_t zero_point = make_float_scalar(0.0f);
    flexi_tensor_t input = FLEXI_TENSOR_INITIALIZER;
    flexi_tensor_t quantized = FLEXI_TENSOR_INITIALIZER;
    flexi_tensor_t wrong_output = FLEXI_TENSOR_INITIALIZER;
    const float values[4] = {0.9f, 2.6f, 7.5f, -13.0f};
    require(flexi_tensor_create(&input, FLEXI_TUPLE(4), BF16) == FLEXI_TENSOR_SUCCESS, "create BF16 tensor");
    require(flexi_tensor_initialize_from_unpacked_buffer(&input, (const byte_t*)values) == FLEXI_TENSOR_SUCCESS, "initialize BF16 tensor");
    require(flexi_tensor_op_quant_per_tensor(&quantized, &input, scale, zero_point, FP4) == FLEXI_TENSOR_SUCCESS, "tensor FP4 quantization");
    const float* fp4 = (const float*)quantized._buffer;
    require(fp4[0] == 0.5f && fp4[1] == 1.5f && fp4[2] == 4.0f && fp4[3] == -6.0f, "tensor FP4 bins");

    require(flexi_tensor_create(&wrong_output, FLEXI_TUPLE(4), FP32) == FLEXI_TENSOR_SUCCESS, "create wrong output");
    require(flexi_tensor_op_quant_per_tensor(&wrong_output, &input, scale, zero_point, FP4) == FLEXI_TENSOR_ERR_MISMATCH, "reject initialized output dtype mismatch");
    flexi_tensor_destroy(&wrong_output);
    flexi_tensor_destroy(&quantized);
    flexi_tensor_destroy(&input);
}

}

int main()
{
    try {
        test_mixed_dtype_cast();
        test_fp4_quantization();
        test_integer_quantization();
        test_validation_and_backend_registration();
        test_tensor_quantization();
        test_tensor_fp4_and_output_validation();
        std::cout << "All quantization tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Test failure: " << error.what() << '\n';
        return 1;
    }
}
