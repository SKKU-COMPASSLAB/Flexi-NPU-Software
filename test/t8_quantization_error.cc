#include "flexi_tensor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

fnblas_scalar_t make_fp32_scalar(float value)
{
    fnblas_scalar_t scalar{};
    require(fnblas_scalar_create_float(&scalar, value, FP32) == FNBLAS_SUCCESS,
        "failed to create an FP32 scalar");
    return scalar;
}

void print_tensor(const std::string& name, const flexi_tensor_t& tensor)
{
    const float* values = reinterpret_cast<const float*>(tensor._buffer);
    std::cout << name << ":\n[";
    for (size_t index = 0; index < tensor._shape[0]; ++index) {
        if (index != 0)
            std::cout << ", ";
        std::cout << values[index];
    }
    std::cout << "]\n";
}

void test_fp4_quantization() {
    //   - create a BF16 tensor with its length of 32 and initialize it with random values (seed=42)
    constexpr size_t tensor_length = 32;
    std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(-10.0f, 10.0f);
    std::vector<float> random_values(tensor_length);
    for (float& value : random_values)
        value = distribution(generator);

    flexi_tensor_t original = FLEXI_TENSOR_INITIALIZER;
    flexi_tensor_t quantized = FLEXI_TENSOR_INITIALIZER;
    flexi_tensor_t dequantized = FLEXI_TENSOR_INITIALIZER;
    require(flexi_tensor_create(&original, FLEXI_TUPLE(tensor_length), BF16) ==
        FLEXI_TENSOR_SUCCESS, "failed to create the BF16 tensor");
    require(flexi_tensor_initialize_from_unpacked_buffer(
        &original, reinterpret_cast<const byte_t*>(random_values.data())) ==
        FLEXI_TENSOR_SUCCESS, "failed to initialize the BF16 tensor");

    const float* original_values =
        reinterpret_cast<const float*>(original._buffer);
    float max_absolute_value = 0.0f;
    for (size_t index = 0; index < tensor_length; ++index)
        max_absolute_value =
            std::max(max_absolute_value, std::fabs(original_values[index]));

    const float optimal_scale =
        max_absolute_value == 0.0f ? 1.0f : max_absolute_value / 6.0f;
    const fnblas_scalar_t scale = make_fp32_scalar(optimal_scale);
    const fnblas_scalar_t zero_point = make_fp32_scalar(0.0f);
    require(flexi_tensor_op_quant_per_tensor(
        &quantized, &original, scale, zero_point, FP4) ==
        FLEXI_TENSOR_SUCCESS, "failed to quantize the BF16 tensor to FP4");

    require(flexi_tensor_op_dequant_per_tensor(
        &dequantized, &quantized, scale, zero_point, BF16) ==
        FLEXI_TENSOR_SUCCESS, "failed to dequantize the FP4 tensor to BF16");

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "FP4 scale: " << optimal_scale << '\n';
    print_tensor("Original BF16 tensor", original);
    print_tensor("Quantized FP4 tensor", quantized);
    print_tensor("Dequantized BF16 tensor", dequantized);

    const float* restored_values =
        reinterpret_cast<const float*>(dequantized._buffer);
    double squared_error_sum = 0.0;
    for (size_t index = 0; index < tensor_length; ++index) {
        const double error =
            static_cast<double>(original_values[index]) - restored_values[index];
        squared_error_sum += error * error;
    }
    const double mse = squared_error_sum / tensor_length;
    require(std::isfinite(mse), "quantization MSE is not finite");
    std::cout << "Quantization error (MSE): " << mse << '\n';

    flexi_tensor_destroy(&dequantized);
    flexi_tensor_destroy(&quantized);
    flexi_tensor_destroy(&original);
}

}

int main()
{
    try {
        test_fp4_quantization();
        std::cout << "All tests passed." << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}
