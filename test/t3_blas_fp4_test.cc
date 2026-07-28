// Comprehensive FP4 validation for the C flexi_blas API.
#include "flexi_blas.h"

#include <cerrno>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr float TOLERANCE = 1.0e-5f;

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void require_close(float actual, float expected, const std::string& message)
{
    if (std::fabs(actual - expected) > TOLERANCE)
        throw std::runtime_error(message);
}

void set_value(fnblas_vector_t* vector, std::size_t index, float value)
{
    fnblas_vector_set(
        vector, index, reinterpret_cast<const byte_t*>(&value)
    );
}

float get_value(const fnblas_vector_t* vector, std::size_t index)
{
    float value = 0.0f;
    fnblas_vector_at(vector, index, reinterpret_cast<byte_t*>(&value));
    return value;
}

fnblas_vector_t make_vector(const std::vector<float>& values, fnblas_dtype_t dtype = FP4)
{
    fnblas_vector_t result = FNBLAS_VECTOR_INITIALIZER;
    fnblas_vector_create(&result, values.size(), dtype);
    require(values.empty() || result._buffer != nullptr, "vector allocation");
    for (std::size_t index = 0; index < values.size(); ++index)
        set_value(&result, index, values[index]);
    return result;
}

fnblas_scalar_t make_scalar(float value, fnblas_dtype_t dtype = FP4)
{
    fnblas_scalar_t result{};
    fnblas_scalar_create_float(&result, value, dtype);
    return result;
}

void check_vector(
    const fnblas_vector_t* actual,
    const std::vector<float>& expected,
    fnblas_dtype_t dtype,
    const std::string& message
)
{
    require(fnblas_vector_dtype(actual) == dtype, message + ": dtype");
    require(fnblas_vector_n_elements(actual) == expected.size(),
            message + ": size");
    for (std::size_t index = 0; index < expected.size(); ++index)
        require_close(get_value(actual, index), expected[index], message);
}

void test_scalar_scalar()
{
    fnblas_scalar_t lhs = make_scalar(6.0f);
    fnblas_scalar_t rhs = make_scalar(2.0f);
    fnblas_scalar_t result{};
    fnblas_op_ss_add(&result, &lhs, &rhs);
    require_close(fnblas_scalar_as_unpacked_float(&result), 8.0f,
                  "scalar addition");
    fnblas_op_ss_sub(&result, &lhs, &rhs);
    require_close(fnblas_scalar_as_unpacked_float(&result), 4.0f,
                  "scalar subtraction");
    fnblas_op_ss_mul(&result, &lhs, &rhs);
    require_close(fnblas_scalar_as_unpacked_float(&result), 12.0f,
                  "scalar multiplication");
    fnblas_op_ss_div(&result, &lhs, &rhs);
    require_close(fnblas_scalar_as_unpacked_float(&result), 3.0f,
                  "scalar division");
}

void test_vector_scalar()
{
    fnblas_vector_t input = make_vector({0.5f, 1.0f, 2.0f, 4.0f});
    fnblas_scalar_t scalar = make_scalar(2.0f);
    fnblas_vector_t result = FNBLAS_VECTOR_INITIALIZER;
    fnblas_op_sv_add(&result, &input, &scalar);
    check_vector(&result, {2.5f, 3.0f, 4.0f, 6.0f}, FP4,
                 "vector + scalar");
    fnblas_vector_destroy(&result);
    fnblas_op_sv_sub(&result, &input, &scalar);
    check_vector(&result, {-1.5f, -1.0f, 0.0f, 2.0f}, FP4,
                 "vector - scalar");
    fnblas_vector_destroy(&result);
    fnblas_op_sv_mul(&result, &input, &scalar);
    check_vector(&result, {1.0f, 2.0f, 4.0f, 8.0f}, FP4,
                 "vector * scalar");
    fnblas_vector_destroy(&result);
    fnblas_op_sv_div(&result, &input, &scalar);
    check_vector(&result, {0.25f, 0.5f, 1.0f, 2.0f}, FP4,
                 "vector / scalar");
    fnblas_vector_destroy(&result);
    fnblas_vector_destroy(&input);
}

void test_vector_vector_and_dot()
{
    fnblas_vector_t lhs = make_vector({1.0f, 2.0f, 3.0f, 6.0f});
    fnblas_vector_t rhs = make_vector({0.5f, 1.0f, 1.5f, 3.0f});
    fnblas_vector_t result = FNBLAS_VECTOR_INITIALIZER;
    fnblas_op_ve_add(&result, &lhs, &rhs);
    check_vector(&result, {1.5f, 3.0f, 4.5f, 9.0f}, FP4,
                 "vector addition");
    fnblas_vector_destroy(&result);
    fnblas_op_ve_sub(&result, &lhs, &rhs);
    check_vector(&result, {0.5f, 1.0f, 1.5f, 3.0f}, FP4,
                 "vector subtraction");
    fnblas_vector_destroy(&result);
    fnblas_op_ve_mul(&result, &lhs, &rhs);
    check_vector(&result, {0.5f, 2.0f, 4.5f, 18.0f}, FP4,
                 "vector multiplication");
    fnblas_vector_destroy(&result);
    fnblas_op_ve_div(&result, &lhs, &rhs);
    check_vector(&result, {2.0f, 2.0f, 2.0f, 2.0f}, FP4,
                 "vector division");
    fnblas_vector_destroy(&result);

    fnblas_scalar_t dot{};
    fnblas_op_vv_dot(&dot, &lhs, &rhs);
    require_close(fnblas_scalar_as_unpacked_float(&dot), 25.0f,
                  "dot product");
    fnblas_vector_destroy(&lhs);
    fnblas_vector_destroy(&rhs);
}

void test_fp4_packed_buffer()
{
    fnblas_vector_t source = make_vector({0.5f, -1.0f, 2.0f, 6.0f, -0.0f});
    std::vector<byte_t> packed(3);
    fnblas_vector_get_packed_buffer(&source, packed.data());
    require(packed[0] == 0xA1u, "first packed byte");
    require(packed[1] == 0x74u, "second packed byte");
    require(packed[2] == 0x08u, "odd packed byte");

    fnblas_vector_t restored = FNBLAS_VECTOR_INITIALIZER;
    fnblas_vector_create(&restored, 5, FP4);
    fnblas_vector_initialize_from_packed_buffer(&restored, packed.data());
    check_vector(&restored, {0.5f, -1.0f, 2.0f, 6.0f, -0.0f}, FP4,
                 "FP4 packed round trip");
    fnblas_vector_destroy(&source);
    fnblas_vector_destroy(&restored);
}

void test_fp4_result_to_bf16()
{
    fnblas_vector_t lhs = make_vector({1.0f, 2.0f, 3.0f, 6.0f});
    fnblas_vector_t rhs = make_vector({0.5f, 1.0f, 1.5f, 3.0f});
    fnblas_vector_t fp4_result = FNBLAS_VECTOR_INITIALIZER;
    fnblas_op_ve_add(&fp4_result, &lhs, &rhs);

    fnblas_vector_t fp32_output = FNBLAS_VECTOR_INITIALIZER;
    fnblas_vector_t bf16_output = FNBLAS_VECTOR_INITIALIZER;
    fnblas_vector_create(&fp32_output, fp4_result._n_elements, FP32);
    fnblas_vector_create(&bf16_output, fp4_result._n_elements, BF16);
    for (std::size_t index = 0; index < fp4_result._n_elements; ++index) {
        fnblas_scalar_t fp4 = make_scalar(get_value(&fp4_result, index), FP4);
        const fnblas_scalar_t fp32 = fnblas_scalar_cast_to(&fp4, FP32);
        const fnblas_scalar_t bf16 = fnblas_scalar_cast_to(&fp32, BF16);
        const float fp32_value = fnblas_scalar_as_unpacked_float(&fp32);
        const float bf16_value = fnblas_scalar_as_unpacked_float(&bf16);
        set_value(&fp32_output, index, fp32_value);
        set_value(&bf16_output, index, bf16_value);
    }
    check_vector(&fp32_output, {1.5f, 3.0f, 4.5f, 9.0f}, FP32,
                 "FP4 to FP32 cast");
    check_vector(&bf16_output, {1.5f, 3.0f, 4.5f, 9.0f}, BF16,
                 "FP4 result to BF16 cast");

    std::vector<byte_t> packed(
        bf16_output._n_elements * fnblas_dtype_size_of(BF16)
    );
    fnblas_vector_get_packed_buffer(&bf16_output, packed.data());
    for (std::size_t index = 0; index < bf16_output._n_elements; ++index) {
        std::uint16_t bits;
        std::memcpy(&bits, packed.data() + index * sizeof(bits), sizeof(bits));
        require_close(fnblas_dtype_bf16_bits_to_fp32(bits),
                      get_value(&bf16_output, index), "BF16 packed output");
    }

    fnblas_vector_destroy(&lhs);
    fnblas_vector_destroy(&rhs);
    fnblas_vector_destroy(&fp4_result);
    fnblas_vector_destroy(&fp32_output);
    fnblas_vector_destroy(&bf16_output);
}

void test_validation()
{
    fnblas_vector_t lhs = FNBLAS_VECTOR_INITIALIZER;
    fnblas_vector_t wrong_size = FNBLAS_VECTOR_INITIALIZER;
    fnblas_vector_t wrong_dtype = FNBLAS_VECTOR_INITIALIZER;
    fnblas_vector_t result = FNBLAS_VECTOR_INITIALIZER;
    fnblas_vector_create(&lhs, 4, FP4);
    fnblas_vector_create(&wrong_size, 3, FP4);
    fnblas_vector_create(&wrong_dtype, 4, FP32);
    require(fnblas_op_ve_add(&result, &lhs, &wrong_size) ==
                FNBLAS_ERR_MISMATCH,
            "size mismatch validation");
    require(fnblas_op_ve_add(&result, &lhs, &wrong_dtype) ==
                FNBLAS_ERR_MISMATCH,
            "dtype mismatch validation");
    fnblas_vector_destroy(&lhs);
    fnblas_vector_destroy(&wrong_size);
    fnblas_vector_destroy(&wrong_dtype);
}

} // namespace

int main()
{
    try {
        test_scalar_scalar();
        test_vector_scalar();
        test_vector_vector_and_dot();
        test_fp4_packed_buffer();
        test_fp4_result_to_bf16();
        test_validation();
        std::cout << "t3_blas_fp4_test: all tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "t3_blas_fp4_test: FAILED: " << error.what() << '\n';
        return 1;
    }
}
