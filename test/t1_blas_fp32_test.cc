// Comprehensive FP32 validation for the C flexi_blas API.
#include "flexi_blas.h"

#include <cerrno>
#include <cmath>
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

fnblas_vector_t make_vector(const std::vector<float>& values)
{
    fnblas_vector_t result = FNBLAS_VECTOR_INITIALIZER;
    fnblas_vector_create(&result, values.size(), FP32);
    require(values.empty() || result._buffer != nullptr, "vector allocation");
    for (std::size_t index = 0; index < values.size(); ++index)
        set_value(&result, index, values[index]);
    return result;
}

void check_vector(
    const fnblas_vector_t* actual,
    const std::vector<float>& expected,
    const std::string& message
)
{
    require(fnblas_vector_dtype(actual) == FP32, message + ": dtype");
    require(fnblas_vector_n_elements(actual) == expected.size(),
            message + ": size");
    for (std::size_t index = 0; index < expected.size(); ++index)
        require_close(get_value(actual, index), expected[index], message);
}

void test_scalar_scalar()
{
    fnblas_scalar_t lhs{}, rhs{}, result{};
    fnblas_scalar_create_float(&lhs, 12.0f, FP32);
    fnblas_scalar_create_float(&rhs, 3.0f, FP32);

    fnblas_op_ss_add(&result, &lhs, &rhs);
    require_close(fnblas_scalar_as_unpacked_float(&result), 15.0f,
                  "scalar addition");
    fnblas_op_ss_sub(&result, &lhs, &rhs);
    require_close(fnblas_scalar_as_unpacked_float(&result), 9.0f,
                  "scalar subtraction");
    fnblas_op_ss_mul(&result, &lhs, &rhs);
    require_close(fnblas_scalar_as_unpacked_float(&result), 36.0f,
                  "scalar multiplication");
    fnblas_op_ss_div(&result, &lhs, &rhs);
    require_close(fnblas_scalar_as_unpacked_float(&result), 4.0f,
                  "scalar division");

    const fnblas_scalar_t as_int = fnblas_scalar_cast_to(&lhs, INT32);
    require(fnblas_scalar_as_unpacked_int32(&as_int) == 12,
            "scalar FP32 to INT32 cast");
    require(fnblas_scalar_size(&lhs) == sizeof(float), "scalar size");
    require(fnblas_scalar_dtype(&lhs) == FP32, "scalar dtype");
}

void test_vector_scalar()
{
    fnblas_vector_t input = make_vector({1.0f, 2.0f, 4.0f, 8.0f});
    fnblas_scalar_t scalar{};
    fnblas_scalar_create_float(&scalar, 2.0f, FP32);

    fnblas_vector_t result = FNBLAS_VECTOR_INITIALIZER;
    fnblas_op_sv_add(&result, &input, &scalar);
    check_vector(&result, {3.0f, 4.0f, 6.0f, 10.0f}, "vector + scalar");
    fnblas_vector_destroy(&result);

    fnblas_op_sv_sub(&result, &input, &scalar);
    check_vector(&result, {-1.0f, 0.0f, 2.0f, 6.0f}, "vector - scalar");
    fnblas_vector_destroy(&result);

    fnblas_op_sv_mul(&result, &input, &scalar);
    check_vector(&result, {2.0f, 4.0f, 8.0f, 16.0f}, "vector * scalar");
    fnblas_vector_destroy(&result);

    fnblas_op_sv_div(&result, &input, &scalar);
    check_vector(&result, {0.5f, 1.0f, 2.0f, 4.0f}, "vector / scalar");
    fnblas_vector_destroy(&result);
    fnblas_vector_destroy(&input);
}

void test_vector_vector_and_dot()
{
    fnblas_vector_t lhs = make_vector({2.0f, 4.0f, 8.0f, 16.0f});
    fnblas_vector_t rhs = make_vector({1.0f, 2.0f, 4.0f, 8.0f});
    fnblas_vector_t result = FNBLAS_VECTOR_INITIALIZER;

    fnblas_op_ve_add(&result, &lhs, &rhs);
    check_vector(&result, {3.0f, 6.0f, 12.0f, 24.0f}, "vector addition");
    fnblas_vector_destroy(&result);
    fnblas_op_ve_sub(&result, &lhs, &rhs);
    check_vector(&result, {1.0f, 2.0f, 4.0f, 8.0f}, "vector subtraction");
    fnblas_vector_destroy(&result);
    fnblas_op_ve_mul(&result, &lhs, &rhs);
    check_vector(&result, {2.0f, 8.0f, 32.0f, 128.0f},
                 "vector multiplication");
    fnblas_vector_destroy(&result);
    fnblas_op_ve_div(&result, &lhs, &rhs);
    check_vector(&result, {2.0f, 2.0f, 2.0f, 2.0f}, "vector division");
    fnblas_vector_destroy(&result);

    fnblas_scalar_t dot{};
    fnblas_op_vv_dot(&dot, &lhs, &rhs);
    require_close(fnblas_scalar_as_unpacked_float(&dot), 170.0f,
                  "dot product");
    fnblas_vector_destroy(&lhs);
    fnblas_vector_destroy(&rhs);

    fnblas_vector_t empty_lhs = FNBLAS_VECTOR_INITIALIZER;
    fnblas_vector_t empty_rhs = FNBLAS_VECTOR_INITIALIZER;
    fnblas_vector_create(&empty_lhs, 0, FP32);
    fnblas_vector_create(&empty_rhs, 0, FP32);
    fnblas_op_vv_dot(&dot, &empty_lhs, &empty_rhs);
    require_close(fnblas_scalar_as_unpacked_float(&dot), 0.0f,
                  "empty dot product");
    fnblas_vector_destroy(&empty_lhs);
    fnblas_vector_destroy(&empty_rhs);
}

void test_buffer_api()
{
    fnblas_vector_t source = make_vector({1.0f, -2.5f, 3.25f});
    const std::size_t packed_size =
        source._n_elements * fnblas_dtype_size_of(FP32);
    std::vector<byte_t> packed(packed_size);
    fnblas_vector_get_packed_buffer(&source, packed.data());
    require(std::memcmp(packed.data(), source._buffer, packed_size) == 0,
            "FP32 packed copy");

    fnblas_vector_t restored = FNBLAS_VECTOR_INITIALIZER;
    fnblas_vector_create(&restored, source._n_elements, FP32);
    fnblas_vector_initialize_from_packed_buffer(&restored, packed.data());
    check_vector(&restored, {1.0f, -2.5f, 3.25f}, "FP32 round trip");
    fnblas_vector_destroy(&source);
    fnblas_vector_destroy(&restored);
}

void test_validation()
{
    fnblas_vector_t lhs = FNBLAS_VECTOR_INITIALIZER;
    fnblas_vector_t wrong_size = FNBLAS_VECTOR_INITIALIZER;
    fnblas_vector_t wrong_dtype = FNBLAS_VECTOR_INITIALIZER;
    fnblas_vector_t result = FNBLAS_VECTOR_INITIALIZER;
    fnblas_vector_create(&lhs, 4, FP32);
    fnblas_vector_create(&wrong_size, 3, FP32);
    fnblas_vector_create(&wrong_dtype, 4, BF16);

    require(fnblas_op_ve_add(&result, &lhs, &wrong_size) ==
                FNBLAS_ERR_MISMATCH &&
            result._buffer == nullptr,
            "size mismatch validation");
    require(fnblas_op_ve_add(&result, &lhs, &wrong_dtype) ==
                FNBLAS_ERR_MISMATCH &&
            result._buffer == nullptr,
            "dtype mismatch validation");

    byte_t* const lhs_buffer = lhs._buffer;
    require(fnblas_vector_create(&lhs, 8, FP32) ==
                FNBLAS_ERR_UNKNOWN,
            "duplicate vector create validation");
    require(lhs._buffer == lhs_buffer && lhs._n_elements == 4,
            "duplicate create changed vector");

    fnblas_vector_t reusable = FNBLAS_VECTOR_INITIALIZER;
    fnblas_vector_create(&reusable, 4, FP32);
    byte_t* const reusable_buffer = reusable._buffer;
    require(fnblas_op_ve_add(&reusable, &lhs, &wrong_dtype) ==
                FNBLAS_ERR_MISMATCH,
            "preallocated result dtype validation");
    require(reusable._buffer == reusable_buffer,
            "mismatch changed preallocated result");

    require(fnblas_op_ve_add(&reusable, &lhs, &lhs) ==
                FNBLAS_SUCCESS,
            "preallocated result operation failed");
    require(reusable._buffer == reusable_buffer,
            "operator reallocated result buffer");
    for (std::size_t index = 0; index < reusable._n_elements; ++index)
        require_close(get_value(&reusable, index), 0.0f,
                      "preallocated result value");

    fnblas_vector_destroy(&lhs);
    fnblas_vector_destroy(&wrong_size);
    fnblas_vector_destroy(&wrong_dtype);
    fnblas_vector_destroy(&reusable);
}

} // namespace

int main()
{
    try {
        test_scalar_scalar();
        test_vector_scalar();
        test_vector_vector_and_dot();
        test_buffer_api();
        test_validation();
        std::cout << "t1_blas_fp32_test: all tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "t1_blas_fp32_test: FAILED: " << error.what() << '\n';
        return 1;
    }
}
