#include "flexi_blas.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

fnblas_scalar_t make_float(float value)
{
    fnblas_scalar_t scalar{};
    require(fnblas_scalar_create_float(&scalar, value, FP32) == FNBLAS_SUCCESS, "create FP32 scalar");
    return scalar;
}

fnblas_scalar_t make_int(int32_t value, fnblas_dtype_t dtype)
{
    fnblas_scalar_t scalar{};
    require(fnblas_scalar_create_int(&scalar, value, dtype) == FNBLAS_SUCCESS, "create integer scalar");
    return scalar;
}

void test_float_to_integer_rounding_and_clamping()
{
    fnblas_scalar_t source = make_float(1.6f);
    fnblas_scalar_t converted = fnblas_scalar_cast_to(&source, INT8);
    require(fnblas_scalar_as_unpacked_int32(&converted) == 2, "positive rounding");

    source = make_float(-1.6f);
    converted = fnblas_scalar_cast_to(&source, INT8);
    require(fnblas_scalar_as_unpacked_int32(&converted) == -2, "negative rounding");

    source = make_float(1000.0f);
    converted = fnblas_scalar_cast_to(&source, INT8);
    require(fnblas_scalar_as_unpacked_int32(&converted) == INT8_MAX, "INT8 upper clamp");

    source = make_float(-1000.0f);
    converted = fnblas_scalar_cast_to(&source, INT8);
    require(fnblas_scalar_as_unpacked_int32(&converted) == INT8_MIN, "INT8 lower clamp");

    source = make_float(100.0f);
    converted = fnblas_scalar_cast_to(&source, INT4);
    require(fnblas_scalar_as_unpacked_int32(&converted) == 7, "INT4 upper clamp");

    source = make_float(-100.0f);
    converted = fnblas_scalar_cast_to(&source, INT4);
    require(fnblas_scalar_as_unpacked_int32(&converted) == -8, "INT4 lower clamp");

    source = make_float(std::numeric_limits<float>::infinity());
    converted = fnblas_scalar_cast_to(&source, INT32);
    require(fnblas_scalar_as_unpacked_int32(&converted) == INT32_MAX, "INT32 upper clamp");

    const fnblas_scalar_t int4 = make_int(100, INT4);
    require(fnblas_scalar_as_unpacked_int32(&int4) == 7, "INT4 scalar creation clamp");
}

void test_float_cast_preserves_unpacked_value()
{
    const float value = 1.234567f;
    const fnblas_scalar_t source = make_float(value);
    const fnblas_scalar_t fp16 = fnblas_scalar_cast_to(&source, FP16);
    const fnblas_scalar_t bf16 = fnblas_scalar_cast_to(&source, BF16);
    const fnblas_scalar_t fp4 = fnblas_scalar_cast_to(&source, FP4);
    require(fnblas_scalar_as_unpacked_float(&fp16) == value, "FP16 cast preserves unpacked value");
    require(fnblas_scalar_as_unpacked_float(&bf16) == value, "BF16 cast preserves unpacked value");
    require(fnblas_scalar_as_unpacked_float(&fp4) == value, "FP4 cast preserves unpacked value");
}

void test_signed_int8_and_saturating_arithmetic()
{
    const int8_t values[4] = {INT8_MIN, -1, 0, INT8_MAX};
    byte_t packed[4] = {};
    int8_t restored[4] = {};
    fnblas_vector_t vector = FNBLAS_VECTOR_INITIALIZER;
    require(fnblas_vector_create(&vector, 4, INT8) == FNBLAS_SUCCESS, "create INT8 vector");
    require(fnblas_vector_initialize_from_unpacked_buffer(&vector, (const byte_t*)values) == FNBLAS_SUCCESS, "initialize INT8 vector");
    require(fnblas_vector_get_packed_buffer(&vector, packed) == FNBLAS_SUCCESS, "pack INT8 vector");
    require(fnblas_vector_initialize_from_packed_buffer(&vector, packed) == FNBLAS_SUCCESS, "unpack INT8 vector");
    std::memcpy(restored, vector._buffer, sizeof(restored));
    require(std::memcmp(values, restored, sizeof(values)) == 0, "signed INT8 packed round trip");

    fnblas_scalar_t minimum{};
    fnblas_scalar_t maximum{};
    require(fnblas_op_vr_min(&minimum, &vector) == FNBLAS_SUCCESS, "signed INT8 minimum");
    require(fnblas_op_vr_max(&maximum, &vector) == FNBLAS_SUCCESS, "signed INT8 maximum");
    require(fnblas_scalar_as_unpacked_int32(&minimum) == INT8_MIN, "INT8 negative ordering");
    require(fnblas_scalar_as_unpacked_int32(&maximum) == INT8_MAX, "INT8 positive ordering");
    fnblas_vector_destroy(&vector);

    fnblas_scalar_t lhs = make_int(120, INT8);
    fnblas_scalar_t rhs = make_int(20, INT8);
    fnblas_scalar_t result{};
    require(fnblas_op_ss_add(&result, &lhs, &rhs) == FNBLAS_SUCCESS, "INT8 saturating add");
    require(fnblas_scalar_as_unpacked_int32(&result) == INT8_MAX, "INT8 addition clamp");

    lhs = make_int(-120, INT8);
    require(fnblas_op_ss_sub(&result, &lhs, &rhs) == FNBLAS_SUCCESS, "INT8 saturating subtract");
    require(fnblas_scalar_as_unpacked_int32(&result) == INT8_MIN, "INT8 subtraction clamp");
}

void test_packing_clamps_reduced_precision_types()
{
    float fp16_values[2] = {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()};
    uint16_t fp16_packed[2] = {};
    fnblas_vector_t fp16 = FNBLAS_VECTOR_INITIALIZER;
    require(fnblas_vector_create(&fp16, 2, FP16) == FNBLAS_SUCCESS, "create FP16 vector");
    require(fnblas_vector_initialize_from_unpacked_buffer(&fp16, (const byte_t*)fp16_values) == FNBLAS_SUCCESS, "initialize FP16 vector");
    require(fnblas_vector_get_packed_buffer(&fp16, (byte_t*)fp16_packed) == FNBLAS_SUCCESS, "pack FP16 vector");
    require(fnblas_dtype_fp16_bits_to_fp32(fp16_packed[0]) == 65504.0f, "FP16 positive clamp");
    require(fnblas_dtype_fp16_bits_to_fp32(fp16_packed[1]) == -65504.0f, "FP16 negative clamp");
    fnblas_vector_destroy(&fp16);

    float bf16_values[2] = {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()};
    uint16_t bf16_packed[2] = {};
    fnblas_vector_t bf16 = FNBLAS_VECTOR_INITIALIZER;
    require(fnblas_vector_create(&bf16, 2, BF16) == FNBLAS_SUCCESS, "create BF16 vector");
    require(fnblas_vector_initialize_from_unpacked_buffer(&bf16, (const byte_t*)bf16_values) == FNBLAS_SUCCESS, "initialize BF16 vector");
    require(fnblas_vector_get_packed_buffer(&bf16, (byte_t*)bf16_packed) == FNBLAS_SUCCESS, "pack BF16 vector");
    require(bf16_packed[0] == UINT16_C(0x7F7F), "BF16 positive clamp");
    require(bf16_packed[1] == UINT16_C(0xFF7F), "BF16 negative clamp");
    fnblas_vector_destroy(&bf16);

    float fp4_values[2] = {100.0f, -100.0f};
    byte_t fp4_packed = 0;
    fnblas_vector_t fp4 = FNBLAS_VECTOR_INITIALIZER;
    require(fnblas_vector_create(&fp4, 2, FP4) == FNBLAS_SUCCESS, "create FP4 vector");
    require(fnblas_vector_initialize_from_unpacked_buffer(&fp4, (const byte_t*)fp4_values) == FNBLAS_SUCCESS, "initialize FP4 vector");
    require(fnblas_vector_get_packed_buffer(&fp4, &fp4_packed) == FNBLAS_SUCCESS, "pack FP4 vector");
    require(fp4_packed == UINT8_C(0xF7), "FP4 clamp");
    fnblas_vector_destroy(&fp4);

    int32_t int4_values[2] = {100, -100};
    byte_t int4_packed = 0;
    fnblas_vector_t int4 = FNBLAS_VECTOR_INITIALIZER;
    require(fnblas_vector_create(&int4, 2, INT4) == FNBLAS_SUCCESS, "create INT4 vector");
    require(fnblas_vector_initialize_from_unpacked_buffer(&int4, (const byte_t*)int4_values) == FNBLAS_SUCCESS, "initialize INT4 vector");
    require(fnblas_vector_get_packed_buffer(&int4, &int4_packed) == FNBLAS_SUCCESS, "pack INT4 vector");
    require(int4_packed == UINT8_C(0x87), "INT4 clamp");
    fnblas_vector_destroy(&int4);
}

}

int main()
{
    try {
        test_float_to_integer_rounding_and_clamping();
        test_float_cast_preserves_unpacked_value();
        test_signed_int8_and_saturating_arithmetic();
        test_packing_clamps_reduced_precision_types();
        std::cout << "All dtype conversion tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Test failure: " << error.what() << '\n';
        return 1;
    }
}
