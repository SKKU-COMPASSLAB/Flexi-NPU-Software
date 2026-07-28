#ifndef FLEXI_BLAS_INTERNAL_H
#define FLEXI_BLAS_INTERNAL_H

#include "flexi_blas.h"

#include <stddef.h>
#include <stdint.h>

#define FNBLAS_STATUS_BUFFER_ALLOCATED UINT8_C(0x01)
#define FNBLAS_STATUS_VIEW UINT8_C(0x02)

int _fnblas_buffer_is_allocated(uint8_t status);
int _fnblas_buffer_is_view(uint8_t status);

int _fnblas_dtype_is_float(fnblas_dtype_t dtype);
int _fnblas_dtype_is_integer(fnblas_dtype_t dtype);
int _fnblas_dtype_is_valid(fnblas_dtype_t dtype);
size_t _fnblas_dtype_unpacked_size_of(fnblas_dtype_t dtype);
size_t _fnblas_packed_buffer_size(size_t n_elements, fnblas_dtype_t dtype);

float _fnblas_fp4_bits_to_fp32(byte_t value);
byte_t _fnblas_fp32_to_fp4_bits(float value);
float _fnblas_fp8_bits_to_fp32(byte_t value);
byte_t _fnblas_fp32_to_fp8_bits(float value);
uint16_t _fnblas_fp32_to_bf16_bits(float value);
int32_t _fnblas_sign_extend_int4(byte_t value);
byte_t _fnblas_int32_to_int4_bits(int32_t value);

int32_t _fnblas_int32_arithmetic(
    int32_t lhs, int32_t rhs, fnblas_arithmetic_op_t op
);
uint8_t _fnblas_uint8_arithmetic(
    uint8_t lhs, uint8_t rhs, fnblas_arithmetic_op_t op
);
float _fnblas_float_arithmetic(
    float lhs, float rhs, fnblas_arithmetic_op_t op
);

fnblas_error_t _fnblas_allocate_buffer(
    byte_t** buffer,
    size_t n_elements,
    fnblas_dtype_t dtype
);
void _fnblas_deallocate_buffer(byte_t* buffer);
void _fnblas_pack_buffer(
    const byte_t* unpacked_buffer,
    size_t n_elements,
    fnblas_dtype_t dtype,
    byte_t* packed_buffer
);
void _fnblas_unpack_buffer(
    byte_t* unpacked_buffer,
    size_t n_elements,
    fnblas_dtype_t dtype,
    const byte_t* packed_buffer
);

#endif
