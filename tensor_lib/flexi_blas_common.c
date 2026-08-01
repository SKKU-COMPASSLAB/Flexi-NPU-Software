#include "flexi_blas_internal.h"

#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static const float FP4_E2M1_MAGNITUDES[8] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f
};

int _fnblas_buffer_is_allocated(uint8_t status)
{
    return (status & FNBLAS_STATUS_BUFFER_ALLOCATED) != 0;
}

int _fnblas_buffer_is_view(uint8_t status)
{
    return (status & FNBLAS_STATUS_VIEW) != 0;
}

int _fnblas_dtype_is_float(fnblas_dtype_t dtype)
{
    return dtype == FP4 || dtype == FP8 || dtype == FP16 ||
           dtype == BF16 || dtype == FP32;
}

int _fnblas_dtype_is_integer(fnblas_dtype_t dtype)
{
    return dtype == INT4 || dtype == INT8 || dtype == INT32;
}

int _fnblas_dtype_is_valid(fnblas_dtype_t dtype)
{
    return _fnblas_dtype_is_float(dtype) ||
           _fnblas_dtype_is_integer(dtype);
}

int _fnblas_dtype_is_same_family(fnblas_dtype_t lhs, fnblas_dtype_t rhs)
{
    return (_fnblas_dtype_is_float(lhs) && _fnblas_dtype_is_float(rhs)) || (_fnblas_dtype_is_integer(lhs) && _fnblas_dtype_is_integer(rhs));
}

size_t _fnblas_dtype_unpacked_size_of(fnblas_dtype_t dtype)
{
    if (_fnblas_dtype_is_float(dtype))
        return sizeof(float);
    if (dtype == INT4 || dtype == INT32)
        return sizeof(int32_t);
    if (dtype == INT8)
        return sizeof(int8_t);
    errno = EINVAL;
    return 0;
}

size_t _fnblas_packed_buffer_size(size_t n_elements, fnblas_dtype_t dtype)
{
    const size_t pack_size = fnblas_dtype_pack_size_of(dtype);
    const size_t element_size = fnblas_dtype_size_of(dtype);
    if (pack_size == 0 || element_size == 0)
        return 0;
    if (n_elements > SIZE_MAX / element_size) {
        errno = EOVERFLOW;
        return 0;
    }
    return (n_elements * element_size + pack_size - 1) / pack_size;
}

size_t fnblas_dtype_size_of(fnblas_dtype_t dtype)
{
    switch (dtype) {
        case FP4:
        case INT4:
        case FP8:
        case BYTE:
        case INT8:  return 1;
        case FP16:
        case BF16:  return 2;
        case FP32:
        case INT32: return 4;
        default:
            errno = EINVAL;
            return 0;
    }
}

size_t fnblas_dtype_pack_size_of(fnblas_dtype_t dtype)
{
    switch (dtype) {
        case FP4:
        case INT4: return 2;
        case BYTE:
        case FP8:
        case INT8:
        case FP16:
        case BF16:
        case FP32:
        case INT32: return 1;
        default:
            errno = EINVAL;
            return 0;
    }
}

uint32_t fnblas_dtype_float_to_bits(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float fnblas_dtype_bits_to_float(uint32_t bits)
{
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

float fnblas_dtype_softfloat_bf16_to_fp32(float value)
{
    uint32_t bits = fnblas_dtype_float_to_bits(value);
    const uint32_t abs_bits = bits & UINT32_C(0x7FFFFFFF);
    if (abs_bits > UINT32_C(0x7F800000)) {
        return fnblas_dtype_bits_to_float(
            (bits & UINT32_C(0x80000000)) | UINT32_C(0x7FC00000)
        );
    }
    bits += UINT32_C(0x7FFF) + ((bits >> 16) & 1u);
    return fnblas_dtype_bits_to_float(bits & UINT32_C(0xFFFF0000));
}

uint16_t fnblas_dtype_fp32_to_fp16_bits(float value)
{
    const uint32_t bits = fnblas_dtype_float_to_bits(value);
    const uint16_t sign = (uint16_t)((bits >> 16) & 0x8000u);
    const uint32_t exponent = (bits >> 23) & 0xFFu;
    const uint32_t mantissa = bits & 0x7FFFFFu;
    int unbiased_exponent;

    if (exponent == 0xFFu)
        return (uint16_t)(sign | (mantissa == 0 ? 0x7C00u : 0x7E00u));

    unbiased_exponent = (int)exponent - 127;
    if (unbiased_exponent > 15)
        return (uint16_t)(sign | 0x7C00u);
    if (unbiased_exponent >= -14) {
        uint32_t half_exponent = (uint32_t)(unbiased_exponent + 15);
        uint32_t half_mantissa = mantissa >> 13;
        const uint32_t remainder = mantissa & 0x1FFFu;
        if (remainder > 0x1000u ||
            (remainder == 0x1000u && (half_mantissa & 1u))) {
            ++half_mantissa;
            if (half_mantissa == 0x400u) {
                half_mantissa = 0;
                if (++half_exponent >= 0x1Fu)
                    return (uint16_t)(sign | 0x7C00u);
            }
        }
        return (uint16_t)(
            sign | (half_exponent << 10) | half_mantissa
        );
    }
    if (unbiased_exponent < -25)
        return sign;
    {
        const uint32_t significand = mantissa | 0x800000u;
        const unsigned shift = (unsigned)(-unbiased_exponent - 1);
        uint32_t half_mantissa = significand >> shift;
        const uint32_t mask = (UINT32_C(1) << shift) - 1;
        const uint32_t remainder = significand & mask;
        const uint32_t halfway = UINT32_C(1) << (shift - 1);
        if (remainder > halfway ||
            (remainder == halfway && (half_mantissa & 1u)))
            ++half_mantissa;
        return (uint16_t)(sign | half_mantissa);
    }
}

float fnblas_dtype_fp16_bits_to_fp32(uint16_t value)
{
    const uint32_t sign = ((uint32_t)value & 0x8000u) << 16;
    uint32_t exponent = (value >> 10) & 0x1Fu;
    uint32_t mantissa = value & 0x3FFu;
    uint32_t bits;

    if (exponent == 0) {
        int unbiased_exponent = -14;
        if (mantissa == 0)
            return fnblas_dtype_bits_to_float(sign);
        while ((mantissa & 0x400u) == 0) {
            mantissa <<= 1;
            --unbiased_exponent;
        }
        mantissa &= 0x3FFu;
        bits = sign |
            ((uint32_t)(unbiased_exponent + 127) << 23) |
            (mantissa << 13);
    } else if (exponent == 0x1Fu) {
        bits = sign | 0x7F800000u | (mantissa << 13);
        if (mantissa != 0)
            bits |= 0x00400000u;
    } else {
        bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }
    return fnblas_dtype_bits_to_float(bits);
}

float fnblas_dtype_softfloat_fp16_to_fp32(float value)
{
    return fnblas_dtype_fp16_bits_to_fp32(
        fnblas_dtype_fp32_to_fp16_bits(value)
    );
}

float fnblas_dtype_bf16_bits_to_fp32(uint16_t value)
{
    return fnblas_dtype_bits_to_float((uint32_t)value << 16);
}

float _fnblas_fp4_bits_to_fp32(byte_t value)
{
    const byte_t fp4 = (byte_t)(value & 0x0Fu);
    const float magnitude = FP4_E2M1_MAGNITUDES[fp4 & 0x07u];
    return (fp4 & 0x08u) != 0 ? -magnitude : magnitude;
}

byte_t _fnblas_fp32_to_fp4_bits(float value)
{
    byte_t best_code = 0;
    float best_distance = INFINITY;
    byte_t code;
    const float magnitude = fabsf(value);

    if (isnan(value)) {
        errno = EDOM;
        return 0;
    }
    if (isinf(magnitude) || magnitude >= FP4_E2M1_MAGNITUDES[7]) {
        best_code = 7;
    } else {
        for (code = 0; code < 8; ++code) {
            const float distance =
                fabsf(magnitude - FP4_E2M1_MAGNITUDES[code]);
            if (distance < best_distance ||
                (distance == best_distance &&
                 (code & 1u) == 0 && (best_code & 1u) != 0)) {
                best_distance = distance;
                best_code = code;
            }
        }
    }
    return (byte_t)((signbit(value) ? 0x08u : 0u) | best_code);
}

float _fnblas_fp8_bits_to_fp32(byte_t value)
{
    const unsigned sign = value >> 7;
    const unsigned exponent = (value >> 3) & 0x0Fu;
    const unsigned mantissa = value & 0x07u;
    float result;
    if (exponent == 0) {
        result = mantissa == 0
            ? 0.0f
            : ldexpf((float)mantissa / 8.0f, -6);
    } else if (exponent == 0x0Fu) {
        result = mantissa == 0 ? INFINITY : NAN;
    } else {
        result = ldexpf(
            1.0f + (float)mantissa / 8.0f,
            (int)exponent - 7
        );
    }
    return sign ? -result : result;
}

byte_t _fnblas_fp32_to_fp8_bits(float value)
{
    byte_t best = 0;
    float best_distance = INFINITY;
    unsigned code;
    const unsigned sign = signbit(value) ? 0x80u : 0u;
    const float magnitude = fabsf(value);
    if (isnan(value))
        return (byte_t)(sign | 0x7Fu);
    if (isinf(value))
        return (byte_t)(sign | 0x78u);
    for (code = 0; code < 0x78u; ++code) {
        const float candidate = _fnblas_fp8_bits_to_fp32((byte_t)code);
        const float distance = fabsf(magnitude - candidate);
        if (distance < best_distance ||
            (distance == best_distance &&
             (code & 1u) == 0 && (best & 1u) != 0)) {
            best = (byte_t)code;
            best_distance = distance;
        }
    }
    return (byte_t)(sign | best);
}

uint16_t _fnblas_fp32_to_bf16_bits(float value)
{
    const float rounded = fnblas_dtype_softfloat_bf16_to_fp32(value);
    return (uint16_t)(fnblas_dtype_float_to_bits(rounded) >> 16);
}

int32_t _fnblas_sign_extend_int4(byte_t value)
{
    const byte_t nibble = (byte_t)(value & 0x0Fu);
    return (nibble & 0x08u) != 0
        ? (int32_t)nibble - 16
        : (int32_t)nibble;
}

byte_t _fnblas_int32_to_int4_bits(int32_t value)
{
    if (value < -8) {
        errno = ERANGE;
        value = -8;
    } else if (value > 7) {
        errno = ERANGE;
        value = 7;
    }
    return (byte_t)((uint32_t)value & 0x0Fu);
}

int32_t _fnblas_int32_arithmetic(
    int32_t lhs, int32_t rhs, fnblas_arithmetic_op_t op)
{
    int64_t value;
    switch (op) {
        case ADD: value = (int64_t)lhs + (int64_t)rhs; break;
        case SUBTRACT: value = (int64_t)lhs - (int64_t)rhs; break;
        case MULTIPLY: value = (int64_t)lhs * (int64_t)rhs; break;
        case DIVIDE:
            if (rhs == 0) {
                errno = EDOM;
                return 0;
            }
            value = lhs == INT32_MIN && rhs == -1 ? INT32_MAX : lhs / rhs;
            break;
        default:
            errno = EINVAL;
            return 0;
    }
    if (value < INT32_MIN) {
        errno = ERANGE;
        return INT32_MIN;
    }
    if (value > INT32_MAX) {
        errno = ERANGE;
        return INT32_MAX;
    }
    return (int32_t)value;
}

int8_t _fnblas_int8_arithmetic(
    int8_t lhs, int8_t rhs, fnblas_arithmetic_op_t op)
{
    int32_t value;
    switch (op) {
        case ADD: value = (int32_t)lhs + (int32_t)rhs; break;
        case SUBTRACT: value = (int32_t)lhs - (int32_t)rhs; break;
        case MULTIPLY: value = (int32_t)lhs * (int32_t)rhs; break;
        case DIVIDE:
            if (rhs == 0) {
                errno = EDOM;
                return 0;
            }
            value = lhs == INT8_MIN && rhs == -1 ? INT8_MAX : lhs / rhs;
            break;
        default:
            errno = EINVAL;
            value = 0;
            break;
    }
    if (value < INT8_MIN) {
        errno = ERANGE;
        return INT8_MIN;
    }
    if (value > INT8_MAX) {
        errno = ERANGE;
        return INT8_MAX;
    }
    return (int8_t)value;
}

float _fnblas_float_arithmetic(
    float lhs, float rhs, fnblas_arithmetic_op_t op)
{
    switch (op) {
        case ADD: return lhs + rhs;
        case SUBTRACT: return lhs - rhs;
        case MULTIPLY: return lhs * rhs;
        case DIVIDE: return lhs / rhs;
        default:
            errno = EINVAL;
            return 0.0f;
    }
}

#if !FNBLAS_MEM_DYNAMIC_ENABLE
static fnblas_static_memory_t fnblas_static_memory;
static int fnblas_static_memory_registered = 0;

fnblas_error_t fnblas_create_static_memory(byte_t *mem, size_t size, size_t n_buffers) {
    uintptr_t begin;
    uintptr_t end;
    size_t slot;
    if (mem == NULL || size == 0 || n_buffers == 0 || n_buffers > FNBLAS_MAX_N_BUFFERS_PER_STATIC_MEMORY) {
        errno = EINVAL;
        return FNBLAS_ERR_MISMATCH;
    }
    if (fnblas_static_memory_registered) {
        for (slot = 0; slot < fnblas_static_memory.n_buffers; slot++) {
            if (fnblas_static_memory._buffer_used[slot]) {
                errno = EBUSY;
                return FNBLAS_ERR_UNKNOWN;
            }
        }
    }
    begin = (uintptr_t)mem;
    if (size > UINTPTR_MAX - begin) {
        errno = EOVERFLOW;
        return FNBLAS_ERR_MISMATCH;
    }
    end = begin + size;
    if (end <= begin) {
        errno = EOVERFLOW;
        return FNBLAS_ERR_MISMATCH;
    }
    memset(&fnblas_static_memory, 0, sizeof(fnblas_static_memory));
    fnblas_static_memory._mem = mem;
    fnblas_static_memory._size = size;
    fnblas_static_memory.n_buffers = n_buffers;
    fnblas_static_memory_registered = 1;
    return FNBLAS_SUCCESS;
}
#endif

fnblas_error_t _fnblas_allocate_buffer(
    byte_t** buffer, size_t n_elements, fnblas_dtype_t dtype)
{
    const size_t element_size = _fnblas_dtype_unpacked_size_of(dtype);
    size_t size;
    size_t allocation_size;

    if (buffer == NULL || element_size == 0) {
        errno = EINVAL;
        return FNBLAS_ERR_UNKNOWN;
    }

    *buffer = NULL;
    if (n_elements == 0)
        return FNBLAS_SUCCESS;
    if (n_elements > SIZE_MAX / element_size) {
        errno = EOVERFLOW;
        return FNBLAS_ERR_MALLOC_FAIL;
    }
    size = n_elements * element_size;
    if (size > SIZE_MAX - (FNBLAS_MEM_ALIGNMENT - 1)) {
        errno = EOVERFLOW;
        return FNBLAS_ERR_MALLOC_FAIL;
    }
    allocation_size = (size + FNBLAS_MEM_ALIGNMENT - 1) &
                      ~(size_t)(FNBLAS_MEM_ALIGNMENT - 1);
#if FNBLAS_MEM_DYNAMIC_ENABLE
    *buffer = (byte_t*)aligned_alloc(FNBLAS_MEM_ALIGNMENT, allocation_size);
    if (*buffer == NULL) {
        errno = ENOMEM;
        return FNBLAS_ERR_MALLOC_FAIL;
    }
    memset(*buffer, 0, size);
    return FNBLAS_SUCCESS;
#else
    uintptr_t pool_begin;
    uintptr_t pool_end;
    uintptr_t candidate;
    size_t free_slot = SIZE_MAX;
    size_t slot;
    if (!fnblas_static_memory_registered) {
        errno = ENODEV;
        return FNBLAS_ERR_UNKNOWN;
    }
    for (slot = 0; slot < fnblas_static_memory.n_buffers; slot++) {
        if (!fnblas_static_memory._buffer_used[slot]) {
            free_slot = slot;
            break;
        }
    }
    if (free_slot == SIZE_MAX) {
        errno = ENOSPC;
        return FNBLAS_ERR_MALLOC_FAIL;
    }
    pool_begin = (uintptr_t)fnblas_static_memory._mem;
    pool_end = pool_begin + fnblas_static_memory._size;
    if (pool_begin > UINTPTR_MAX - (FNBLAS_MEM_ALIGNMENT - 1)) {
        errno = EOVERFLOW;
        return FNBLAS_ERR_MALLOC_FAIL;
    }
    candidate = (pool_begin + FNBLAS_MEM_ALIGNMENT - 1) & ~(uintptr_t)(FNBLAS_MEM_ALIGNMENT - 1);
    for (;;) {
        uintptr_t next_begin = pool_end;
        uintptr_t next_end = pool_end;
        for (slot = 0; slot < fnblas_static_memory.n_buffers; slot++) {
            uintptr_t block_begin;
            if (!fnblas_static_memory._buffer_used[slot]) continue;
            block_begin = (uintptr_t)fnblas_static_memory._buffer_ptrs[slot];
            if (block_begin >= candidate && block_begin < next_begin) {
                next_begin = block_begin;
                next_end = block_begin + fnblas_static_memory._buffer_sizes[slot];
            }
        }
        if (candidate <= next_begin && allocation_size <= next_begin - candidate) break;
        if (next_begin == pool_end || next_end > UINTPTR_MAX - (FNBLAS_MEM_ALIGNMENT - 1)) {
            errno = ENOMEM;
            return FNBLAS_ERR_MALLOC_FAIL;
        }
        candidate = (next_end + FNBLAS_MEM_ALIGNMENT - 1) & ~(uintptr_t)(FNBLAS_MEM_ALIGNMENT - 1);
        if (candidate > pool_end) {
            errno = ENOMEM;
            return FNBLAS_ERR_MALLOC_FAIL;
        }
    }
    *buffer = (byte_t*)candidate;
    fnblas_static_memory._buffer_ptrs[free_slot] = *buffer;
    fnblas_static_memory._buffer_sizes[free_slot] = allocation_size;
    fnblas_static_memory._buffer_used[free_slot] = 1;
    memset(*buffer, 0, allocation_size);
    return FNBLAS_SUCCESS;
#endif
}

void _fnblas_deallocate_buffer(byte_t* buffer)
{
#if FNBLAS_MEM_DYNAMIC_ENABLE
    free(buffer);
#else
    size_t slot;
    if (buffer == NULL) return;
    if (!fnblas_static_memory_registered) {
        errno = ENODEV;
        return;
    }
    for (slot = 0; slot < fnblas_static_memory.n_buffers; slot++) {
        if (fnblas_static_memory._buffer_used[slot] && fnblas_static_memory._buffer_ptrs[slot] == buffer) {
            fnblas_static_memory._buffer_ptrs[slot] = NULL;
            fnblas_static_memory._buffer_sizes[slot] = 0;
            fnblas_static_memory._buffer_used[slot] = 0;
            return;
        }
    }
    errno = EINVAL;
#endif
}

static float fnblas_clamp_float_for_packing(float value, fnblas_dtype_t dtype)
{
    const float maximum = dtype == FP16
        ? 65504.0f
        : fnblas_dtype_bits_to_float(UINT32_C(0x7F7F0000));
    if (isnan(value))
        return value;
    if (value > maximum) {
        errno = ERANGE;
        return maximum;
    }
    if (value < -maximum) {
        errno = ERANGE;
        return -maximum;
    }
    return value;
}

void _fnblas_pack_buffer(
    const byte_t* unpacked, size_t n, fnblas_dtype_t dtype, byte_t* packed)
{
    size_t index;
    if (n != 0 && (unpacked == NULL || packed == NULL)) {
        errno = EINVAL;
        return;
    }
    if (dtype == FP4 || dtype == INT4) {
        memset(packed, 0, _fnblas_packed_buffer_size(n, dtype));
        for (index = 0; index < n; ++index) {
            byte_t bits;
            if (dtype == FP4)
                bits = _fnblas_fp32_to_fp4_bits(
                    ((const float*)unpacked)[index]
                );
            else
                bits = _fnblas_int32_to_int4_bits(
                    ((const int32_t*)unpacked)[index]
                );
            if ((index & 1u) == 0)
                packed[index / 2] |= bits;
            else
                packed[index / 2] |= (byte_t)(bits << 4);
        }
    } else if (dtype == FP8) {
        for (index = 0; index < n; ++index)
            packed[index] = _fnblas_fp32_to_fp8_bits(
                ((const float*)unpacked)[index]
            );
    } else if (dtype == FP16 || dtype == BF16) {
        for (index = 0; index < n; ++index) {
            const float value = fnblas_clamp_float_for_packing(((const float*)unpacked)[index], dtype);
            const uint16_t bits = dtype == FP16
                ? fnblas_dtype_fp32_to_fp16_bits(value)
                : _fnblas_fp32_to_bf16_bits(value);
            memcpy(packed + index * sizeof(bits), &bits, sizeof(bits));
        }
    } else {
        const size_t size = n * _fnblas_dtype_unpacked_size_of(dtype);
        if (size != 0)
            memcpy(packed, unpacked, size);
    }
}

void _fnblas_unpack_buffer(
    byte_t* unpacked, size_t n, fnblas_dtype_t dtype, const byte_t* packed)
{
    size_t index;
    if (n != 0 && (unpacked == NULL || packed == NULL)) {
        errno = EINVAL;
        return;
    }
    if (dtype == FP4 || dtype == INT4) {
        for (index = 0; index < n; ++index) {
            const byte_t byte = packed[index / 2];
            const byte_t bits = (index & 1u) == 0
                ? (byte_t)(byte & 0x0Fu)
                : (byte_t)((byte >> 4) & 0x0Fu);
            if (dtype == FP4)
                ((float*)unpacked)[index] =
                    _fnblas_fp4_bits_to_fp32(bits);
            else
                ((int32_t*)unpacked)[index] =
                    _fnblas_sign_extend_int4(bits);
        }
    } else if (dtype == FP8) {
        for (index = 0; index < n; ++index)
            ((float*)unpacked)[index] =
                _fnblas_fp8_bits_to_fp32(packed[index]);
    } else if (dtype == FP16 || dtype == BF16) {
        for (index = 0; index < n; ++index) {
            uint16_t bits;
            memcpy(&bits, packed + index * sizeof(bits), sizeof(bits));
            ((float*)unpacked)[index] = dtype == FP16
                ? fnblas_dtype_fp16_bits_to_fp32(bits)
                : fnblas_dtype_bf16_bits_to_fp32(bits);
        }
    } else {
        const size_t size = n * _fnblas_dtype_unpacked_size_of(dtype);
        if (size != 0)
            memcpy(unpacked, packed, size);
    }
}
