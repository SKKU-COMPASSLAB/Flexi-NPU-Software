#ifndef FLEXI_BLAS_COMMON_H
#define FLEXI_BLAS_COMMON_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef FNBLAS_MEM_DYNAMIC_ENABLE
#define FNBLAS_MEM_DYNAMIC_ENABLE   1
#endif

#ifndef FNBLAS_MAX_N_BUFFERS_PER_STATIC_MEMORY
#define FNBLAS_MAX_N_BUFFERS_PER_STATIC_MEMORY  1024
#endif

#ifndef FNBLAS_MEM_ALIGNMENT
#define FNBLAS_MEM_ALIGNMENT 64
#endif

typedef uint8_t byte_t;

typedef enum {
    INT8 = 0,
    INT32 = 2,
    FP16 = 4,
    FP32 = 6,

    FP4 = 1,
    INT4 = 3,
    FP8 = 5,
    BF16 = 7,
    BYTE = 9
} fnblas_dtype_t;

typedef enum {
    ADD,
    SUBTRACT,
    MULTIPLY,
    DIVIDE
} fnblas_arithmetic_op_t;

typedef enum {
    FNBLAS_SUCCESS = 0,
    FNBLAS_ERR_MALLOC_FAIL,
    FNBLAS_ERR_MISMATCH,
    FNBLAS_ERR_UNKNOWN
} fnblas_error_t;

typedef struct {
    byte_t *_mem;       // memory pointer
    size_t _size;       // size of the static memory
    size_t n_buffers;   // total number of buffers that can be allocated in this static memory

    byte_t *_buffer_ptrs[FNBLAS_MAX_N_BUFFERS_PER_STATIC_MEMORY];
    size_t _buffer_sizes[FNBLAS_MAX_N_BUFFERS_PER_STATIC_MEMORY];
    uint8_t _buffer_used[FNBLAS_MAX_N_BUFFERS_PER_STATIC_MEMORY];
} fnblas_static_memory_t;

#if !FNBLAS_MEM_DYNAMIC_ENABLE
fnblas_error_t fnblas_create_static_memory(byte_t *mem, size_t size, size_t n_buffers);
#endif

// Data type and Softfloat Utilities
size_t      fnblas_dtype_size_of(fnblas_dtype_t dtype);
size_t      fnblas_dtype_pack_size_of(fnblas_dtype_t dtype);

uint32_t    fnblas_dtype_float_to_bits(float value);
float       fnblas_dtype_bits_to_float(uint32_t bits);

float       fnblas_dtype_softfloat_bf16_to_fp32(float value);
uint16_t    fnblas_dtype_fp32_to_fp16_bits(float value);
float       fnblas_dtype_fp16_bits_to_fp32(uint16_t value);
float       fnblas_dtype_softfloat_fp16_to_fp32(float value);
float       fnblas_dtype_bf16_bits_to_fp32(uint16_t value);

// Scalar
typedef struct {
    union {
        float f32;
        int32_t i32;
        int8_t i8;
        uint16_t u16;
    } _value;
    fnblas_dtype_t _dtype;
} fnblas_scalar_t;

// Vector
typedef struct {
    byte_t* _buffer;
    size_t _n_elements;
    fnblas_dtype_t _dtype;
    uint8_t _status;
} fnblas_vector_t;

#define FNBLAS_VECTOR_INITIALIZER { NULL, 0, FP32, 0 }

// Matrix
typedef struct {
    byte_t* _buffer;
    size_t _n_rows;
    size_t _n_cols;
    fnblas_dtype_t _dtype;
    uint8_t _status;
} fnblas_matrix_t;

#define FNBLAS_MATRIX_INITIALIZER { NULL, 0, 0, FP32, 0 }

#ifdef __cplusplus
}
#endif

#endif // FLEXI_BLAS_COMMON_H
