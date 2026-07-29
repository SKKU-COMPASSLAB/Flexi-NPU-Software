# fnblas

## Introduction

`fnblas`는 scalar, vector 및 matrix 자료구조와 기본 연산 kernel을 제공하는 C API이다. 전체 public API는 `flexi_blas.h`를 포함하여 사용할 수 있다.

```c
#include "flexi_blas.h"
```

대부분의 API는 `fnblas_error_t`를 반환한다. 정상적으로 완료되면 `FNBLAS_SUCCESS`를 반환하며, 메모리 할당 실패나 입력 shape 및 dtype 불일치는 각각 대응하는 error 값으로 보고한다.

## Data Types

`fnblas_dtype_t`는 객체의 원소 데이터 타입을 나타낸다. 현재 정수 타입 `INT4`, `INT8`, `INT32`와 부동소수점 타입 `FP4`, `FP8`, `FP16`, `BF16`, `FP32`를 정의한다.

연산에 사용하는 vector와 matrix의 내부 buffer는 unpacked 형식이다. 낮은 정밀도의 부동소수점 타입도 내부 연산 buffer에서는 원소당 `float`로 저장하되, 해당 dtype의 정밀도로 값을 표현한다. Packed 형식은 외부 데이터 또는 가속기 데이터와 교환할 때 전용 packing API를 통해 사용한다.

|Representation|Description|
|---|---|
|Unpacked buffer|객체가 연산에 직접 사용하는 내부 표현|
|Packed buffer|dtype의 실제 bit width에 맞춘 입출력 및 저장 표현|

`fnblas_dtype_size_of()`와 `fnblas_dtype_pack_size_of()`를 이용하면 dtype의 저장 크기와 한 byte에 packing되는 원소 수를 확인할 수 있다.

## Data Structures

### Scalar

`fnblas_scalar_t`는 하나의 값과 dtype을 저장한다. 값은 내부 union에 저장되며 별도의 heap buffer를 할당하지 않으므로 destroy API가 필요하지 않다.

주요 생성 및 변환 API는 다음과 같다.

|API|Description|
|---|---|
|`fnblas_scalar_create_from_dtype()`|지정한 dtype의 0으로 scalar를 생성한다.|
|`fnblas_scalar_create_float()`|부동소수점 값으로 scalar를 생성하고 지정한 dtype 정밀도를 적용한다.|
|`fnblas_scalar_create_int()`|정수 값으로 scalar를 생성한다.|
|`fnblas_scalar_cast_to()`|새로운 dtype으로 변환한 scalar를 반환한다.|
|`fnblas_scalar_as_unpacked_float()`|scalar 값을 unpacked `float`로 읽는다.|
|`fnblas_scalar_as_unpacked_int32()`|scalar 값을 unpacked `int32_t`로 읽는다.|

### Vector

`fnblas_vector_t`는 연속된 1차원 buffer, 원소 수, dtype 및 내부 상태 flag를 보관한다.

```c
typedef struct {
    byte_t* _buffer;
    size_t _n_elements;
    fnblas_dtype_t _dtype;
    uint8_t _status;
} fnblas_vector_t;
```

모든 vector 객체는 사용 전에 `FNBLAS_VECTOR_INITIALIZER`로 초기화해야 한다.

### Matrix

`fnblas_matrix_t`는 row-major 형식의 2차원 buffer, row와 column 수, dtype 및 내부 상태 flag를 보관한다.

```c
typedef struct {
    byte_t* _buffer;
    size_t _n_rows;
    size_t _n_cols;
    fnblas_dtype_t _dtype;
    uint8_t _status;
} fnblas_matrix_t;
```

모든 matrix 객체는 사용 전에 `FNBLAS_MATRIX_INITIALIZER`로 초기화해야 한다.

## Object Lifecycle

Vector와 matrix는 동일한 lifecycle을 사용한다.

1. initializer macro로 객체를 초기화한다.
2. `create` 또는 `create_view` 계열 API로 shape와 dtype을 설정한다.
3. packed 또는 unpacked buffer로 값을 초기화한다.
4. 연산과 원소 접근 API를 사용한다.
5. `destroy` API로 객체를 정리한다.

### Creation

`fnblas_vector_create()`와 `fnblas_matrix_create()`는 객체의 unpacked buffer를 할당하고 0으로 초기화한다. 이미 생성되었거나 view로 사용 중인 객체를 다시 생성할 수 없으므로 먼저 destroy해야 한다.

View API는 새로운 데이터 buffer를 할당하지 않고 기존 buffer를 참조한다.

|API|Description|
|---|---|
|`fnblas_vector_create_view()`|기존 buffer를 vector로 참조한다.|
|`fnblas_matrix_create_view()`|기존 buffer를 row-major matrix로 참조한다.|
|`fnblas_matrix_create_reshape_view()`|원소 수가 같은 matrix를 다른 shape로 참조한다.|
|`fnblas_matrix_create_row_vector_view()`|matrix의 한 row를 vector로 참조한다.|

View와 원본은 같은 데이터를 공유한다. 원본 buffer의 lifetime이 view보다 길어야 하며, view를 destroy해도 공유 buffer는 해제되지 않는다.

### Initialization and Access

Vector와 matrix는 각각 다음과 같은 공통 형태의 API를 제공한다.

|Operation|Vector API|Matrix API|
|---|---|---|
|Packed buffer로 초기화|`fnblas_vector_initialize_from_packed_buffer()`|`fnblas_matrix_initialize_from_packed_buffer()`|
|Unpacked buffer로 초기화|`fnblas_vector_initialize_from_unpacked_buffer()`|`fnblas_matrix_initialize_from_unpacked_buffer()`|
|Packed buffer 추출|`fnblas_vector_get_packed_buffer()`|`fnblas_matrix_get_packed_buffer()`|
|원소 읽기|`fnblas_vector_at()`|`fnblas_matrix_at()`|
|원소 쓰기|`fnblas_vector_set()`|`fnblas_matrix_set()`|
|객체 제거|`fnblas_vector_destroy()`|`fnblas_matrix_destroy()`|

`initialize_from_unpacked_buffer()`에 전달하는 buffer는 객체 dtype의 unpacked 원소 표현을 사용해야 한다. Packed buffer를 연산 buffer처럼 직접 연결하지 않는다.

다음은 FP32 vector를 생성하고 초기화하는 간단한 예이다.

```c
float values[4] = {1.0f, 2.0f, 3.0f, 4.0f};
fnblas_vector_t vector = FNBLAS_VECTOR_INITIALIZER;

if (fnblas_vector_create(&vector, 4, FP32) != FNBLAS_SUCCESS) {
    return 1;
}
if (fnblas_vector_initialize_from_unpacked_buffer(&vector, (const byte_t*)values) != FNBLAS_SUCCESS) {
    fnblas_vector_destroy(&vector);
    return 1;
}

fnblas_vector_destroy(&vector);
```

## Kernel Classes

연산 API의 이름은 `fnblas_op_<class>_<operation>` 형식을 사용한다. Class는 입력 및 출력 객체의 조합이나 연산 방식을 나타낸다.

|Class|Category|Representative APIs|
|---|---|---|
|`SS`|Scalar-Scalar arithmetic|`fnblas_op_ss_add()`, `fnblas_op_ss_mul()`|
|`SV`|Vector와 scalar 사이의 elementwise 연산|`fnblas_op_sv_add()`, `fnblas_op_sv_max()`|
|`VE`|두 vector 사이의 elementwise 연산|`fnblas_op_ve_add()`, `fnblas_op_ve_mul()`|
|`VV`|두 vector를 하나의 값으로 결합하는 연산|`fnblas_op_vv_dot()`|
|`VR`|Vector reduction|`fnblas_op_vr_sum()`, `fnblas_op_vr_mean()`, `fnblas_op_vr_max()`|
|`VI`|하나의 vector에 적용하는 원소별 연산|`fnblas_op_vi_exp()`|
|`MI`|하나의 matrix에 적용하는 연산|`fnblas_op_mi_transpose()`|
|`ME`|두 matrix 사이의 elementwise 연산|`fnblas_op_me_add()`, `fnblas_op_me_mul()`|
|`MM`|Matrix multiplication|`fnblas_op_mm_matmul()`|
|`MMV`|Matrix 연산 결과에 vector를 결합하는 연산|`fnblas_op_mmv_axpy()`|
|`MMM`|Matrix 연산 결과에 matrix를 결합하는 연산|`fnblas_op_mmm_axpy()`|

`fnblas_op_mm_matmul()`의 오른쪽 입력 `rhs_t`는 오른쪽 행렬을 transpose한 형태로 전달한다. `MMV`의 AXPY 연산은 matrix 곱셈 결과의 각 row에 vector를 더하며, `MMM`의 AXPY 연산은 matrix 곱셈 결과에 matrix를 더한다.

대부분의 vector 및 matrix kernel은 비어 있는 result 객체를 전달하면 필요한 shape와 dtype으로 result를 생성한다. 이미 생성된 result를 전달하는 경우에는 입력과 호환되는 shape와 dtype이어야 한다. 입력과 result의 aliasing은 일반적으로 지원하지 않으므로 별도 객체를 사용한다.

## Memory Management

메모리 관리 방식은 컴파일 시 `FNBLAS_MEM_DYNAMIC_ENABLE`로 결정된다. 애플리케이션 코드에서 두 방식을 런타임에 전환하는 구조가 아니며, 필요한 방식으로 빌드된 라이브러리를 링크해야 한다.

### Dynamic Memory

`FNBLAS_MEM_DYNAMIC_ENABLE=1`에서는 객체 생성 시 `aligned_alloc()`으로 64-byte 정렬된 buffer를 할당하고, destroy 시 `free()`로 해제한다.

|Library|Target|
|---|---|
|`libflexi_tensor.a`|Host|
|`libflexi_tensor_rv.a`|RISC-V|

별도의 메모리 초기화 API 없이 객체를 생성할 수 있다.

### Static Memory

`FNBLAS_MEM_DYNAMIC_ENABLE=0`에서는 사용자가 등록한 memory pool 안에서 객체의 buffer를 할당한다. 객체를 생성하기 전에 다음 API를 한 번 호출해야 한다.

```c
fnblas_error_t fnblas_create_static_memory(byte_t* mem, size_t size, size_t n_buffers);
```

`mem`과 `size`는 pool의 시작 주소와 전체 크기이며, `n_buffers`는 동시에 추적할 수 있는 allocation 수이다. 값은 `FNBLAS_MAX_N_BUFFERS_PER_STATIC_MEMORY`를 초과할 수 없다. Pool 내부 allocation은 `FNBLAS_MEM_ALIGNMENT`에 맞춰 정렬되며, destroy된 객체가 사용하던 영역은 이후 allocation에서 재사용된다.

```c
#define MEMORY_SIZE (2 * 1024 * 1024)
#define MAX_BUFFERS 128

static byte_t memory_pool[MEMORY_SIZE] __attribute__((aligned(FNBLAS_MEM_ALIGNMENT)));

int main(void) {
    fnblas_vector_t vector = FNBLAS_VECTOR_INITIALIZER;

    if (fnblas_create_static_memory(memory_pool, sizeof(memory_pool), MAX_BUFFERS) != FNBLAS_SUCCESS) {
        return 1;
    }
    if (fnblas_vector_create(&vector, 256, BF16) != FNBLAS_SUCCESS) {
        return 1;
    }

    fnblas_vector_destroy(&vector);
    return 0;
}
```

큰 memory pool을 baremetal 프로그램의 지역 변수로 선언하면 제한된 stack을 손상시킬 수 있으므로 전역 또는 `static` 영역에 배치한다. Tensor 객체는 data buffer 외에도 shape와 stride를 위한 allocation을 사용하므로 `n_buffers`를 정할 때 이 allocation들도 고려해야 한다.

|Library|Target|
|---|---|
|`libflexi_tensor_static.a`|Host|
|`libflexi_tensor_rv_static.a`|RISC-V|
