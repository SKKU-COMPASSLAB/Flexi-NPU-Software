# flexi-tensor

## Introduction

`flexi-tensor`는 최대 4차원의 tensor 자료구조와 tensor operator API를 제공하는 C 라이브러리이다. Operator 구현은 backend abstraction을 통해 `fnblas`, NPU 또는 사용자가 제공하는 kernel을 선택할 수 있다.

전체 public API는 `flexi_tensor.h`를 포함하여 사용할 수 있다.

```c
#include "flexi_tensor.h"
```

API는 `flexi_tensor_error_t`를 반환한다. 정상적으로 완료되면 `FLEXI_TENSOR_SUCCESS`를 반환하고, 메모리 할당 실패, shape 또는 dtype 불일치, 미지원 kernel 및 backend 오류를 구분하여 보고한다.

## Tensor Data Structure

`flexi_tensor_t`는 row-major tensor의 buffer, 차원 정보, dtype과 내부 상태를 보관한다.

```c
typedef struct {
    byte_t* _buffer;
    size_t _n_dims;
    size_t* _shape;
    size_t* _strides;
    fnblas_dtype_t _dtype;
    uint8_t _status;
} flexi_tensor_t;
```

|Field|Description|
|---|---|
|`_buffer`|Tensor 원소가 저장된 연속된 unpacked buffer|
|`_n_dims`|차원 수. 1부터 4까지 지원한다.|
|`_shape`|각 차원의 크기를 저장한 배열|
|`_strides`|Row-major offset 계산에 사용하는 stride 배열|
|`_dtype`|원소의 `fnblas_dtype_t`|
|`_status`|Buffer 소유 여부와 view 여부를 나타내는 내부 flag|

Tensor shape와 index는 `flexi_tuple_t`로 전달한다. `FLEXI_TUPLE(...)` macro를 사용하면 1차원부터 4차원까지의 tuple을 간단히 생성할 수 있다.

```c
flexi_tuple_t shape = FLEXI_TUPLE(1, 28, 28, 3);
flexi_tuple_t index = FLEXI_TUPLE(0, 10, 10, 2);
```

Tensor의 내부 데이터 저장 규칙은 `fnblas`와 동일하다. FP16, BF16 또는 FP4 tensor도 연산 buffer에서는 원소당 `float`를 사용하는 unpacked 형식이며, dtype에 맞게 quantize된 값을 저장한다. Packed 데이터는 전용 packing API를 사용하여 변환한다.

## Tensor Lifecycle

모든 tensor 객체는 사용 전에 `FLEXI_TENSOR_INITIALIZER`로 초기화해야 한다.

### Creation

`flexi_tensor_create()`는 shape와 dtype에 맞는 data buffer, shape 배열 및 stride 배열을 할당한다. 생성된 tensor는 row-major layout을 사용하며 data buffer는 0으로 초기화된다.

```c
flexi_tensor_t tensor = FLEXI_TENSOR_INITIALIZER;

if (flexi_tensor_create(&tensor, FLEXI_TUPLE(1, 28, 28, 3), FP32) != FLEXI_TENSOR_SUCCESS) {
    return 1;
}
```

기존 buffer나 tensor를 추가 복사 없이 참조하려면 view API를 사용한다.

|API|Description|
|---|---|
|`flexi_tensor_create_view()`|기존 unpacked buffer를 지정한 shape의 tensor로 참조한다.|
|`flexi_tensor_create_reshape_view()`|원소 수가 같은 tensor를 다른 shape로 참조한다.|
|`flexi_tensor_create_matrix_view()`|Tensor의 inner-most 2개 차원을 `fnblas_matrix_t`로 참조한다.|
|`flexi_tensor_create_vector_view()`|Tensor의 inner-most 차원을 `fnblas_vector_t`로 참조한다.|

Matrix 또는 vector view를 만들 때 `index`는 view로 남겨 둘 inner dimension을 제외한 outer dimension의 위치를 지정한다. 예를 들어 shape가 `(N, H, W, C)`인 tensor에서 `FLEXI_TUPLE(n, h)`를 전달해 `(W, C)` matrix view를 만들 수 있다.

View는 원본과 같은 data buffer를 공유한다. View를 destroy하면 view가 관리하는 metadata만 정리되고 공유 data buffer는 해제되지 않는다. 따라서 원본 buffer는 모든 view의 사용이 끝날 때까지 유효해야 한다.

### Initialization and Access

|API|Description|
|---|---|
|`flexi_tensor_initialize_from_unpacked_buffer()`|Unpacked buffer의 값을 tensor에 복사한다.|
|`flexi_tensor_initialize_from_packed_buffer()`|Packed buffer를 dtype에 맞게 unpack하여 tensor를 초기화한다.|
|`flexi_tensor_get_packed_buffer()`|Tensor의 값을 packed buffer로 변환한다.|
|`flexi_tensor_at()`|Tuple index의 원소를 읽는다.|
|`flexi_tensor_set()`|Tuple index의 원소를 설정한다.|
|`flexi_tensor_n_dims()`|Tensor의 차원 수를 반환한다.|
|`flexi_tensor_dim_size()`|지정한 차원의 크기를 반환한다.|
|`flexi_tensor_dtype()`|Tensor dtype을 반환한다.|
|`flexi_tensor_packed_buffer_size()`|Packed buffer에 필요한 byte 수를 반환한다.|
|`flexi_tensor_unpacked_buffer_size()`|Unpacked buffer에 필요한 byte 수를 반환한다.|

다음은 FP32 tensor를 생성하고 unpacked buffer로 초기화한 뒤 제거하는 예이다.

```c
float values[2][3] = {
    {1.0f, 2.0f, 3.0f},
    {4.0f, 5.0f, 6.0f}
};
flexi_tensor_t tensor = FLEXI_TENSOR_INITIALIZER;

if (flexi_tensor_create(&tensor, FLEXI_TUPLE(2, 3), FP32) != FLEXI_TENSOR_SUCCESS) {
    return 1;
}
if (flexi_tensor_initialize_from_unpacked_buffer(&tensor, (const byte_t*)values) != FLEXI_TENSOR_SUCCESS) {
    flexi_tensor_destroy(&tensor);
    return 1;
}

flexi_tensor_destroy(&tensor);
```

### Destruction

`flexi_tensor_destroy()`는 tensor가 소유한 data buffer와 shape 및 stride metadata를 정리하고 객체를 초기 상태로 되돌린다. 정상 생성된 tensor뿐 아니라 view도 사용이 끝나면 destroy해야 한다.

동적 메모리 방식과 정적 memory pool 방식의 선택 및 초기화 방법은 [fnblas 문서](fnblas.md#memory-management)를 참고한다. Tensor의 data buffer와 metadata도 동일한 메모리 관리 계층을 사용한다.

## Tensor Operators

Public tensor operator는 다음과 같다.

|API|Description|
|---|---|
|`flexi_tensor_op_linear()`|입력의 마지막 차원에 linear 연산을 적용한다.|
|`flexi_tensor_op_relu()`|모든 원소에 ReLU를 적용한다.|
|`flexi_tensor_op_conv2d()`|NHWC 입력과 `(FH, FW, K, C)` weight를 사용하여 2D convolution을 수행한다.|
|`flexi_tensor_op_maxpool2d()`|NHWC tensor에 2D max pooling을 수행한다.|
|`flexi_tensor_op_avgpool2d()`|NHWC tensor에 2D average pooling을 수행한다.|
|`flexi_tensor_op_softmax()`|가장 안쪽 차원을 기준으로 softmax를 수행한다.|

Operator의 `out`에 초기화되지 않은 tensor를 전달하면 필요한 output shape로 자동 생성된다. 이미 생성된 tensor나 view를 전달할 수도 있지만 연산이 요구하는 shape와 dtype이 일치해야 한다. 현재 operator는 입력과 출력이 같은 buffer를 공유하는 in-place 실행을 지원하지 않는다.

## Tensor Operator Backend

### Overview

`flexi_tensor_backend_t`는 tensor operator가 사용하는 저수준 scalar/vector/matrix kernel의 function pointer 집합이다. Tensor operator는 입력 tensor를 적절한 fnblas view로 나누거나 reshape한 뒤, 현재 backend에 등록된 kernel을 호출하여 연산을 수행한다.

기본 backend인 `flexi_tensor_fnblas_backend`는 모든 kernel을 `fnblas` 연산으로 구현한다. NPU backend는 필요한 kernel만 가속기 구현으로 교체할 수 있다. 예를 들어 NPU matrix multiplication을 사용하려면 `op_mm_matmul`에 NPU kernel을 등록하고 나머지 function pointer는 `NULL`로 둘 수 있다.

```text
Tensor operator
      |
      v
Tensor를 vector/matrix view로 변환
      |
      v
현재 backend의 kernel function pointer 호출
      |
      +-- fnblas kernel
      +-- NPU kernel
      +-- custom kernel
```

### Backend Kernels

Backend의 kernel class와 function signature는 `fnblas` 연산 분류를 따른다.

|Kernel Group|Examples|Used By|
|---|---|---|
|Scalar-Vector|`op_sv_max`|ReLU, pooling|
|Vector Elementwise|`op_ve_add`, `op_ve_div`|Bias addition, pooling, softmax|
|Vector Reduction|`op_vr_sum`, `op_vr_max`|Pooling, softmax|
|Vector Individual|`op_vi_exp`|Softmax|
|Matrix Elementwise|`op_me_add`|Convolution partial sum|
|Matrix-Matrix|`op_mm_matmul`|Linear, convolution|

Backend kernel은 `fnblas_vector_t`, `fnblas_matrix_t` 및 `fnblas_scalar_t`를 인자로 받고 `flexi_tensor_error_t`를 반환한다. Custom kernel은 전달받은 view의 shape, dtype 및 output buffer 규칙을 지켜야 한다.

### Selecting a Backend

Backend는 `flexi_tensor_use_backend()`로 전역 선택한다.

```c
flexi_tensor_error_t flexi_tensor_use_backend(const flexi_tensor_backend_t* backend);
```

호출 시 현재 backend는 먼저 전체 fnblas backend로 초기화된 후, 전달한 backend에서 `NULL`이 아닌 function pointer만 교체된다. 따라서 custom backend가 구현하지 않은 kernel은 자동으로 기본 fnblas kernel을 사용한다.

다음은 matrix multiplication만 교체하는 backend의 기본 형태이다.

```c
static flexi_tensor_error_t custom_matmul(fnblas_matrix_t* result, const fnblas_matrix_t* lhs, const fnblas_matrix_t* rhs_t) {
    /* NPU 또는 custom matrix multiplication 구현 */
    return FLEXI_TENSOR_SUCCESS;
}

static const flexi_tensor_backend_t custom_backend = {
    .op_mm_matmul = custom_matmul
};

int main(void) {
    if (flexi_tensor_use_backend(&custom_backend) != FLEXI_TENSOR_SUCCESS) {
        return 1;
    }

    /* 이후 tensor operator는 custom matmul과 나머지 fnblas kernel을 사용한다. */
    return 0;
}
```

Backend 선택은 개별 tensor가 아니라 라이브러리의 현재 backend 전체에 적용된다. 기본 구현으로 되돌리려면 다음과 같이 fnblas backend를 다시 선택한다.

```c
flexi_tensor_use_backend(&flexi_tensor_fnblas_backend);
```

Backend kernel에서 발생한 실패는 대응하는 `flexi_tensor_error_t`로 tensor operator에 전달된다.
