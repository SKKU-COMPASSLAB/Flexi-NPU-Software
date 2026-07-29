# Flexi Tensor Library

## Introduction

Flexi Tensor Library는 텐서 연산을 구현하기 위한 C 라이브러리이다. Flexi NPU 소프트웨어 스택에서 공통으로 사용할 수 있는 벡터, 행렬 및 텐서 자료구조와 연산을 제공한다.

이 라이브러리는 다음과 같은 환경에서 활용된다.

* `libflexi`와 같은 NPU 시뮬레이션 모델의 연산 구현
* RISC-V baremetal 테스트 코드 작성
* 호스트 환경에서의 연산 및 기능 검증

Flexi Tensor Library는 동적 메모리 할당 방식과 사용자가 제공한 memory pool을 이용하는 정적 메모리 관리 방식을 모두 제공한다. 따라서 일반적인 호스트 프로그램뿐 아니라 동적 메모리 사용이 제한되는 baremetal 환경에서도 사용할 수 있다.

라이브러리는 기능에 따라 `fnblas`와 `flexi-tensor`로 구성된다.

|Component|Header|Description|
|---|---|---|
|`fnblas`|`flexi_blas.h`|Scalar, vector 및 matrix 자료구조와 연산 API|
|`flexi-tensor`|`flexi_tensor.h`|Tensor 자료구조, tensor 연산 및 backend API|

## fnblas

`fnblas`는 벡터와 행렬 연산을 구현하기 위한 기본 API를 제공한다. Scalar, vector 및 matrix 객체의 생성과 초기화, 데이터 타입 변환, 원소 단위 연산, reduction 및 행렬 연산 등을 지원한다.

`flexi-tensor`와 NPU 시뮬레이션 모델은 공통 자료구조와 기본 연산을 구현하기 위해 `fnblas`를 사용한다.

구체적인 자료구조와 API 사용법은 [fnblas 문서](docs/fnblas.md)에서 설명한다.

## flexi-tensor

`flexi-tensor`는 다차원 tensor 자료구조와 tensor 연산 API를 제공한다. Linear, activation, convolution, pooling 및 softmax와 같은 연산을 제공하며, tensor의 일부 영역을 추가 복사 없이 참조하기 위한 view도 지원한다.

Tensor 연산은 backend를 통해 실행된다. 기본 backend는 `fnblas`를 사용하며, 사용자는 backend에 연산 kernel을 등록하여 NPU 또는 다른 연산 구현으로 전환할 수 있다. 이를 통해 동일한 tensor API를 유지하면서 실행 대상을 변경할 수 있다.

구체적인 tensor 연산과 backend API 사용법은 [flexi-tensor 문서](docs/flexi-tensor.md)에서 설명한다.

## Memory Management

메모리 관리 방식은 라이브러리를 빌드할 때 `FNBLAS_MEM_DYNAMIC_ENABLE` 설정으로 결정된다.

|Mode|Configuration|Description|
|---|---|---|
|동적 메모리|`FNBLAS_MEM_DYNAMIC_ENABLE=1`|라이브러리가 필요한 buffer를 동적으로 할당하고 해제한다.|
|정적 메모리|`FNBLAS_MEM_DYNAMIC_ENABLE=0`|사용자가 미리 등록한 memory pool에서 buffer를 할당하고 반환한다.|

정적 메모리 방식에서는 vector, matrix 또는 tensor를 생성하기 전에 `fnblas_create_static_memory()`로 정렬된 memory pool을 등록해야 한다. 각 방식은 별도의 정적 라이브러리로 제공되므로 실행 환경에 맞는 파일을 선택하여 링크한다.

## Requirements

호스트 라이브러리는 시스템 C compiler를 사용한다. RISC-V 라이브러리를 빌드하려면 RISC-V GNU toolchain이 필요하며, `RISCV` 환경 변수가 toolchain 설치 prefix를 가리켜야 한다.

```bash
export RISCV=/path/to/riscv/install
source env.sh
```

## Usages

`tensor_lib/Makefile`은 최상위 `Makefile`에 포함되므로 다음 명령은 레포지토리 루트에서 실행한다.

```bash
# 호스트와 RISC-V용 라이브러리를 모두 빌드
make tensor-libs

# 필요한 라이브러리만 개별 빌드
make tensor-lib-host
make tensor-lib-rv
make tensor-lib-host-static
make tensor-lib-rv-static

# tensor library 빌드 결과 삭제
make tensor-lib-clean
```

빌드 결과는 `tensor_lib/build`에 생성된다.

|Output|Target|Memory Management|
|---|---|---|
|`libflexi_tensor.a`|Host|동적 메모리|
|`libflexi_tensor_rv.a`|RISC-V|동적 메모리|
|`libflexi_tensor_static.a`|Host|정적 메모리|
|`libflexi_tensor_rv_static.a`|RISC-V|정적 메모리|

프로그램에서는 필요한 API에 따라 public header를 포함하고, 실행 환경과 메모리 관리 방식에 맞는 라이브러리를 링크한다.

```c
#include "flexi_blas.h"
#include "flexi_tensor.h"
```

호스트용 동적 메모리 라이브러리를 사용하는 프로그램은 다음과 같이 빌드할 수 있다.

```bash
gcc -I./tensor_lib example.c -L./tensor_lib/build -lflexi_tensor -lm -o example
```
