# Flexi-NPU Simulation Model

## Introduction

`sim` 디렉터리는 Flexi-NPU 가속기 ISA를 Spike ISS에서 실행하기 위한 시뮬레이션 모델을 제공한다. 시뮬레이션 모델은 Spike의 RoCC custom instruction extension으로 구현되며, 호스트에서 빌드되는 `flexi_tensor` 라이브러리를 사용하여 가속기의 동작을 모사한다.

|Library|Description|
|---|---|
|`libflexi`|DMA와 Matrix Multiplication Unit(MXU)을 포함하는 Flexi NPU 시뮬레이션 모델|
|`libflexi_hx`|Fast Walsh-Hadamard Transform(FWHT)을 수행하는 Flexi HX 시뮬레이션 모델|

## libflexi

`libflexi`는 Flexi NPU의 DMA와 MXU 관련 명령을 동작 수준에서 시뮬레이션한다. 가속기 내부에는 512 KiB 크기의 on-chip memory가 있으며, DMA 명령으로 시스템 메모리와 on-chip memory 사이의 데이터를 이동한 뒤 MXU 명령으로 행렬 곱셈을 수행한다.

Flexi NPU ISA의 주요 동작은 다음과 같다.

|Operation|Description|
|---|---|
|DMA Load|시스템 메모리의 행렬 데이터를 on-chip memory로 읽어 들인다. Stride, transpose 및 zero padding을 지원한다.|
|DMA Store|on-chip memory의 연산 결과를 시스템 메모리에 저장한다.|
|GEMM Preload|가중치 행렬 또는 partial sum 행렬을 MXU에 미리 적재한다.|
|GEMM Execute|입력 행렬과 미리 적재된 가중치 행렬의 곱셈을 수행하고, 필요한 경우 partial sum을 누산한다.|
|GEMM Flush|MXU에 보관된 최종 결과를 on-chip memory로 내보낸다.|

이 모델은 명령에 인코딩된 주소, 행렬 크기, 데이터 타입 등의 필드를 해석하여 DMA 및 GEMM 동작을 실행한다. 실제 하드웨어의 cycle, pipeline 또는 메모리 대역폭을 정밀하게 모델링하지 않으며, 소프트웨어와 ISA의 기능 검증을 위한 동작 수준 모델이다.

### MXU dimension

MXU의 타일 크기는 빌드 시 `FLEXI_MXU_DIM`으로 결정된다. 현재 지원하는 값은 32와 128이며, 각 설정에서 MXU의 M, N, K 타일 크기가 모두 해당 값으로 설정된다.

빌드 시 두 설정의 라이브러리가 모두 생성된다.

|Output|Description|
|---|---|
|`sim/build/libflexi_dim32.so`|`FLEXI_MXU_DIM=32`로 빌드한 모델|
|`sim/build/libflexi_dim128.so`|`FLEXI_MXU_DIM=128`로 빌드한 모델|
|`sim/build/libflexi.so`|현재 선택된 DIM 모델을 가리키는 심볼릭 링크|

`FLEXI_MXU_DIM`을 지정하지 않으면 128이 기본값으로 사용된다. 32 또는 128 이외의 값은 지원하지 않는다.

## libflexi_hx

`libflexi_hx`는 BF16 벡터에 대한 정규화 Fast Walsh-Hadamard Transform을 수행하는 Flexi HX 가속기의 동작 수준 시뮬레이션 모델이다.

Flexi HX ISA의 주요 동작은 다음과 같다.

|Operation|Description|
|---|---|
|HX FWHT|시스템 메모리에서 하나 이상의 BF16 벡터를 읽고 FWHT를 수행한 뒤, 벡터 길이에 따른 정규화 계수를 적용하여 결과 주소에 저장한다.|
|HX Sync|가속기 연산의 동기화를 나타낸다. 명령이 순차적으로 실행되는 ISS에서는 별도의 동작을 수행하지 않는다.|

HX FWHT 명령은 입력 주소와 출력 주소, 벡터 길이를 나타내는 `logN`, 처리할 벡터 수를 전달받는다. 각 벡터의 원소 수는 `2^logN`이며 입력과 출력은 packed BF16 형식이다. 이 모델 역시 하드웨어의 정확한 실행 시간을 표현하지 않고 ISA의 기능적 결과를 검증한다.

## Requirements

빌드와 실행을 위해 RISC-V GNU toolchain과 Spike ISS가 필요하다. `RISCV` 환경 변수는 두 소프트웨어가 설치된 prefix를 가리켜야 한다.

```bash
export RISCV=/path/to/riscv/install
source env.sh
```

## Usages

`sim/Makefile`은 최상위 `Makefile`에 포함되므로 모든 명령은 레포지토리 루트에서 실행한다.

```bash
# 두 DIM의 libflexi와 libflexi_hx를 빌드
make sim-libs

# libflexi.so가 가리킬 DIM 선택
make sim-select-dim32
make sim-select-dim128

# 선택한 모델과 libflexi_hx를 ${RISCV}/lib에 설치
make sim-install-dim32
make sim-install-dim128

# 시뮬레이션 모델 빌드 결과 삭제
make sim-clean
```

`sim-install-dim32`와 `sim-install-dim128`은 각각 선택한 DIM의 모델을 `${RISCV}/lib/libflexi.so`로 설치하고, HX 모델을 `${RISCV}/lib/libflexi_hx.so`로 설치한다.

설치한 extension은 Spike 실행 시 다음과 같이 활성화할 수 있다.

```bash
${RISCV}/bin/spike --isa=rv64gcv --extension=flexi --extension=flexi_hx <baremetal-binary>
```
