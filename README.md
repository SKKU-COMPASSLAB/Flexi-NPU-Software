# Flexi-NPU Software

## Introduction

이 레포지토리는 Flexi-NPU를 위한 소프트웨어 스택 전반을 포함하는 라이브러리이다. 전체적인 구성은 다음과 같다. 

|Directory|Description|
|---|---|
|`sim`|RoCC accelerator simulation model|
|`tensor_lib`|Tensor BLAS library for simulation model and runtime software|
|`sw`|Flexi-NPU ISA specification and runtime software|
|`test`|Test code for validation|

## Prerequisites

RISC-V GNU toolchain과 Spike ISS가 정상설치 되어있어야 하며, `RISCV` 환경변수가 RISC-V 소프트웨어 설치 경로로 초기화되어있어야 한다. 또한, `RISCV`가 `PATH`에 등록되어있어야 한다.

## Installation

```bash
# build simulation model and tensor library
make

# install simulation model as an extension to the Spike ISS
make install

# run unit-tests with the simulation model
make test-model
```

## Documentation

* [Tensor Library Documentation](tensor_lib/README.md)
* [Flexi-NPU Simulation Model](sim/README.md)
* [Flexi-NPU Software](sw/README.md)
* [Flexi-NPU Test](test/README.md)