# Flexi-NPU Software

## Introduction

이 레포지토리는 Flexi-NPU를 위한 소프트웨어 스택 전반을 포함하는 라이브러리이다. 전체적인 구성은 다음과 같다. 

|Directory|Description|
|---|---|
|`sim`|RoCC accelerator simulation model|
|`tensor_lib`|Tensor BLAS library for simulation model and tests|
|`sw`|Flexi-NPU ISA specification and test softwares|
|`test`|Test code for validation (not used for Flexi-NPU hardware)|

> 기존에 제공하던 [TDS Simulator Spike Extension](https://github.com/SKKU-COMPASSLAB/TDS-Simulator-Spike-Extension) 레포지토리의 경우 파이썬 기반의 NPU 시뮬레이션 모델을 기반으로 하고 있었으나, 해당 레포지토리는 순수 C 기반의 NPU 시뮬레이션 모델로 변경되었다. 따라서 파이썬 관련 의존성은 존재하지 않는다.

> 기존에 제공하던 [TDS Simulator Spike Extension](https://github.com/SKKU-COMPASSLAB/TDS-Simulator-Spike-Extension) 레포지토리의 경우 IREE 런타임 시뮬레이션을 위해 Buildroot를 활용한 리눅스 환경 시뮬레이션을 제공하였으나, 본 레포지토리에서는 제공하지 않는다. 모든 소프트웨어는 baremetal로 빌드되며, 별도 표준 GNU 라이브러리를 링크하지 않는다. 추후 필요한 경우 제공할 예정이다.

## Requirements

시뮬레이터는 아래 환경에서 테스트되었다.

* Ubuntu 22.04
* Intel(R) Xeon(R) Platinum 8362 CPU @ 2.80GHz

RISC-V GNU toolchain과 Spike ISS가 정상설치 되어있어야 하며, `RISCV` 환경변수가 RISC-V 소프트웨어 설치 경로로 초기화되어있어야 한다. 또한, `RISCV`가 `PATH`에 등록되어있어야 한다. 기존에 설치된 툴체인이 있다면 `RISCV` 환경변수만 설정하면 된다. 자세한 내용은 [RISC-V 의존성 문서](docs/riscv-dependencies.md)를 참고한다.

* [RISC-V GNU Toolchain](https://github.com/riscv-collab/riscv-gnu-toolchain)
* [RISC-V Spike ISS](https://github.com/riscv-software-src/riscv-isa-sim)

> RISC-V GNU 툴체인 및 Spike ISS는 설치 과정에서 문제가 있어 본 레포지토리에서 제공하지 않는다. 별도로 의존성 소프트웨어를 설치하고 `RISCV` 환경변수를 설정하여 올바르게 시뮬레이터가 빌드 될 수 있도록 해야 한다.

> 본 시뮬레이터는 RV64GCV ISA로 컴파일된 GNU 툴체인을 필요로 한다.

> RVV 관련 이슈를 최대한 방지하기 위해서 `GCC 16.1` 혹은 최신 버전의 툴체인을 설치하는 것을 권장한다. `chipyard`에 내장된 RISC-V GNU 툴체인의 경우 RVV intrinsic 빌드 과정에서 `vsetvli` lowering할 때 문제가 발생하는 것이 확인되었다.  

> 해당 레포지토리를 통해 생성된 RISC-V ELF는 `chipyard`를 통해 생성된 RTL 코드를 `verilator` 및 `firesim`을 활용하여 시뮬레이션할 때 활용될 수 있다. 하지만, 이 경우에도 가급적이면 `chipyard`에 내장된 RISC-V GNU 툴체인을 사용하지 않는 것을 권장한다. 소규모의 테스트를 통해 `chipyard v1.13.0` 기준으로 최신버전의 툴체인을 사용해도 실행에 문제가 없음을 확인하였다.

필요 요구 사항을 설치 후 `env.sh`를 실행하여 `RISCV`를 PATH에 등록하고, 필요한 툴체인이 전부 설치되었는지를 확인한다.

```bash
source env.sh
```

## Usages

### `DIM=32` Simulation Model

BF16 정밀도를 지원하는 새로운 Flexi-NPU 시뮬레이션 모델의 경우 `DIM`의 크기가 32이다. 해당 시뮬레이션 모델은 별도의 `make` 타겟으로 설정되어있으며, 아래와 같이 빌드 및 설치가 가능하다.

```bash
# build libraries
make

# install simulation model
make sim-install-dim32

# run Flexi-NPU ISA tests
cd sw/flexi-isa-tests
make -j
cd ../..
spike --isa=rv64gcv_zvl512b_zicsr_zifencei_zicntr_zihpm --extension=flexi --extension=flexi_hx ./sw/flexi-isa-tests/build/flexi_gemm_bf16_test-baremetal

# run tests with tensor library
make -C sw/flexi-tensor-tests
spike --isa=rv64gcv_zvl512b_zicsr_zifencei_zicntr_zihpm --extension=flexi --extension=flexi_hx ./sw/flexi-tensor-tests/bin/t1_validate_matmul_kernel 
```

### `DIM=128` Simulation Model

FP4 정밀도를 지원하는 새로운 Flexi-NPU 시뮬레이션 모델의 경우 `DIM`의 크기가 128이다. 해당 시뮬레이션 모델은 별도의 `make` 타겟으로 설정되어있으며, 아래와 같이 빌드 및 설치가 가능하다. 

```bash
# build libraries
make

# install simulation model
make sim-install-dim128

# run Flexi-NPU ISA tests
cd sw/flexi-isa-tests
make -j
cd ../..
spike --isa=rv64gcv_zvl512b_zicsr_zifencei_zicntr_zihpm --extension=flexi --extension=flexi_hx ./sw/flexi-isa-tests/build/flexi_gemm_fp4_test-baremetal

# run tests with tensor library
make -C sw/flexi-tensor-tests
spike --isa=rv64gcv_zvl512b_zicsr_zifencei_zicntr_zihpm --extension=flexi --extension=flexi_hx ./sw/flexi-tensor-tests/bin/t4_fp4_validate_matmul_kernel 
```

## Documentation

* [Flexi-NPU Simulation Model](sim/README.md)
* [Flexi-NPU Software](sw/README.md)
* [Tensor Library](tensor_lib/README.md)