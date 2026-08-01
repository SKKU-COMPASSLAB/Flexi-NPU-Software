# Flexi-NPU Tensor Backend Test

`Flexi-NPU Tensor Library` 기반의 테스트를 제공한다. 라이브러리 API에 대한 설명은 [공식 문서](../../tensor_lib/README.md)를 참고한다.

## Unit Tests

|Test|Precision|Description|
|---|---|---|
|`t1_validate_matmul_kernel`|`bfloat16`|행렬곱 혹은 Linear 커널 검증 테스트|
|`t2_validate_conv2d_kernel`|`bfloat16`|Conv2d 커널 검증 테스트|
|`t3_validate_quant_kernel`|`bfloat16`|BF16-INT8 양자화 커널 검증 테스트|
|`t4_fp4_validate_matmul_kernel`|`float4`|행렬곱 혹은 Linear 커널 검증 테스트|
|`t5_fp4_validate_conv2d_kernel`|`float4`|Conv2d 커널 검증 테스트|
|`t6_fp4_validate_quant_kernel`|`float4`|FP32-FP4 양자화 커널 검증 테스트|

```bash
# make sure that $RISCV is properly setup and registered as $PATH
cd sw/flexi-tensor-tests
spike --isa=rv64gcv --extension=flexi --extension=flexi_hx ./bin/<binary path>
```

## End-to-End Tests

### Requirements

* Python >=3.11
* PyTorch

> `torch`가 설치된 파이썬 인터프리터를 `make`할 때 `PYTHON` 변수로 넘겨주거나, 혹은 환경을 활성화해서 시스템의 `python3`가 `torch`가 설치된 인터프리터를 가리키도록 한다.

### Simple MLP Test

```bash
cd sw/flexi-tensor-tests                    # move to flexi-tensor-tests directory
make PYTHON=/path/to/python e1_simple_mlp   # generate dumpfiles, header, linker script, and build binary
```