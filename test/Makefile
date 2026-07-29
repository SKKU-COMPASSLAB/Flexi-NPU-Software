TEST_DIR := test
TEST_BIN_DIR := $(TEST_DIR)/bin
T1_BLAS_FP32_TEST := $(TEST_BIN_DIR)/t1_blas_fp32_test
T2_BLAS_BF16_TEST := $(TEST_BIN_DIR)/t2_blas_bf16_test
T3_BLAS_FP4_TEST := $(TEST_BIN_DIR)/t3_blas_fp4_test
T4_BLAS_MATRIX_TEST := $(TEST_BIN_DIR)/t4_blas_matrix_test
T5_TENSOR_VIEW_TEST := $(TEST_BIN_DIR)/t5_tensor_view
T6_DTYPE_CONVERSION_TEST := $(TEST_BIN_DIR)/t6_dtype_conversion_test
T7_QUANTIZATION_TEST := $(TEST_BIN_DIR)/t7_quantization_test
T8_QUANTIZATION_ERROR := $(TEST_BIN_DIR)/t8_quantization_error
MODEL_TEST_NAMES := \
	b0_dram_loopback \
	b1_dram_tiled_loopback \
	b2_dma_transpose \
	b3_dma_zero_padding \
	b4_gemm_128 \
	b5_gemm_tiled_256 \
	b6_hx_fwht
MODEL_TESTS := $(addprefix $(TEST_BIN_DIR)/,$(addsuffix -baremetal,$(MODEL_TEST_NAMES)))
B0_DRAM_LOOPBACK := $(TEST_BIN_DIR)/b0_dram_loopback-baremetal
BLAS_TESTS := \
	$(T1_BLAS_FP32_TEST) \
	$(T2_BLAS_BF16_TEST) \
	$(T3_BLAS_FP4_TEST) \
	$(T4_BLAS_MATRIX_TEST) \
	$(T5_TENSOR_VIEW_TEST) \
	$(T6_DTYPE_CONVERSION_TEST) \
	$(T7_QUANTIZATION_TEST) \
	$(T8_QUANTIZATION_ERROR)

RUNTIME_DIR := sw/flexi-isa-tests
BENCH_COMMON := $(RUNTIME_DIR)/riscv-tests/benchmarks/common
BAREMETAL_INCLUDES := \
	-I$(CURDIR)/.backup \
	-I$(CURDIR)/tensor_lib \
	-I$(RUNTIME_DIR)/riscv-tests \
	-I$(RUNTIME_DIR)/riscv-tests/env \
	-I$(RUNTIME_DIR)
ID_STRING ?= b0_dram_loopback
TEST_HOST_CFLAGS := -std=c++20 -O3 -Wall -Wextra -Werror $(FLEXI_TENSOR_CFLAGS)
TEST_HOST_LDFLAGS := $(FLEXI_TENSOR_LDFLAGS)
TEST_BAREMETAL_CFLAGS := \
	-DPREALLOCATE=1 \
	-DMULTITHREAD=1 \
	-mcmodel=medany \
	-std=gnu99 -O2 -fno-common -fno-builtin-printf \
	-fno-tree-loop-distribute-patterns \
	-march=rv64gc -Wa,-march=rv64gc \
	$(BAREMETAL_INCLUDES) \
	-DID_STRING=$(ID_STRING) \
	-DPRINT_TILE=0 \
	-ffreestanding \
	-nostdlib \
	-nostartfiles \
	-static \
	-DBAREMETAL=1

.PHONY: \
	t1_blas_fp32_test t2_blas_bf16_test t3_blas_fp4_test \
	t4_blas_matrix_test t5_tensor_view t6_dtype_conversion_test \
	t7_quantization_test t8_quantization_error \
	test-blas-fp32 test-blas-bf16 \
	test-blas-fp4 test-blas-matrix test-tensor-view test-blas \
	$(MODEL_TEST_NAMES) model-tests test-dram-loopback test-model test-clean

$(TEST_BIN_DIR):
	mkdir -p $@

$(T1_BLAS_FP32_TEST): $(TEST_DIR)/t1_blas_fp32_test.cc $(FLEXI_TENSOR_HOST_LIB) | $(TEST_BIN_DIR)
	$(CXX) $(TEST_HOST_CFLAGS) $< $(TEST_HOST_LDFLAGS) -o $@

$(T2_BLAS_BF16_TEST): $(TEST_DIR)/t2_blas_bf16_test.cc $(FLEXI_TENSOR_HOST_LIB) | $(TEST_BIN_DIR)
	$(CXX) $(TEST_HOST_CFLAGS) $< $(TEST_HOST_LDFLAGS) -o $@

$(T3_BLAS_FP4_TEST): $(TEST_DIR)/t3_blas_fp4_test.cc $(FLEXI_TENSOR_HOST_LIB) | $(TEST_BIN_DIR)
	$(CXX) $(TEST_HOST_CFLAGS) $< $(TEST_HOST_LDFLAGS) -o $@

$(T4_BLAS_MATRIX_TEST): $(TEST_DIR)/t4_blas_matrix_test.cc $(FLEXI_TENSOR_HOST_LIB) | $(TEST_BIN_DIR)
	$(CXX) $(TEST_HOST_CFLAGS) $< $(TEST_HOST_LDFLAGS) -o $@

$(T5_TENSOR_VIEW_TEST): $(TEST_DIR)/t5_tensor_view.cc $(FLEXI_TENSOR_HOST_LIB) | $(TEST_BIN_DIR)
	$(CXX) $(TEST_HOST_CFLAGS) $< $(TEST_HOST_LDFLAGS) -o $@

$(T6_DTYPE_CONVERSION_TEST): $(TEST_DIR)/t6_dtype_conversion_test.cc $(FLEXI_TENSOR_HOST_LIB) | $(TEST_BIN_DIR)
	$(CXX) $(TEST_HOST_CFLAGS) $< $(TEST_HOST_LDFLAGS) -o $@

$(T7_QUANTIZATION_TEST): $(TEST_DIR)/t7_quantization_test.cc $(FLEXI_TENSOR_HOST_LIB) | $(TEST_BIN_DIR)
	$(CXX) $(TEST_HOST_CFLAGS) $< $(TEST_HOST_LDFLAGS) -o $@

$(T8_QUANTIZATION_ERROR): $(TEST_DIR)/t8_quantization_error.cc $(FLEXI_TENSOR_HOST_LIB) | $(TEST_BIN_DIR)
	$(CXX) $(TEST_HOST_CFLAGS) $< $(TEST_HOST_LDFLAGS) -o $@

$(TEST_BIN_DIR)/%-baremetal: $(TEST_DIR)/%.c .backup/flexi/flexi.h | $(TEST_BIN_DIR)
	$(CC_RV) $(TEST_BAREMETAL_CFLAGS) -T $(BENCH_COMMON)/test.ld $< $(wildcard $(BENCH_COMMON)/*.c) $(wildcard $(BENCH_COMMON)/*.S) -lgcc -o $@

t1_blas_fp32_test: $(T1_BLAS_FP32_TEST)
t2_blas_bf16_test: $(T2_BLAS_BF16_TEST)
t3_blas_fp4_test: $(T3_BLAS_FP4_TEST)
t4_blas_matrix_test: $(T4_BLAS_MATRIX_TEST)
t5_tensor_view: $(T5_TENSOR_VIEW_TEST)
t6_dtype_conversion_test: $(T6_DTYPE_CONVERSION_TEST)
t7_quantization_test: $(T7_QUANTIZATION_TEST)
t8_quantization_error: $(T8_QUANTIZATION_ERROR)
$(MODEL_TEST_NAMES): %: $(TEST_BIN_DIR)/%-baremetal
model-tests: $(MODEL_TESTS)

test-blas-fp32: $(T1_BLAS_FP32_TEST)
	$(T1_BLAS_FP32_TEST)

test-blas-bf16: $(T2_BLAS_BF16_TEST)
	$(T2_BLAS_BF16_TEST)

test-blas-fp4: $(T3_BLAS_FP4_TEST)
	$(T3_BLAS_FP4_TEST)

test-blas-matrix: $(T4_BLAS_MATRIX_TEST)
	$(T4_BLAS_MATRIX_TEST)

test-tensor-view: $(T5_TENSOR_VIEW_TEST)
	$(T5_TENSOR_VIEW_TEST)

test-blas: $(BLAS_TESTS)
	$(T1_BLAS_FP32_TEST)
	$(T2_BLAS_BF16_TEST)
	$(T3_BLAS_FP4_TEST)
	$(T4_BLAS_MATRIX_TEST)
	$(T5_TENSOR_VIEW_TEST)
	$(T6_DTYPE_CONVERSION_TEST)
	$(T7_QUANTIZATION_TEST)
	$(T8_QUANTIZATION_ERROR)

test-dram-loopback: $(B0_DRAM_LOOPBACK)
	$(SPIKE) --isa=rv64gc --extension=flexi $(B0_DRAM_LOOPBACK)

test-model: $(MODEL_TESTS)
	@set -e; for test_binary in $(MODEL_TESTS); do echo "Running $$test_binary"; $(SPIKE) --isa=rv64gc --extension=flexi --extension=flexi_hx $$test_binary; done

test-clean:
	rm -f $(BLAS_TESTS) $(MODEL_TESTS)
