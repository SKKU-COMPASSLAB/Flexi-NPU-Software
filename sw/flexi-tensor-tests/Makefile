FLEXI_NPU_TEST_DIR := $(CURDIR)
FLEXI_NPU_SRC_DIR := $(FLEXI_NPU_TEST_DIR)/srcs
FLEXI_NPU_TEST_SRC_DIR := $(FLEXI_NPU_TEST_DIR)/tests
FLEXI_NPU_BUILD_DIR := $(FLEXI_NPU_TEST_DIR)/build
FLEXI_NPU_BIN_DIR := $(FLEXI_NPU_TEST_DIR)/bin
FLEXI_NPU_OBJ_DIR := $(FLEXI_NPU_BUILD_DIR)/obj
FLEXI_NPU_TEST_OBJ_DIR := $(FLEXI_NPU_BUILD_DIR)/tests
FLEXI_NPU_TENSOR_OBJ_DIR := $(FLEXI_NPU_BUILD_DIR)/tensor
FLEXI_LIBFLEXI_DIR := $(patsubst %/sw/flexi-tensor-tests,%,$(FLEXI_NPU_TEST_DIR))
FLEXI_SW_DIR := $(FLEXI_LIBFLEXI_DIR)/sw
FLEXI_TENSOR_DIR := $(FLEXI_LIBFLEXI_DIR)/tensor_lib
FLEXI_RISCV_TESTS_DIR := $(FLEXI_NPU_TEST_DIR)/riscv-tests
FLEXI_BENCH_COMMON_DIR := $(FLEXI_RISCV_TESTS_DIR)/benchmarks/common

FLEXI_NPU_CC ?= $(RISCV)/bin/riscv64-unknown-elf-gcc
FLEXI_NPU_AR ?= $(RISCV)/bin/riscv64-unknown-elf-ar
FLEXI_NPU_OBJCOPY ?= $(RISCV)/bin/riscv64-unknown-elf-objcopy
PYTHON ?= python3
DISABLE_RVV ?= 0

ifeq ($(DISABLE_RVV),1)
FLEXI_NPU_ARCH := rv64gc
else
FLEXI_NPU_ARCH := rv64gcv
endif

FLEXI_NPU_SOURCES := \
	$(FLEXI_NPU_SRC_DIR)/flexi_npu_backend.c \
	$(FLEXI_NPU_SRC_DIR)/flexi_npu_isa.c \
	$(FLEXI_NPU_SRC_DIR)/flexi_npu_baremetal.c
FLEXI_NPU_OBJECTS := $(patsubst $(FLEXI_NPU_SRC_DIR)/%.c,$(FLEXI_NPU_OBJ_DIR)/%.o,$(FLEXI_NPU_SOURCES))
FLEXI_NPU_HEADERS := $(wildcard $(FLEXI_NPU_SRC_DIR)/*.h)
FLEXI_NPU_LIBRARY := $(FLEXI_NPU_BUILD_DIR)/libflexi_npu_backend.a

FLEXI_NPU_TENSOR_SOURCES := \
	$(FLEXI_TENSOR_DIR)/flexi_blas_common.c \
	$(FLEXI_TENSOR_DIR)/flexi_blas_scalar.c \
	$(FLEXI_TENSOR_DIR)/flexi_blas_vector.c \
	$(FLEXI_TENSOR_DIR)/flexi_blas_matrix.c \
	$(FLEXI_TENSOR_DIR)/flexi_tensor_common.c \
	$(FLEXI_TENSOR_DIR)/flexi_tensor_backend.c \
	$(FLEXI_TENSOR_DIR)/flexi_tensor_op.c
FLEXI_NPU_TENSOR_OBJECTS := $(patsubst $(FLEXI_TENSOR_DIR)/%.c,$(FLEXI_NPU_TENSOR_OBJ_DIR)/%.o,$(FLEXI_NPU_TENSOR_SOURCES))
FLEXI_NPU_TENSOR_HEADERS := $(wildcard $(FLEXI_TENSOR_DIR)/*.h)
FLEXI_NPU_TENSOR_LIBRARY := $(FLEXI_NPU_BUILD_DIR)/libflexi_tensor_baremetal.a

FLEXI_NPU_TEST_SOURCES := \
	$(FLEXI_NPU_TEST_SRC_DIR)/t1_validate_matmul_kernel.c \
	$(FLEXI_NPU_TEST_SRC_DIR)/t2_validate_conv2d_kernel.c \
	$(FLEXI_NPU_TEST_SRC_DIR)/t3_validate_quant_kernel.c \
	$(FLEXI_NPU_TEST_SRC_DIR)/t4_fp4_validate_matmul_kernel.c \
	$(FLEXI_NPU_TEST_SRC_DIR)/t5_fp4_validate_conv2d_kernel.c \
	$(FLEXI_NPU_TEST_SRC_DIR)/t6_fp4_validate_quant_kernel.c
FLEXI_NPU_TEST_NAMES := $(basename $(notdir $(FLEXI_NPU_TEST_SOURCES)))
FLEXI_NPU_TEST_OBJECTS := $(patsubst $(FLEXI_NPU_TEST_SRC_DIR)/%.c,$(FLEXI_NPU_TEST_OBJ_DIR)/%.o,$(FLEXI_NPU_TEST_SOURCES))
FLEXI_NPU_TEST_BINARIES := $(addprefix $(FLEXI_NPU_BIN_DIR)/,$(FLEXI_NPU_TEST_NAMES))
FLEXI_NPU_BENCH_SOURCES := $(wildcard $(FLEXI_BENCH_COMMON_DIR)/*.c) $(wildcard $(FLEXI_BENCH_COMMON_DIR)/*.S)
E1_MLP_GENERATOR := $(FLEXI_NPU_TEST_SRC_DIR)/e1_simple_mlp_gen.py
E1_MLP_DUMP_DIR := $(FLEXI_NPU_BUILD_DIR)/e1_simple_mlp_dump
E1_MLP_DUMP_NAMES := x w1 b1 w2 b2 golden
E1_MLP_DUMP_BINS := $(addprefix $(E1_MLP_DUMP_DIR)/,$(addsuffix .bin,$(E1_MLP_DUMP_NAMES)))
E1_MLP_DUMP_HEADER := $(E1_MLP_DUMP_DIR)/e1_simple_mlp_dump.h
E1_MLP_DUMP_LINKER := $(E1_MLP_DUMP_DIR)/e1_simple_mlp_dump.ld
E1_MLP_DUMP_STAMP := $(E1_MLP_DUMP_DIR)/.generated
E1_MLP_DUMP_OBJECTS := $(addprefix $(E1_MLP_DUMP_DIR)/,$(addsuffix .o,$(E1_MLP_DUMP_NAMES)))

FLEXI_NPU_INCLUDES := -I$(FLEXI_NPU_TEST_DIR) -I$(FLEXI_NPU_SRC_DIR) -I$(FLEXI_TENSOR_DIR) -I$(FLEXI_SW_DIR) -I$(FLEXI_RISCV_TESTS_DIR) -I$(FLEXI_RISCV_TESTS_DIR)/env -I$(FLEXI_BENCH_COMMON_DIR)
FLEXI_NPU_CFLAGS := -std=gnu11 -O2 -Wall -Wextra -Werror -mcmodel=medany -march=$(FLEXI_NPU_ARCH) -Wa,-march=$(FLEXI_NPU_ARCH) -mabi=lp64d -ffreestanding -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns -ffunction-sections -fdata-sections -MMD -MP $(FLEXI_NPU_INCLUDES) -DBAREMETAL=1 -DPREALLOCATE=1 -DMULTITHREAD=1 -DFNBLAS_MEM_DYNAMIC_ENABLE=0
FLEXI_NPU_TEST_CFLAGS := $(FLEXI_NPU_CFLAGS) -Wno-unused-parameter
FLEXI_NPU_LINK_CFLAGS := $(filter-out -MMD -MP,$(FLEXI_NPU_CFLAGS)) -Wno-error
FLEXI_NPU_LDFLAGS := -nostdlib -nostartfiles -static -Wl,--gc-sections -T$(FLEXI_BENCH_COMMON_DIR)/test.ld
FLEXI_NPU_LDLIBS := -lm -lc -lgcc

.DEFAULT_GOAL := all

FLEXI_NPU_FLAG_FILE := $(FLEXI_NPU_BUILD_DIR)/.build-flags
$(shell mkdir -p $(FLEXI_NPU_BUILD_DIR); printf '%s' '$(FLEXI_NPU_ARCH)' | cmp -s - $(FLEXI_NPU_FLAG_FILE) 2>/dev/null || printf '%s' '$(FLEXI_NPU_ARCH)' > $(FLEXI_NPU_FLAG_FILE))

.PHONY: all library tests e1_simple_mlp clean

all: library tests

library: $(FLEXI_NPU_LIBRARY)

tests: $(FLEXI_NPU_TEST_BINARIES)

e1_simple_mlp: $(FLEXI_NPU_BIN_DIR)/e1_simple_mlp

$(FLEXI_NPU_LIBRARY): $(FLEXI_NPU_OBJECTS) | $(FLEXI_NPU_BUILD_DIR)
	$(FLEXI_NPU_AR) rcs $@ $^

$(FLEXI_NPU_TENSOR_LIBRARY): $(FLEXI_NPU_TENSOR_OBJECTS) | $(FLEXI_NPU_BUILD_DIR)
	$(FLEXI_NPU_AR) rcs $@ $^

$(FLEXI_NPU_OBJ_DIR)/%.o: $(FLEXI_NPU_SRC_DIR)/%.c $(FLEXI_NPU_HEADERS) $(FLEXI_NPU_TENSOR_HEADERS) $(FLEXI_NPU_FLAG_FILE) | $(FLEXI_NPU_OBJ_DIR)
	$(FLEXI_NPU_CC) $(FLEXI_NPU_CFLAGS) -c $< -o $@

$(FLEXI_NPU_TENSOR_OBJ_DIR)/%.o: $(FLEXI_TENSOR_DIR)/%.c $(FLEXI_NPU_TENSOR_HEADERS) $(FLEXI_NPU_FLAG_FILE) | $(FLEXI_NPU_TENSOR_OBJ_DIR)
	$(FLEXI_NPU_CC) $(FLEXI_NPU_CFLAGS) -c $< -o $@

$(FLEXI_NPU_TEST_OBJ_DIR)/%.o: $(FLEXI_NPU_TEST_SRC_DIR)/%.c $(FLEXI_NPU_HEADERS) $(FLEXI_NPU_TENSOR_HEADERS) $(FLEXI_NPU_FLAG_FILE) | $(FLEXI_NPU_TEST_OBJ_DIR)
	$(FLEXI_NPU_CC) $(FLEXI_NPU_TEST_CFLAGS) -c $< -o $@

$(E1_MLP_DUMP_STAMP): $(E1_MLP_GENERATOR) | $(E1_MLP_DUMP_DIR)
	$(PYTHON) $< --output-dir $(E1_MLP_DUMP_DIR)


$(FLEXI_NPU_TEST_OBJ_DIR)/e1_simple_mlp.o: $(FLEXI_NPU_TEST_SRC_DIR)/e1_simple_mlp.c $(FLEXI_NPU_HEADERS) $(FLEXI_NPU_TENSOR_HEADERS) $(FLEXI_NPU_FLAG_FILE) $(E1_MLP_DUMP_STAMP) | $(FLEXI_NPU_TEST_OBJ_DIR)
	$(FLEXI_NPU_CC) $(FLEXI_NPU_TEST_CFLAGS) -I$(E1_MLP_DUMP_DIR) -c $< -o $@

$(E1_MLP_DUMP_DIR)/%.o: $(E1_MLP_DUMP_STAMP) | $(E1_MLP_DUMP_DIR)
	$(FLEXI_NPU_OBJCOPY) -I binary -O elf64-littleriscv -B riscv --rename-section .data=.e1_simple_mlp_$*,alloc,load,readonly,data,contents $(E1_MLP_DUMP_DIR)/$*.bin $@

$(FLEXI_NPU_BIN_DIR)/e1_simple_mlp: $(FLEXI_NPU_TEST_OBJ_DIR)/e1_simple_mlp.o $(FLEXI_NPU_OBJ_DIR)/flexi_npu_baremetal.o $(FLEXI_NPU_LIBRARY) $(FLEXI_NPU_TENSOR_LIBRARY) $(FLEXI_NPU_BENCH_SOURCES) $(FLEXI_BENCH_COMMON_DIR)/test.ld $(E1_MLP_DUMP_OBJECTS) $(E1_MLP_DUMP_STAMP) | $(FLEXI_NPU_BIN_DIR)
	$(FLEXI_NPU_CC) $(FLEXI_NPU_LINK_CFLAGS) $(FLEXI_NPU_LDFLAGS) -Wl,-T$(E1_MLP_DUMP_LINKER) $(FLEXI_NPU_TEST_OBJ_DIR)/e1_simple_mlp.o $(FLEXI_NPU_OBJ_DIR)/flexi_npu_baremetal.o $(FLEXI_NPU_BENCH_SOURCES) $(E1_MLP_DUMP_OBJECTS) $(FLEXI_NPU_LIBRARY) $(FLEXI_NPU_TENSOR_LIBRARY) $(FLEXI_NPU_LDLIBS) -o $@

$(FLEXI_NPU_TEST_BINARIES): $(FLEXI_NPU_BIN_DIR)/%: $(FLEXI_NPU_TEST_OBJ_DIR)/%.o $(FLEXI_NPU_OBJ_DIR)/flexi_npu_baremetal.o $(FLEXI_NPU_LIBRARY) $(FLEXI_NPU_TENSOR_LIBRARY) $(FLEXI_NPU_BENCH_SOURCES) $(FLEXI_BENCH_COMMON_DIR)/test.ld | $(FLEXI_NPU_BIN_DIR)
	$(FLEXI_NPU_CC) $(FLEXI_NPU_LINK_CFLAGS) $(FLEXI_NPU_LDFLAGS) $< $(FLEXI_NPU_OBJ_DIR)/flexi_npu_baremetal.o $(FLEXI_NPU_BENCH_SOURCES) $(FLEXI_NPU_LIBRARY) $(FLEXI_NPU_TENSOR_LIBRARY) $(FLEXI_NPU_LDLIBS) -o $@

$(FLEXI_NPU_BUILD_DIR) $(FLEXI_NPU_BIN_DIR) $(FLEXI_NPU_OBJ_DIR) $(FLEXI_NPU_TEST_OBJ_DIR) $(FLEXI_NPU_TENSOR_OBJ_DIR) $(E1_MLP_DUMP_DIR):
	mkdir -p $@

-include $(wildcard $(FLEXI_NPU_OBJ_DIR)/*.d)
-include $(wildcard $(FLEXI_NPU_TEST_OBJ_DIR)/*.d)
-include $(wildcard $(FLEXI_NPU_TENSOR_OBJ_DIR)/*.d)

clean:
	rm -rf $(FLEXI_NPU_BUILD_DIR) $(FLEXI_NPU_BIN_DIR)
