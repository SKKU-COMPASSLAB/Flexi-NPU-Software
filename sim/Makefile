SIM_DIR := sim
SIM_BUILD_DIR := $(SIM_DIR)/build
SIM_FLEXI_DIM32_LIB := $(SIM_BUILD_DIR)/libflexi_dim32.so
SIM_FLEXI_DIM128_LIB := $(SIM_BUILD_DIR)/libflexi_dim128.so
SIM_FLEXI_LIB := $(SIM_BUILD_DIR)/libflexi.so
SIM_FLEXI_HX_LIB := $(SIM_BUILD_DIR)/libflexi_hx.so
SIM_HEADERS := $(wildcard $(SIM_DIR)/*.h)

FLEXI_MXU_DIM ?= 128

ifeq ($(filter $(FLEXI_MXU_DIM),32 128),)
$(error FLEXI_MXU_DIM must be either 32 or 128)
endif

SIM_SELECTED_FLEXI_LIB = $(SIM_BUILD_DIR)/libflexi_dim$(FLEXI_MXU_DIM).so

SIM_CFLAGS := \
	-L$(RISCV)/lib \
	-Wl,-rpath,$(RISCV)/lib \
	-I$(SIM_DIR) \
	$(FLEXI_TENSOR_CFLAGS) \
	-I$(RISCV)/include
SIM_LDFLAGS := $(FLEXI_TENSOR_LDFLAGS)

.PHONY: sim-libs sim-select sim-select-dim32 sim-select-dim128 sim-install-dim32 sim-install-dim128 sim-clean

sim-libs: $(SIM_FLEXI_DIM32_LIB) $(SIM_FLEXI_DIM128_LIB) $(SIM_FLEXI_HX_LIB) sim-select

$(SIM_FLEXI_DIM32_LIB): $(SIM_DIR)/flexi.cc $(SIM_HEADERS) $(TENSOR_LIB_HEADERS) $(FLEXI_TENSOR_HOST_LIB) | check-riscv $(SIM_BUILD_DIR)
	$(CXX) $(SIM_CFLAGS) -DFLEXI_MXU_DIM=32 -shared -o $@ -std=c++2a -fPIC -O3 $< $(SIM_LDFLAGS)

$(SIM_FLEXI_DIM128_LIB): $(SIM_DIR)/flexi.cc $(SIM_HEADERS) $(TENSOR_LIB_HEADERS) $(FLEXI_TENSOR_HOST_LIB) | check-riscv $(SIM_BUILD_DIR)
	$(CXX) $(SIM_CFLAGS) -DFLEXI_MXU_DIM=128 -shared -o $@ -std=c++2a -fPIC -O3 $< $(SIM_LDFLAGS)

sim-select: $(SIM_FLEXI_DIM32_LIB) $(SIM_FLEXI_DIM128_LIB) | $(SIM_BUILD_DIR)
	ln -sfn $(notdir $(SIM_SELECTED_FLEXI_LIB)) $(SIM_FLEXI_LIB)

sim-select-dim32: FLEXI_MXU_DIM=32
sim-select-dim32: sim-select

sim-select-dim128: FLEXI_MXU_DIM=128
sim-select-dim128: sim-select

sim-install-dim32: FLEXI_MXU_DIM=32
sim-install-dim32: install

sim-install-dim128: FLEXI_MXU_DIM=128
sim-install-dim128: install

$(SIM_FLEXI_HX_LIB): $(SIM_DIR)/flexi_hx.cc $(SIM_HEADERS) $(TENSOR_LIB_HEADERS) $(FLEXI_TENSOR_HOST_LIB) | check-riscv $(SIM_BUILD_DIR)
	$(CXX) $(SIM_CFLAGS) -shared -o $@ -std=c++2a -fPIC -O3 $< $(SIM_LDFLAGS)

$(SIM_BUILD_DIR):
	mkdir -p $@

sim-clean:
	rm -rf $(SIM_BUILD_DIR)
