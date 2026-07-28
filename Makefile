ifdef RISCV
$(info Running with RISCV=$(RISCV))
endif

CC := gcc
CXX := g++
AR := ar
RISCV_BIN := $(RISCV)/bin
CC_RV := $(RISCV_BIN)/riscv64-unknown-elf-gcc
AR_RV := $(RISCV_BIN)/riscv64-unknown-elf-ar
OBJDUMP_RV := $(RISCV_BIN)/riscv64-unknown-elf-objdump
SPIKE := $(RISCV_BIN)/spike
.DEFAULT_GOAL := default

include tensor_lib/Makefile
include sim/Makefile
include sw/Makefile
include test/Makefile

.PHONY: default check-riscv install clean

default: check-riscv tensor-libs sim-libs sw

check-riscv:
	@test -n "$(RISCV)" || (echo "RISCV is unset"; exit 1)
	@test -x "$(CC_RV)" || (echo "Missing RISC-V compiler: $(CC_RV)"; exit 1)
	@test -x "$(AR_RV)" || (echo "Missing RISC-V archiver: $(AR_RV)"; exit 1)
	@test -x "$(OBJDUMP_RV)" || (echo "Missing RISC-V objdump: $(OBJDUMP_RV)"; exit 1)
	@test -x "$(SPIKE)" || (echo "Missing Spike ISS: $(SPIKE)"; exit 1)

install: check-riscv sim-libs
	cp $(SIM_FLEXI_LIB) $(RISCV)/lib/
	cp $(SIM_FLEXI_HX_LIB) $(RISCV)/lib/

clean: tensor-lib-clean sim-clean sw-clean test-clean
	rm -f *.o *.so
