#ifndef __LIBFLEXI_FLEXI_H
#define __LIBFLEXI_FLEXI_H

#include <random>
#include <limits>
#include <iostream>
#include <stdexcept>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>

#include "riscv/extension.h"
#include "riscv/rocc.h"
#include "riscv/mmu.h"
#include "riscv/trap.h"

#include "flexi_blas.h"

#define XCUSTOM_ACC 3

typedef uint8_t  byte_t;

typedef enum {
  MAT_A = 0,
  MAT_B = 1,
  MAT_C = 2,
  MAT_D = 3
} mat_type;


#ifndef FLEXI_MXU_DIM
#define FLEXI_MXU_DIM           128
#endif
#define FLEXI_MXU_M_TILE_SIZE   FLEXI_MXU_DIM
#define FLEXI_MXU_N_TILE_SIZE   FLEXI_MXU_DIM
#define FLEXI_MXU_K_TILE_SIZE   FLEXI_MXU_DIM
#define FLEXI_ONC_MEM_SIZE      524288  // 512KB


class flexi_t;

#define MAKECUSTOMFN(opcode) custom ## opcode
#define CUSTOMFN(opcode) MAKECUSTOMFN(opcode)

class flexi_t : public extension_t
{
private:
  byte_t*  onc_mem;
  fnblas_matrix_t mxu_wgt_tile;
  fnblas_matrix_t mxu_psum_tile;

  bool is_mxu_wgt_preloaded = false;
  bool is_mxu_psum_preloaded = false;

public:
  // Constructors
  flexi_t();
  ~flexi_t();

  // RoCC Custom Instruction
  reg_t CUSTOMFN(XCUSTOM_ACC)(rocc_insn_t insn, reg_t xs1, reg_t xs2);

  void read_buffer_from_dram(reg_t addr, byte_t *buffer, const size_t buffer_size);
  void write_buffer_to_dram (reg_t addr, byte_t *buffer, const size_t buffer_size);

  // Methods: Spike Interface for Custom RoCC Commands
  void GEMMPreload(reg_t preload_rs1, reg_t preload_rs2);
  void GEMMExecuteS1(reg_t execute_rs1, reg_t execute_rs2);
  void GEMMFlush(reg_t flush_rs1, reg_t flush_rs2);
  void DMALoad(reg_t load_rs1, reg_t load_rs2);
  void DMAStore(reg_t store_rs1, reg_t store_rs2);

  // Methods: Common Interface for RoCC Extension in Spike
  const char* name() const { return "flexi"; }
  void reset() {};
  void set_processor(processor_t* p) { this->p = p; }
  void user(reg_t dram_addr, reg_t sp_addr) {}

  std::vector<insn_desc_t> get_instructions();
  std::vector<insn_desc_t> get_instructions(const processor_t &p);
  std::vector<disasm_insn_t*> get_disasms();
  std::vector<disasm_insn_t*> get_disasms(const processor_t *p);

private:
  // Attributes: Common Variables for RoCC Extension in Spike
  reg_t cause;
  reg_t aux;
  processor_t* p;

  // Attributes: Custom RoCC Command Function Code 
  const unsigned user_funct = 0;
};

#endif  // __LIBFLEXI_FLEXI_H
